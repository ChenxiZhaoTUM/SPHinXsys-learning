/**
 * @file    br_2d_bubble_rising.h
 * @brief   2-D bubble rising benchmark without heat transfer.
 * @details Based on the 2-D rising bubble setup summarized in Zhang, Sun and Ming
 *          (2015): liquid/gas two-phase SPH with gravity, viscosity and surface
 *          tension only. All scalar-transport solvers, wall scalar boundary conditions,
 *          scalar flux output and related post-processing have been removed.
 */
#ifndef BR_2D_BUBBLE_RISING_H
#define BR_2D_BUBBLE_RISING_H

#include "sphinxsys.h"
using namespace SPH;

//----------------------------------------------------------------------
// Geometry and numerical resolution.
// Zhang et al. use the Hysing 2-D rising-bubble benchmark with a
// rectangular tank of width 4R and height 8R, where R = 0.25 m.
//----------------------------------------------------------------------
Real DL = 1.0;                  /**< tank width  [m] = 4R */
Real DH = 2.0;                  /**< tank height [m] = 8R */
Real bubble_radius = 0.25;      /**< initial bubble radius [m] */
Vecd bubble_center(0.5, 0.5);   /**< initial bubble centre [m] */

// 0.02 m gives a 50 x 100 particle layout. Use DH / 200.0 for 100 x 200.
Real resolution_ref = DH / 200.0;
Real BW = resolution_ref * 4.0;

BoundingBox system_domain_bounds(
    Vecd(-BW, -BW),
    Vecd(DL + BW, DH + BW)
);

//----------------------------------------------------------------------
// Material and force parameters for the 2-D benchmark.
//----------------------------------------------------------------------
Real rho0_l = 1000.0;           /**< liquid density [kg/m^3] */
Real rho0_g = 100.0;            /**< gas density    [kg/m^3] */
Real mu_l = 10.0;               /**< liquid dynamic viscosity [Pa s] */
Real mu_g = 1.0;                /**< gas dynamic viscosity    [Pa s] */
Real gravity_g = 0.98;          /**< gravitational acceleration [m/s^2] */
Real surface_tension = 24.5;    /**< surface-tension coefficient [N/m] */

Real U_f = sqrt(gravity_g * DH);
Real c_f = 10.0 * U_f;          /**< weakly-compressible sound speed */

//----------------------------------------------------------------------
// Geometric shapes.
//----------------------------------------------------------------------
std::vector<Vecd> createOuterWall()
{
    std::vector<Vecd> outer_wall_shape;
    outer_wall_shape.push_back(Vecd(-BW, -BW));
    outer_wall_shape.push_back(Vecd(DL + BW, -BW));
    outer_wall_shape.push_back(Vecd(DL + BW, DH + BW));
    outer_wall_shape.push_back(Vecd(-BW, DH + BW));
    outer_wall_shape.push_back(Vecd(-BW, -BW));
    return outer_wall_shape;
}

std::vector<Vecd> createInnerWall()
{
    std::vector<Vecd> inner_wall_shape;
    inner_wall_shape.push_back(Vecd(0.0, 0.0));
    inner_wall_shape.push_back(Vecd(DL, 0.0));
    inner_wall_shape.push_back(Vecd(DL, DH));
    inner_wall_shape.push_back(Vecd(0.0, DH));
    inner_wall_shape.push_back(Vecd(0.0, 0.0));
    return inner_wall_shape;
}

std::vector<Vecd> createBottomWall()
{
    std::vector<Vecd> bottom_wall_shape;
    bottom_wall_shape.push_back(Vecd(-BW, -BW));
    bottom_wall_shape.push_back(Vecd(DL + BW, -BW));
    bottom_wall_shape.push_back(Vecd(DL + BW, 0.0));
    bottom_wall_shape.push_back(Vecd(-BW, 0.0));
    bottom_wall_shape.push_back(Vecd(-BW, -BW));
    return bottom_wall_shape;
}

std::vector<Vecd> createTopWall()
{
    std::vector<Vecd> top_wall_shape;
    top_wall_shape.push_back(Vecd(-BW, DH));
    top_wall_shape.push_back(Vecd(DL + BW, DH));
    top_wall_shape.push_back(Vecd(DL + BW, DH + BW));
    top_wall_shape.push_back(Vecd(-BW, DH + BW));
    top_wall_shape.push_back(Vecd(-BW, DH));
    return top_wall_shape;
}

//----------------------------------------------------------------------
// Application-dependent SPH bodies.
//----------------------------------------------------------------------
namespace SPH
{
class BubbleBody : public MultiPolygonShape
{
  public:
    explicit BubbleBody(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        /** Geometry definition. */
        multi_polygon_.addACircle(bubble_center, bubble_radius, 100, ShapeBooleanOps::add);
    }
};

class LiquidBody : public ComplexShape
{
  public:
    explicit LiquidBody(const std::string &shape_name) : ComplexShape(shape_name)
    {
        MultiPolygon outer_boundary(createInnerWall());
        add<MultiPolygonShape>(outer_boundary, "OuterBoundary");
        MultiPolygon circle(bubble_center, bubble_radius, 100);
        subtract<MultiPolygonShape>(circle);
    }
};

class WallBoundary : public MultiPolygonShape
{
  public:
    explicit WallBoundary(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(createOuterWall(), ShapeBooleanOps::add);
        multi_polygon_.addAPolygon(createInnerWall(), ShapeBooleanOps::sub);
    }
};

class NoSlipWall : public MultiPolygonShape
{
  public:
    explicit NoSlipWall(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(createBottomWall(), ShapeBooleanOps::add);
        multi_polygon_.addAPolygon(createTopWall(), ShapeBooleanOps::add);
    }
};

StdVec<Vecd> createObservationPoints()
{
    // A small fixed vertical probe line through the initial bubble centre.
    StdVec<Vecd> observation_points;
    for (size_t i = 0; i < 5; ++i)
    {
        observation_points.push_back(Vecd(0.5 * DL, 0.25 * DH + Real(i) * 0.125 * DH));
    }
    return observation_points;
}

} // namespace SPH
#endif // BR_2D_BUBBLE_RISING_H
