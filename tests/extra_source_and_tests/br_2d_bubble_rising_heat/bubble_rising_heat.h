/**
 * @file    bubble_rising_heat.h
 * @brief   2-D bubble rising benchmark with left/right thermal convection-diffusion and buoyancy.
 */
#ifndef BUBBLE_RISING_HEAT_H
#define BUBBLE_RISING_HEAT_H

#include "sphinxsys.h"
#include <limits>
#include <fstream>
using namespace SPH;

//----------------------------------------------------------------------
// Geometry and numerical resolution.
//----------------------------------------------------------------------
Real DL = 2.0;
Real DH = 2.0;
Real bubble_radius = 0.25;
Vecd bubble_center(DL/2, 0.5);

Real resolution_ref = DH / 200.0;
Real BW = resolution_ref * 4.0;

BoundingBox system_domain_bounds(
    Vecd(-BW, -BW),
    Vecd(DL + BW, DH + BW)
);

//----------------------------------------------------------------------
// Material and force parameters.
//----------------------------------------------------------------------
Real rho0_l = 1000.0;
Real rho0_g = 100.0;
Real mu_l = 10.0;
Real mu_g = 1.0;

Real gravity_g = 0.98;
Real surface_tension = 24.5;
//----------------------------------------------------------------------
// Temperature / scalar transport parameters.
//----------------------------------------------------------------------
std::string diffusion_species_name = "Phi";

Real left_temperature = 1.0;
Real right_temperature = 0.0;
Real delta_T = left_temperature - right_temperature;
Real heat_flux = 0.0;
Real reference_temperature = 0.5 * (left_temperature + right_temperature);

Real C_p_l = 1.0;
Real C_p_g = 1.0;
//----------------------------------------------------------------------
// Target liquid parameters:
// Ra_l = 1.0e4
// Pr_l = 0.71
//----------------------------------------------------------------------
Real Ra_l = 1.0e4;
Real Pr_l = 0.71;
Real L_ref = DH;

Real nu_l = mu_l / rho0_l;
Real thermal_diffusivity_l = nu_l / Pr_l;

Real thermal_expansion_l =
    Ra_l * nu_l * thermal_diffusivity_l /
    (gravity_g * delta_T * pow(L_ref, 3));

//----------------------------------------------------------------------
// Gas thermal parameters: deliberately different from liquid.
// rho0_g and mu_g are fixed, so nu_g is fixed.
// Here choose gas thermal diffusivity twice liquid value,
// and gas thermal expansion half liquid value.
//----------------------------------------------------------------------
Real nu_g = mu_g / rho0_g;

Real thermal_diffusivity_g = 2.0 * thermal_diffusivity_l;
Real thermal_expansion_g = 0.5 * thermal_expansion_l;

// Thermal conductivity k = rho * Cp * alpha
Real k_l = rho0_l * C_p_l * thermal_diffusivity_l;
Real k_g = rho0_g * C_p_g * thermal_diffusivity_g;

//----------------------------------------------------------------------
// Characteristic velocity and sound speed.
//----------------------------------------------------------------------
Real U_f = sqrt(gravity_g * DH);
Real c_f = 10.0 * U_f;

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
    bottom_wall_shape.push_back(Vecd(0.0, -BW));
    bottom_wall_shape.push_back(Vecd(DL, -BW));
    bottom_wall_shape.push_back(Vecd(DL, 0.0));
    bottom_wall_shape.push_back(Vecd(0.0, 0.0));
    bottom_wall_shape.push_back(Vecd(0.0, -BW));
    return bottom_wall_shape;
}

std::vector<Vecd> createTopWall()
{
    std::vector<Vecd> top_wall_shape;
    top_wall_shape.push_back(Vecd(0.0, DH));
    top_wall_shape.push_back(Vecd(DL, DH));
    top_wall_shape.push_back(Vecd(DL, DH + BW));
    top_wall_shape.push_back(Vecd(0.0, DH + BW));
    top_wall_shape.push_back(Vecd(0.0, DH));
    return top_wall_shape;
}

std::vector<Vecd> createLeftDirichletWall()
{
    std::vector<Vecd> left_wall_shape;
    left_wall_shape.push_back(Vecd(-BW, -BW));
    left_wall_shape.push_back(Vecd(0.0, -BW));
    left_wall_shape.push_back(Vecd(0.0, DH + BW));
    left_wall_shape.push_back(Vecd(-BW, DH + BW));
    left_wall_shape.push_back(Vecd(-BW, -BW));
    return left_wall_shape;
}

std::vector<Vecd> createRightDirichletWall()
{
    std::vector<Vecd> right_wall_shape;
    right_wall_shape.push_back(Vecd(DL, -BW));
    right_wall_shape.push_back(Vecd(DL + BW, -BW));
    right_wall_shape.push_back(Vecd(DL + BW, DH + BW));
    right_wall_shape.push_back(Vecd(DL, DH + BW));
    right_wall_shape.push_back(Vecd(DL, -BW));
    return right_wall_shape;
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

class LeftDirichletWall : public MultiPolygonShape
{
  public:
    explicit LeftDirichletWall(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(createLeftDirichletWall(), ShapeBooleanOps::add);
    }
};

class RightDirichletWall : public MultiPolygonShape
{
  public:
    explicit RightDirichletWall(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(createRightDirichletWall(), ShapeBooleanOps::add);
    }
};

//----------------------------------------------------------------------
// Temperature initial conditions.
//----------------------------------------------------------------------
class DiffusionInitialCondition : public LocalDynamics
{
  public:
    explicit DiffusionInitialCondition(SPHBody &sph_body)
        : LocalDynamics(sph_body),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          phi_(particles_->registerStateVariable<Real>(diffusion_species_name))
    {
    }

    void update(size_t index_i, Real dt)
    {
        Real x = pos_[index_i][0];
        x = SMAX(Real(0), SMIN(x, DL));
        phi_[index_i] =
            left_temperature +
            (right_temperature - left_temperature) * x / DL;
    }

  protected:
    Vecd *pos_;
    Real *phi_;
};

class DirichletWallBoundaryInitialCondition : public LocalDynamics
{
  public:
    explicit DirichletWallBoundaryInitialCondition(SPHBody &sph_body)
        : LocalDynamics(sph_body),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          phi_(particles_->registerStateVariable<Real>(diffusion_species_name))
    {
    }

    void update(size_t index_i, Real dt)
    {
        phi_[index_i] = (pos_[index_i][0] < 0.5 * DL)
                            ? left_temperature
                            : right_temperature;
    }

  protected:
    Vecd *pos_;
    Real *phi_;
};

class NeumannWallBoundaryInitialCondition : public LocalDynamics
{
  public:
    explicit NeumannWallBoundaryInitialCondition(SPHBody &sph_body)
        : LocalDynamics(sph_body),
          phi_(particles_->registerStateVariable<Real>(diffusion_species_name)),
          phi_flux_(particles_->getVariableDataByName<Real>(diffusion_species_name + "Flux"))
    {
        this->particles_->template addVariableToRestart<Real>(diffusion_species_name);
    }

    void update(size_t index_i, Real dt)
    {
        phi_flux_[index_i] = heat_flux;
    }

  protected:
    Real *phi_, *phi_flux_;
};

//----------------------------------------------------------------------
// Diffusion relaxation method.
// Order:
// inner, fluid-fluid contact, left Dirichlet, right Dirichlet, top/bottom Neumann.
//----------------------------------------------------------------------
using MultiPhaseDiffusionBodyRelaxation = MultiPhaseDiffusionBodyRelaxationComplex<
    IsotropicThermalDiffusion,
    KernelGradientInner,
    KernelGradientContact,
    Dirichlet,
    Dirichlet,
    Neumann>;

StdVec<Vecd> createObservationPoints()
{
    StdVec<Vecd> observation_points;
    for (size_t i = 0; i < 5; ++i)
    {
        observation_points.push_back(
            Vecd(0.5 * DL, 0.25 * DH + Real(i) * 0.125 * DH)
        );
    }
    return observation_points;
}

} // namespace SPH

#endif // BUBBLE_RISING_HEAT_H