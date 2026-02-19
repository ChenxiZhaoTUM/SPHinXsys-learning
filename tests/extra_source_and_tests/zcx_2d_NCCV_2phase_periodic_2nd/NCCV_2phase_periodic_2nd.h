/**
 * @file 	natural_convection_square_cavity.h
 * @brief 	This is the head files used by diffusion_NeumannBC.cpp.
 * @author	Chenxi Zhao, Bo Zhang, Chi Zhang and Xiangyu Hu
 */
#ifndef DIFFUSION_NEUMANN_BC_H
#define DIFFUSION_NEUMANN_BC_H

#include "sphinxsys.h"
using namespace SPH;

//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real L = 2 * Pi;
Real H = 2;
Real resolution_ref = H / 50;
Real BW = resolution_ref * 3.0;
BoundingBox system_domain_bounds(Vec2d(-BW, - H/2 -BW), Vec2d(L + BW, H/2 + BW));
//----------------------------------------------------------------------
//	Basic parameters for material properties.
//----------------------------------------------------------------------
const Real Ra = 1.0e4;
const Real Pr = 0.71;
const Real nu  = sqrt(Pr / Ra);
const Real kappa  = 1.0 / sqrt(Pr * Ra);  // thermal diffusivity
const Real g = 9.81;
Real rho0_f = 1.0;                  /**< Reference density of fluid. */
Real mu_f = rho0_f * nu;               /**< Dynamics viscosity. */
//Real mu_f_one = mu_f * 10;
//Real mu_f_two = mu_f * 0.1;
Real mu_f_one = mu_f;
Real mu_f_two = mu_f;
Real C_p = 1.0; 
Real diffusion_coeff = kappa;
Real k = diffusion_coeff * (rho0_f * C_p);
std::string diffusion_species_name = "Phi";
//----------------------------------------------------------------------
//	Initial and boundary conditions.
//----------------------------------------------------------------------
Real up_temperature = 1.0;
Real down_temperature = 2.0;
Real thermal_expansion_coeff = 1/ (g * (down_temperature - up_temperature) * pow(H, 3));
Real heat_flux = 0;
Real U_f = sqrt(g * thermal_expansion_coeff * (down_temperature - up_temperature) * H);   /**< Characteristic velocity. */
Real c_f = 10.0 * U_f;              /**< Reference sound speed. */
//----------------------------------------------------------------------
//	Geometric shapes used in the system.
//----------------------------------------------------------------------
std::vector<Vecd> createThermalDomainUp()
{
    std::vector<Vecd> thermalDomainShape;
    thermalDomainShape.push_back(Vecd(0.0, 0.0));
    thermalDomainShape.push_back(Vecd(0.0, H/2));
    thermalDomainShape.push_back(Vecd(L, H/2));
    thermalDomainShape.push_back(Vecd(L, 0.0));
    thermalDomainShape.push_back(Vecd(0.0, 0.0));

    return thermalDomainShape;
}

std::vector<Vecd> createThermalDomainDown()
{
    std::vector<Vecd> thermalDomainShape;
    thermalDomainShape.push_back(Vecd(0.0, -H/2));
    thermalDomainShape.push_back(Vecd(0.0, 0.0));
    thermalDomainShape.push_back(Vecd(L, 0.0));
    thermalDomainShape.push_back(Vecd(L, -H/2));
    thermalDomainShape.push_back(Vecd(0.0, -H/2));

    return thermalDomainShape;
}

std::vector<Vecd> createOuterWall()
{
    std::vector<Vecd> OuterWallShape;
    OuterWallShape.push_back(Vecd(-BW, -H/2 - BW));
    OuterWallShape.push_back(Vecd(L + BW, -H/2 - BW));
    OuterWallShape.push_back(Vecd(L + BW, H/2 + BW));
    OuterWallShape.push_back(Vecd(-BW, H/2 + BW));
    OuterWallShape.push_back(Vecd(-BW, -H/2 - BW));

    return OuterWallShape;
}

std::vector<Vecd> createInnerWall()
{
    std::vector<Vecd> InnerWallShape;
    InnerWallShape.push_back(Vecd(-BW, -H/2));
    InnerWallShape.push_back(Vecd(L + BW, -H/2));
    InnerWallShape.push_back(Vecd(L + BW, H/2));
    InnerWallShape.push_back(Vecd(-BW, H/2));
    InnerWallShape.push_back(Vecd(-BW, -H/2));

    return InnerWallShape;
}

std::vector<Vecd> createUpWallDomain()
{
    std::vector<Vecd> UpWallDomainShape;
    UpWallDomainShape.push_back(Vecd(0., H/2));
    UpWallDomainShape.push_back(Vecd(L, H/2));
    UpWallDomainShape.push_back(Vecd(L, H/2+BW));
    UpWallDomainShape.push_back(Vecd(0., H/2+BW));
    UpWallDomainShape.push_back(Vecd(0., H/2));

    return UpWallDomainShape;
}

std::vector<Vecd> createDownWallDomain()
{
    std::vector<Vecd> DownWallDomainShape;
    DownWallDomainShape.push_back(Vecd(0., -H/2 - BW));
    DownWallDomainShape.push_back(Vecd(L, -H/2 - BW));
    DownWallDomainShape.push_back(Vecd(L, -H/2));
    DownWallDomainShape.push_back(Vecd(0., -H/2));
    DownWallDomainShape.push_back(Vecd(0., -H/2 - BW));

    return DownWallDomainShape;
}

//----------------------------------------------------------------------
// Define extra classes which are used in the main program.
// These classes are defined under the namespace of SPH.
//----------------------------------------------------------------------
namespace SPH
{
//----------------------------------------------------------------------
//	Define SPH bodies.
//----------------------------------------------------------------------
class UpPhaseDiffusionBody : public MultiPolygonShape
{
  public:
    explicit UpPhaseDiffusionBody(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(createThermalDomainUp(), ShapeBooleanOps::add);
    }
};

class DownPhaseDiffusionBody : public MultiPolygonShape
{
  public:
    explicit DownPhaseDiffusionBody(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(createThermalDomainDown(), ShapeBooleanOps::add);
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

class DirichletWallBoundary : public MultiPolygonShape
{
  public:
    explicit DirichletWallBoundary(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(createUpWallDomain(), ShapeBooleanOps::add);
        multi_polygon_.addAPolygon(createDownWallDomain(), ShapeBooleanOps::add);
    }
};

//----------------------------------------------------------------------
//	Application dependent initial condition.
//----------------------------------------------------------------------
class DiffusionInitialCondition : public LocalDynamics
{
  public:
    explicit DiffusionInitialCondition(SPHBody &sph_body)
        : LocalDynamics(sph_body),
          phi_(particles_->registerStateVariable<Real>(diffusion_species_name)) {};

    void update(size_t index_i, Real dt)
    {
        phi_[index_i] = (up_temperature + down_temperature) /2.0;
    };

  protected:
    Real *phi_;
};

class DirichletWallBoundaryInitialCondition : public LocalDynamics
{
  public:
    explicit DirichletWallBoundaryInitialCondition(SPHBody &sph_body)
        : LocalDynamics(sph_body),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          phi_(particles_->registerStateVariable<Real>(diffusion_species_name)) {};

    void update(size_t index_i, Real dt)
    {
        if (pos_[index_i][1] >= H/2)
        {
            phi_[index_i] = up_temperature;
        }
        else
        {
            phi_[index_i] = down_temperature;
        }
    }

  protected:
    Vecd *pos_;
    Real *phi_;
};

//----------------------------------------------------------------------
//	Specify diffusion relaxation method.
//----------------------------------------------------------------------
using MultiPhaseDiffusionBodyRelaxation = MultiPhaseDiffusionBodyRelaxationComplex<
    IsotropicThermalDiffusion, KernelGradientInner, KernelGradientContact, Dirichlet>;

StdVec<Vecd> createVerticalVelObservationPoints()
{
    StdVec<Vecd> observation_points;
    /** A line of measuring points at the middle line. */
    size_t number_of_observation_points = 21;
    Real range_of_measure = L;
    Real start_of_measure = 0;

    for (size_t i = 0; i < number_of_observation_points; ++i)
    {
        Vec2d point_coordinate(range_of_measure * Real(i) / Real(number_of_observation_points - 1) + start_of_measure, H/2);
        observation_points.push_back(point_coordinate);
    }
    return observation_points;
};

StdVec<Vecd> createHorizontalVelObservationPoints()
{
    StdVec<Vecd> observation_points;
    /** A line of measuring points at the middle line. */
    size_t number_of_observation_points = 21;
    Real range_of_measure = H;
    Real start_of_measure = 0;

    for (size_t i = 0; i < number_of_observation_points; ++i)
    {
        Vec2d point_coordinate(L/2, range_of_measure * Real(i) / Real(number_of_observation_points - 1) + start_of_measure);
        observation_points.push_back(point_coordinate);
    }
    return observation_points;
};
} // namespace SPH
#endif // DIFFUSION_NEUMANN_BC_H