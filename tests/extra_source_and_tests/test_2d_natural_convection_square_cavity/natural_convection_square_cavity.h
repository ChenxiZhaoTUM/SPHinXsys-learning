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
Real L = 0.03628;
Real H = 0.03628;
Real resolution_ref = H / 150.0;
Real BW = resolution_ref * 3.0;
BoundingBox system_domain_bounds(Vec2d(-BW, -BW), Vec2d(L + BW, H + BW));
//----------------------------------------------------------------------
//	Basic parameters for material properties.
//----------------------------------------------------------------------
Real rho0_f = 1.204;                  /**< Reference density of fluid. */
Real U_f = 0.5;                     /**< Characteristic velocity. */
Real c_f = 30.0 * U_f;              /**< Reference sound speed. */
Real mu_f = 1.506E-5 * rho0_f;               /**< Dynamics viscosity. */
Real C_p = 1.006E3;
Real k = 0.02587;
Real diffusion_coeff = k/(rho0_f*C_p);
Real thermal_expansion_coeff = 3.43E-3;
std::string diffusion_species_name = "Phi";
//----------------------------------------------------------------------
//	Initial and boundary conditions.
//----------------------------------------------------------------------
Real initial_temperature = 293.0;
Real left_temperature = 303.0;
Real right_temperature = 283.0;
Real heat_flux = 0;
//----------------------------------------------------------------------
//	Geometric shapes used in the system.
//----------------------------------------------------------------------
std::vector<Vecd> createThermalDomain()
{
    std::vector<Vecd> thermalDomainShape;
    thermalDomainShape.push_back(Vecd(0.0, 0.0));
    thermalDomainShape.push_back(Vecd(0.0, H));
    thermalDomainShape.push_back(Vecd(L, H));
    thermalDomainShape.push_back(Vecd(L, 0.0));
    thermalDomainShape.push_back(Vecd(0.0, 0.0));

    return thermalDomainShape;
}

std::vector<Vecd> left_temperature_region{
    Vecd(-BW, -BW), Vecd(0, -BW), Vecd(0, H + BW),
    Vecd(-BW, H + BW), Vecd(-BW, -BW)};

std::vector<Vecd> right_temperature_region{
    Vecd(L, -BW), Vecd(L + BW, -BW), Vecd(L + BW, H + BW),
    Vecd(L, H + BW), Vecd(L, -BW)};

std::vector<Vecd> up_heat_flux_region{
    Vecd(0, H), Vecd(L, H), Vecd(L, H + BW),
    Vecd(0, H + BW), Vecd(0, H)};

std::vector<Vecd> down_heat_flux_region{
    Vecd(0, -BW), Vecd(L, -BW), Vecd(L, 0),
    Vecd(0, 0), Vecd(0, -BW)};

//----------------------------------------------------------------------
// Define extra classes which are used in the main program.
// These classes are defined under the namespace of SPH.
//----------------------------------------------------------------------
namespace SPH
{
//----------------------------------------------------------------------
//	Define SPH bodies.
//----------------------------------------------------------------------
class DiffusionBody : public MultiPolygonShape
{
  public:
    explicit DiffusionBody(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(createThermalDomain(), ShapeBooleanOps::add);
    }
};

class DirichletWallBoundary : public MultiPolygonShape
{
  public:
    explicit DirichletWallBoundary(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(left_temperature_region, ShapeBooleanOps::add);
        multi_polygon_.addAPolygon(right_temperature_region, ShapeBooleanOps::add);
    }
};

class NeumannWallBoundary : public MultiPolygonShape
{
  public:
    explicit NeumannWallBoundary(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(up_heat_flux_region, ShapeBooleanOps::add);
        multi_polygon_.addAPolygon(down_heat_flux_region, ShapeBooleanOps::add);
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
          phi_(particles_->registerStateVariableData<Real>(diffusion_species_name)) {};

    void update(size_t index_i, Real dt)
    {
        phi_[index_i] = initial_temperature;
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
          phi_(particles_->registerStateVariableData<Real>(diffusion_species_name)) {};

    void update(size_t index_i, Real dt)
    {
        if (pos_[index_i][0] < 0)
        {
            phi_[index_i] = left_temperature;
        }
        if (pos_[index_i][0] > L)
        {
            phi_[index_i] = right_temperature;
        }
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
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          phi_(particles_->registerStateVariableData<Real>(diffusion_species_name)),
          phi_flux_(particles_->getVariableDataByName<Real>(diffusion_species_name + "Flux")) {}

    void update(size_t index_i, Real dt)
    {
        phi_flux_[index_i] = heat_flux;
    }

  protected:
    Vecd *pos_;
    Real *phi_, *phi_flux_;
};
//----------------------------------------------------------------------
//	Specify diffusion relaxation method.
//----------------------------------------------------------------------
using DiffusionBodyRelaxation = DiffusionBodyRelaxationComplex<
    IsotropicDiffusion, KernelGradientInner, KernelGradientContact, Dirichlet, Neumann>;

//StdVec<Vecd> createObservationPoints()
//{
//    StdVec<Vecd> observation_points;
//    /** A line of measuring points at the middle line. */
//    size_t number_of_observation_points = 5;
//    Real range_of_measure = L;
//    Real start_of_measure = 0;
//
//    for (size_t i = 0; i < number_of_observation_points; ++i)
//    {
//        Vec2d point_coordinate(0.5 * L, range_of_measure * Real(i) /
//                                                Real(number_of_observation_points - 1) +
//                                            start_of_measure);
//        observation_points.push_back(point_coordinate);
//    }
//    return observation_points;
//};
} // namespace SPH
#endif // DIFFUSION_NEUMANN_BC_H