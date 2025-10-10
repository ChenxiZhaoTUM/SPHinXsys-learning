/**
 * @file 	natural_convection_cylinder_variableTw.h
 * @brief
 * @author
 */
#ifndef DIFFUSION_NEUMANN_BC_H
#define DIFFUSION_NEUMANN_BC_H

#include "sphinxsys.h"
using namespace SPH;

//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Vecd circle_center = Vecd(0., 0.);
Real inner_circle_radius = 0.076;       /**< Radius of the cylinder. */
Real resolution_ref = inner_circle_radius / 20.0;
Real BW = resolution_ref * 4.0;
BoundingBox system_domain_bounds(Vec2d(-inner_circle_radius-BW , -inner_circle_radius-BW ), Vec2d(inner_circle_radius + BW, inner_circle_radius + BW));
//----------------------------------------------------------------------
//	Basic parameters for material properties.
//----------------------------------------------------------------------
Real rho0_f = 7.89;                  /**< Reference density of fluid. */
Real mu_f = 2.35E-6 * rho0_f;               /**< Dynamics viscosity. */
Real C_p = 1.0168E3;
Real diffusion_coeff = 3.28E-6;
Real thermal_expansion_coeff = 3.4E-3;
std::string diffusion_species_name = "Phi";
//----------------------------------------------------------------------
//	Initial and boundary conditions.
//----------------------------------------------------------------------
Real initial_temperature = 25;
Real wall_temperature = 23.4;
Real U_f = sqrt(9.81 * thermal_expansion_coeff * (initial_temperature - wall_temperature) * 2 * inner_circle_radius);                     /**< Characteristic velocity. */
Real c_f = 10.0 * U_f;              /**< Reference sound speed. */
//----------------------------------------------------------------------
// Define extra classes which are used in the main program.
// These classes are defined under the namespace of SPH.
//----------------------------------------------------------------------
namespace SPH
{
//----------------------------------------------------------------------
//	Define SPH bodies.
//----------------------------------------------------------------------
class Cylinder : public MultiPolygonShape
{
  public:
    explicit Cylinder(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addACircle(circle_center, inner_circle_radius, 100, ShapeBooleanOps::add);
    }
};

class Wall : public MultiPolygonShape
{
  public:
    explicit Wall(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addACircle(circle_center, inner_circle_radius + BW, 100, ShapeBooleanOps::add);
        multi_polygon_.addACircle(circle_center, inner_circle_radius, 100, ShapeBooleanOps::sub);
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
          phi_(particles_->registerStateVariable<Real>(diffusion_species_name)),
          physical_time_(sph_system_.getSystemVariableDataByName<Real>("PhysicalTime")) {};

    void update(size_t index_i, Real dt)
    {

        phi_[index_i] = wall_temperature + (0.99-0.05 * (*physical_time_) * sqrt(9.81*thermal_expansion_coeff*diffusion_coeff/inner_circle_radius)) * (initial_temperature - wall_temperature);
    }

  protected:
    Vecd *pos_;
    Real *phi_;
    Real *physical_time_;
};
//----------------------------------------------------------------------
//	Specify diffusion relaxation method.
//----------------------------------------------------------------------
using DiffusionBodyRelaxation = DiffusionBodyRelaxationComplex<
    IsotropicDiffusion, KernelGradientInner, KernelGradientContact, Dirichlet>;
} // namespace SPH
#endif // DIFFUSION_NEUMANN_BC_H