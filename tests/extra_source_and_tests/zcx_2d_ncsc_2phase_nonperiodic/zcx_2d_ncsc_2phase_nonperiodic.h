/**
 * @file 	natural_convection_square_cavity.h
 * @brief 	This is the head files used by diffusion_NeumannBC.cpp.
 * @author	Chenxi Zhao, Bo Zhang, Chi Zhang and Xiangyu Hu
 */
#ifndef NATURAL_CONVECTION_CV_H
#define NATURAL_CONVECTION_CV_H

#include "sphinxsys.h"
using namespace SPH;

//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real L = 4;
Real H = 2;
Real resolution_ref = H / 50;
Real BW = resolution_ref * 3.0;
BoundingBox system_domain_bounds(Vec2d(-BW, - H/2 -BW), Vec2d(L + BW, H/2 + BW));
//----------------------------------------------------------------------
//	Basic parameters for material properties.
//----------------------------------------------------------------------
const Real Ra = 8.0e4;      // use 8e4 first; 1e5 also OK
const Real Pr = 0.71;       // keep your original value; use 1.0 if matching paper figures more strictly
const Real g  = 9.81;

const Real epsilon = 0.15;  // epsilon = beta * delta_T, literature validation value
const Real Ca = 4.6e-4;     // literature validation value

Real thermal_expansion_one = epsilon / 1.0;
Real thermal_expansion_two = thermal_expansion_one; 
// If upper-layer circulation is still too weak, try:
// Real thermal_expansion_two = 1.2 * thermal_expansion_one;
// but for matching the paper, keep them equal.

// Recompute nu and kappa from the actual beta, instead of using sqrt(Pr/Ra)
// Ra = g * beta * delta_T * H^3 / (nu * kappa), Pr = nu / kappa
Real nu = sqrt(g * thermal_expansion_one * 1.0 * pow(H, 3) * Pr / Ra);
Real kappa = nu / Pr;
Real diffusion_coeff = kappa;

//----------------------------------------------------------------------
//  Material properties
//  Literature-like density ratio: rho_upper / rho_lower = 0.33
//----------------------------------------------------------------------

Real rho0_f_one = 1000.0;     // lower heavy phase
Real rho0_f_two = 330.0;      // upper light phase, density ratio = 0.33

Real nu_one = nu;
Real nu_two = nu;             // keep same kinematic viscosity first

Real mu_f_one = rho0_f_one * nu_one;
Real mu_f_two = rho0_f_two * nu_two;

Real C_p_one = 1.0;
Real C_p_two = 1.0;

// Keep the same thermal diffusivity in both phases:
// alpha_i = k_i / (rho_i * Cp_i) = kappa
Real k_one = rho0_f_one * C_p_one * kappa;
Real k_two = rho0_f_two * C_p_two * kappa;

std::string diffusion_species_name = "Phi";
//----------------------------------------------------------------------
//	Initial and boundary conditions.
//----------------------------------------------------------------------
Real up_temperature = 1.0;
Real down_temperature = 2.0;
Real heat_flux = 0;
Real U_f = sqrt(g * thermal_expansion_two * (down_temperature - up_temperature) * H);   /**< Characteristic velocity. */
Real c_f = 10.0 * U_f;              /**< Reference sound speed. */
//----------------------------------------------------------------------
//	Geometric shapes used in the system.
//----------------------------------------------------------------------
std::vector<Vecd> createThermalDomainOne()
{
    std::vector<Vecd> thermalDomainShape;
    thermalDomainShape.push_back(Vecd(0.0, -H/2));
    thermalDomainShape.push_back(Vecd(0.0, 0.0));
    thermalDomainShape.push_back(Vecd(L, 0.0));
    thermalDomainShape.push_back(Vecd(L, -H/2));
    thermalDomainShape.push_back(Vecd(0.0, -H/2));

    return thermalDomainShape;
}

std::vector<Vecd> createThermalDomainTwo()
{
    std::vector<Vecd> thermalDomainShape;
    thermalDomainShape.push_back(Vecd(0.0, 0.0));
    thermalDomainShape.push_back(Vecd(0.0, H/2));
    thermalDomainShape.push_back(Vecd(L, H/2));
    thermalDomainShape.push_back(Vecd(L, 0.0));
    thermalDomainShape.push_back(Vecd(0.0, 0.0));

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

std::vector<Vecd> LeftDiffusionDomain
{
    Vecd(0., -H / 2), Vecd(L / 3, -H / 2), Vecd(L / 3, H / 2),
    Vecd(0.,  H / 2), Vecd(0., -H / 2)
};

MultiPolygon createLeftDiffusionDomain()
{
    MultiPolygon multi_polygon;
    multi_polygon.addAPolygon(LeftDiffusionDomain, ShapeBooleanOps::add);
    return multi_polygon;
}

std::vector<Vecd> MiddleDiffusionDomain
{
    Vecd(L / 3, -H / 2), Vecd(2 * L / 3, -H / 2), Vecd(2 * L / 3, H / 2),
    Vecd(L / 3,  H / 2), Vecd(L / 3, -H / 2)
};

MultiPolygon createMiddleDiffusionDomain()
{
    MultiPolygon multi_polygon;
    multi_polygon.addAPolygon(MiddleDiffusionDomain, ShapeBooleanOps::add);
    return multi_polygon;
}

std::vector<Vecd> RightDiffusionDomain
{
     Vecd(2 * L / 3, -H / 2), Vecd(L, -H / 2), Vecd(L, H / 2),
     Vecd(2 * L / 3,  H / 2), Vecd(2 * L / 3, -H / 2)
};

MultiPolygon createRightDiffusionDomain()
{
    MultiPolygon multi_polygon;
    multi_polygon.addAPolygon(RightDiffusionDomain, ShapeBooleanOps::add);
    return multi_polygon;
}

std::vector<Vecd> LeftDownWallDomain
{
    Vecd(0., -H/2 - BW), Vecd(L / 3, -H/2 - BW), Vecd(L / 3, -H/2),
    Vecd(0.,  -H/2), Vecd(0., -H/2 - BW)
};

MultiPolygon createLeftDownWallDomain()
{
    MultiPolygon multi_polygon;
    multi_polygon.addAPolygon(LeftDownWallDomain, ShapeBooleanOps::add);
    return multi_polygon;
}

std::vector<Vecd> MiddleDownWallDomain
{
    Vecd(L / 3, -H/2 - BW), Vecd(2 * L / 3, -H/2 - BW), Vecd(2 * L / 3, -H/2),
    Vecd(L / 3,  -H/2), Vecd(L / 3, -H/2 - BW)
};

MultiPolygon createMiddleDownWallDomain()
{
    MultiPolygon multi_polygon;
    multi_polygon.addAPolygon(MiddleDownWallDomain, ShapeBooleanOps::add);
    return multi_polygon;
}

std::vector<Vecd> RightDownWallDomain
{
    Vecd(2 * L / 3, -H/2 - BW), Vecd(L, -H/2 - BW), Vecd(L, -H/2),
    Vecd(2 * L / 3,  -H/2), Vecd(2 * L / 3, -H/2 - BW)
};

MultiPolygon createRightDownWallDomain()
{
    MultiPolygon multi_polygon;
    multi_polygon.addAPolygon(RightDownWallDomain, ShapeBooleanOps::add);
    return multi_polygon;
}

std::vector<Vecd> createLeftWallDomain()
{
    std::vector<Vecd> LeftWallDomainShape;
    LeftWallDomainShape.push_back(Vecd(0. - BW, -H/2 - BW));
    LeftWallDomainShape.push_back(Vecd(0., -H/2 - BW));
    LeftWallDomainShape.push_back(Vecd(0., H/2 + BW));
    LeftWallDomainShape.push_back(Vecd(0. - BW, H/2 + BW));
    LeftWallDomainShape.push_back(Vecd(0. - BW, -H/2 - BW));

    return LeftWallDomainShape;
}

std::vector<Vecd> createRightWallDomain()
{
    std::vector<Vecd> RightWallDomainShape;
    RightWallDomainShape.push_back(Vecd(L, -H/2 - BW));
    RightWallDomainShape.push_back(Vecd(L + BW, -H/2 - BW));
    RightWallDomainShape.push_back(Vecd(L + BW, H/2 + BW));
    RightWallDomainShape.push_back(Vecd(L, H/2+BW));
    RightWallDomainShape.push_back(Vecd(L, -H/2 - BW));

    return RightWallDomainShape;
}

class NeumannBoundary : public MultiPolygonShape
{
  public:
    explicit NeumannBoundary(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(createLeftWallDomain(), ShapeBooleanOps::add);
        multi_polygon_.addAPolygon(createRightWallDomain(), ShapeBooleanOps::add);
    }
};

//----------------------------------------------------------------------
// Define extra classes which are used in the main program.
// These classes are defined under the namespace of SPH.
//----------------------------------------------------------------------
namespace SPH
{
//----------------------------------------------------------------------
//	Define SPH bodies.
//----------------------------------------------------------------------
class PhaseOneDiffusionBody : public MultiPolygonShape
{
  public:
    explicit PhaseOneDiffusionBody(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(createThermalDomainOne(), ShapeBooleanOps::add);
    }
};

class PhaseTwoDiffusionBody : public MultiPolygonShape
{
  public:
    explicit PhaseTwoDiffusionBody(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(createThermalDomainTwo(), ShapeBooleanOps::add);
    }
};

/* non-periodic boundary */
class WallBoundary : public MultiPolygonShape
{
  public:
    explicit WallBoundary(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(createOuterWall(), ShapeBooleanOps::add);
        multi_polygon_.addAPolygon(createThermalDomainOne(), ShapeBooleanOps::sub);
        multi_polygon_.addAPolygon(createThermalDomainTwo(), ShapeBooleanOps::sub);
    }
};

class UpDirichlet : public MultiPolygonShape
{
  public:
    explicit UpDirichlet(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(createUpWallDomain(), ShapeBooleanOps::add);
    }
};

class DownDirichlet : public MultiPolygonShape
{
  public:
    explicit DownDirichlet(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
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
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          phi_(particles_->registerStateVariable<Real>(diffusion_species_name)) {};

    void update(size_t index_i, Real dt)
    {
        //phi_[index_i] = down_temperature + (up_temperature - down_temperature) * (pos_[index_i][1] + H/2) / H;
        phi_[index_i] = (up_temperature + down_temperature) /2.0;
    };

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

class NeumannWallBoundaryInitialCondition : public LocalDynamics
{
  public:
    explicit NeumannWallBoundaryInitialCondition(SPHBody &sph_body)
        : LocalDynamics(sph_body),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
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
    Vecd *pos_;
    Real *phi_, *phi_flux_;
};

//----------------------------------------------------------------------
//	Specify diffusion relaxation method.
//----------------------------------------------------------------------
using MultiPhaseDiffusionBodyRelaxation = MultiPhaseDiffusionBodyRelaxationComplex<
    IsotropicThermalDiffusion, KernelGradientInner, KernelGradientContact, Dirichlet, Dirichlet, Neumann>;

StdVec<Vecd> createObservationPoints()
{
    Real dx = L / 31;
    Real dy = H / 9;
    StdVec<Vecd> observation_points;
    for (size_t i = 0; i < 30; ++i)
    {
        for (size_t j = 0; j < 8; ++j)
        {
            Real x_i = (i + 0.5) * dx;
            Real y_i = H / 2 - (j + 0.5) * dy;
            observation_points.push_back(Vecd(x_i, y_i));
        }
    }
    return observation_points;
}

} // namespace SPH
#endif  // NATURAL_CONVECTION_CV_H