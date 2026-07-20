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
Real L = 2.0;
Real H = 1.0;

Real resolution_ref = H / 50.0;
Real BW = resolution_ref * 3.0;

BoundingBox system_domain_bounds(
    Vec2d(-BW, -H / 2.0 - BW),
    Vec2d(L + BW,  H / 2.0 + BW)
);
//----------------------------------------------------------------------
// Temperature
//----------------------------------------------------------------------
Real up_temperature   = 1.0;
Real down_temperature = 2.0;
Real delta_T = down_temperature - up_temperature;
//----------------------------------------------------------------------
// Literature-like non-dimensional parameters
//----------------------------------------------------------------------
const Real g = 9.81;

const Real Ra1 = 8.0e4;
const Real Pr1 = 1.0;          
const Real epsilon1 = 0.15;    // epsilon1 = beta1 * delta_T
const Real Ca1 = 4.6e-4;

// Property ratios from Chang & Alexander
const Real rho_ratio  = 0.33;  // rho2 / rho1
const Real beta_ratio = 2.0;   // beta2 / beta1
const Real k_ratio    = 0.7;   // k2 / k1
const Real cp_ratio   = 0.4;   // cv2 / cv1
const Real nu_ratio   = 1.0;   // nu2 / nu1

//----------------------------------------------------------------------
// Phase 1: lower heavy fluid
// Phase 2: upper light fluid
//----------------------------------------------------------------------

Real rho0_f_one = 1000.0;
Real rho0_f_two = rho_ratio * rho0_f_one;  // 0.33

Real thermal_expansion_one = epsilon1 / delta_T;                 // 0.15
Real thermal_expansion_two = beta_ratio * thermal_expansion_one; // 0.30

// Literature Rayleigh number based on total height H
// Ra1 = g * beta1 * delta_T * H^3 / (nu1 * kappa1)
// Pr1 = nu1 / kappa1
Real nu_one = sqrt(g * thermal_expansion_one * delta_T * pow(H, 3) * Pr1 / Ra1);
Real kappa_one = nu_one / Pr1;

Real nu_two = nu_ratio * nu_one;

Real mu_f_one = rho0_f_one * nu_one;
Real mu_f_two = rho0_f_two * nu_two;

Real C_p_one = 1.0;
Real C_p_two = cp_ratio * C_p_one;

// Thermal conductivity ratio from paper: k2 / k1 = 0.7
Real k_one = rho0_f_one * C_p_one * kappa_one;
Real k_two = k_ratio * k_one;

// For reference only:
Real kappa_two = k_two / (rho0_f_two * C_p_two);

// If your diffusion solver still needs a scalar diffusion_coeff,
// use the lower-layer thermal diffusivity as reference.
Real diffusion_coeff = kappa_one;

std::string diffusion_species_name = "Phi";
Real heat_flux = 0.0;

// Characteristic velocity used in Ca definition
Real U_f = sqrt(g * thermal_expansion_one * delta_T * H);
Real c_f = 10.0 * U_f;

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