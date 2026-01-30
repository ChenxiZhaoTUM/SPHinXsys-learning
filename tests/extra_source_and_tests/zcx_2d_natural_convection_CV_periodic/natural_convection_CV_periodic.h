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
Real C_p = 1.0;
Real diffusion_coeff = kappa;
std::string diffusion_species_name = "Phi";
//----------------------------------------------------------------------
//	Initial and boundary conditions.
//----------------------------------------------------------------------
Real up_temperature = 1.0;
Real down_temperature = 2.0;
Real thermal_expansion_coeff = 1/ (g * (down_temperature - up_temperature) * pow(H, 3));
Real heat_flux = 0;
Real U_f = sqrt(g * thermal_expansion_coeff * (down_temperature - up_temperature) * H);                     /**< Characteristic velocity. */
Real c_f = 10.0 * U_f;              /**< Reference sound speed. */
//----------------------------------------------------------------------
//	Geometric shapes used in the system.
//----------------------------------------------------------------------
std::vector<Vecd> createThermalDomain()
{
    std::vector<Vecd> thermalDomainShape;
    thermalDomainShape.push_back(Vecd(0.0, -H/2));
    thermalDomainShape.push_back(Vecd(0.0, H/2));
    thermalDomainShape.push_back(Vecd(L, H/2));
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

class WallBoundary : public MultiPolygonShape
{
  public:
    explicit WallBoundary(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(createOuterWall(), ShapeBooleanOps::add);
        multi_polygon_.addAPolygon(createInnerWall(), ShapeBooleanOps::sub);
    }
};

class UpDirichletWallBoundary : public MultiPolygonShape
{
  public:
    explicit UpDirichletWallBoundary(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(createUpWallDomain(), ShapeBooleanOps::add);
    }
};

class DownDirichletWallBoundary : public MultiPolygonShape
{
  public:
    explicit DownDirichletWallBoundary(const std::string &shape_name) : MultiPolygonShape(shape_name)
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

//----------------------------------------------------------------------
//	Specify diffusion relaxation method.
//----------------------------------------------------------------------
using DiffusionBodyRelaxation = DiffusionBodyRelaxationComplex<
    IsotropicDiffusion, KernelGradientInner, KernelGradientContact, Dirichlet, Dirichlet>;

StdVec<Vecd> createObservationPoints()
{
    Real dx = L / 33;
    Real dy = H / 9;
    StdVec<Vecd> observation_points;
    for (size_t i = 0; i < 32; ++i)
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