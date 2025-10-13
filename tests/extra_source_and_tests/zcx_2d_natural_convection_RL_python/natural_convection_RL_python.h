#include "sphinxsys.h"

using namespace SPH;

class SphBasicGeometrySetting
{
  protected:
    //------------------------------------------------------------------------------
    // global parameters for the case
    //------------------------------------------------------------------------------
    Real L = 2 * Pi;
    Real H = 2;
    Real particle_spacing_ref = H / 100.0;
    Real BW = particle_spacing_ref * 3.0;
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
    inline static std::array<Real, 3> down_wall_segment_T = {2.0 * 1.2, 2.0 * 1.0, 2.0 * 0.8};
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
    
    // 8 * 30 probes
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
  public:
    // --- NEW: setters to update the three segment temperatures ---
    static void setDownWallSegmentTemperatures(Real T_left, Real T_middle, Real T_right)
    {
        down_wall_segment_T = {T_left, T_middle, T_right};
    }
    static void setDownWallSegmentTemperatures(const StdVec<Real> &Ts)
    {
        if (Ts.size() >= 3) down_wall_segment_T = {Ts[0], Ts[1], Ts[2]};
    }
};
//------------------------------------------------------------------------------
// geometric shapes for the bodies used in the case
//------------------------------------------------------------------------------
class DiffusionBody : public MultiPolygonShape, public SphBasicGeometrySetting
{
  public:
    explicit DiffusionBody(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(createThermalDomain(), ShapeBooleanOps::add);
    }
};

class WallBoundary : public MultiPolygonShape, public SphBasicGeometrySetting
{
  public:
    explicit WallBoundary(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(createOuterWall(), ShapeBooleanOps::add);
        multi_polygon_.addAPolygon(createThermalDomain(), ShapeBooleanOps::sub);
    }
};

class DirichletWallBoundary : public MultiPolygonShape, public SphBasicGeometrySetting
{
  public:
    explicit DirichletWallBoundary(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(createUpWallDomain(), ShapeBooleanOps::add);
        multi_polygon_.addAPolygon(createDownWallDomain(), ShapeBooleanOps::add);
    }
};

class UpDirichletWallBoundary : public MultiPolygonShape, public SphBasicGeometrySetting
{
  public:
    explicit UpDirichletWallBoundary(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(createUpWallDomain(), ShapeBooleanOps::add);
    }
};

class DownDirichletWallBoundary : public MultiPolygonShape, public SphBasicGeometrySetting
{
  public:
    explicit DownDirichletWallBoundary(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(createDownWallDomain(), ShapeBooleanOps::add);
    }
};

class NeumannWallBoundary : public MultiPolygonShape, public SphBasicGeometrySetting
{
  public:
    explicit NeumannWallBoundary(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(createLeftWallDomain(), ShapeBooleanOps::add);
        multi_polygon_.addAPolygon(createRightWallDomain(), ShapeBooleanOps::add);
    }
};

//----------------------------------------------------------------------
//	Application dependent initial condition.
//----------------------------------------------------------------------
class DiffusionInitialCondition : public LocalDynamics, public SphBasicGeometrySetting
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

class DirichletWallBoundaryInitialCondition : public LocalDynamics, public SphBasicGeometrySetting
{
  public:
    explicit DirichletWallBoundaryInitialCondition(SPHBody &sph_body)
        : LocalDynamics(sph_body),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          phi_(particles_->registerStateVariable<Real>(diffusion_species_name)) {};

    void update(size_t index_i, Real dt)
    {
        if (pos_[index_i][1] <= -H/2 && pos_[index_i][0] < L/3)
        {
            phi_[index_i] = down_wall_segment_T[0];
        }
        else if (pos_[index_i][1] <= -H / 2 && pos_[index_i][0] >= L / 3 && pos_[index_i][0] < 2 * L / 3)
        {
            phi_[index_i] = down_wall_segment_T[1];
        }
        else if (pos_[index_i][1] <= -H / 2 && pos_[index_i][0] >= 2 * L / 3)
        {
            phi_[index_i] = down_wall_segment_T[2];
        }

        if (pos_[index_i][1] >= H/2)
        {
            phi_[index_i] = up_temperature;
        }
    }

  protected:
    Vecd *pos_;
    Real *phi_;
};

class NeumannWallBoundaryInitialCondition : public LocalDynamics, public SphBasicGeometrySetting
{
  public:
    explicit NeumannWallBoundaryInitialCondition(SPHBody &sph_body)
        : LocalDynamics(sph_body),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          phi_(particles_->registerStateVariable<Real>(diffusion_species_name)),
          phi_flux_(particles_->getVariableDataByName<Real>(diffusion_species_name + "Flux")) {}

    void update(size_t index_i, Real dt)
    {
        phi_flux_[index_i] = heat_flux;
    }

  protected:
    Vecd *pos_;
    Real *phi_, *phi_flux_;
};
