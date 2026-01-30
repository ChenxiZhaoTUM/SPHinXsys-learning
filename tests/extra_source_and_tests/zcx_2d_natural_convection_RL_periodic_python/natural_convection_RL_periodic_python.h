#include "sphinxsys.h"
#include <algorithm>
#include <cmath>

using namespace SPH;

class SphBasicGeometrySetting
{
  protected:
    //------------------------------------------------------------------------------
    // global parameters for the case
    //------------------------------------------------------------------------------
    Real L = 2 * Pi;
    Real H = 2;
    //Real particle_spacing_ref = H / 100.0;
    Real particle_spacing_ref = H / 50.0;
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

    size_t n_seg = 10;
    inline static StdVec<Real> down_wall_segment_T = StdVec<Real>(10, 2.0);
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
    static void setDownWallSegmentTemperatures(const StdVec<Real> &Ts)
    {
        if (!Ts.empty())
        {
            down_wall_segment_T = Ts;
        }
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
        multi_polygon_.addAPolygon(createInnerWall(), ShapeBooleanOps::sub);
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

//----------------------------------------------------------------------
//	Application dependent initial condition.
//----------------------------------------------------------------------
class DiffusionInitialCondition : public LocalDynamics, public SphBasicGeometrySetting
{
  public:
    explicit DiffusionInitialCondition(SPHBody &sph_body)
        : LocalDynamics(sph_body),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          phi_(particles_->registerStateVariable<Real>(diffusion_species_name)) 
    {
        this->particles_->template addVariableToRestart<Real>(diffusion_species_name);
    };

    void update(size_t index_i, Real dt)
    {
        //phi_[index_i] = down_temperature + (up_temperature - down_temperature) * (pos_[index_i][1] + H/2) / H;
        phi_[index_i] = (up_temperature + down_temperature) /2.0;
    };

  protected:
    Vecd *pos_;
    Real *phi_;
};

class DirichletWallBoundaryInitialCondition
    : public LocalDynamics
    , public SphBasicGeometrySetting
{
public:
    explicit DirichletWallBoundaryInitialCondition(SPHBody& sph_body)
        : LocalDynamics(sph_body)
        , pos_(particles_->getVariableDataByName<Vecd>("Position"))
        , phi_(particles_->registerStateVariable<Real>(diffusion_species_name))
    {
        particles_->template addVariableToRestart<Real>(diffusion_species_name);
    }

    void update(size_t index_i, Real dt)
    {
        const Real x = pos_[index_i][0];
        const Real y = pos_[index_i][1];

        // Top wall: uniform Dirichlet
        if (y >= H * Real(0.5))
        {
            phi_[index_i] = up_temperature;
            return;
        }

        // Bottom wall: segmented Dirichlet with smoothing
        if (y > -H * Real(0.5))
            return;

        const size_t n = down_wall_segment_T.size();
        if (n == 0)
            return;

        const Real seg_len = L / Real(n);

        // smoothing half-width = 10% of segment length, clamped
        Real dx = Real(0.1) * seg_len;
        dx = std::min(dx, Real(0.49) * seg_len);
        dx = std::max(dx, Real(0.0));

        // Determine segment index k in [0, n-1]
        // Assumes x in [0, L]. If x can be negative or periodic, apply wrapping before this.
        size_t k = static_cast<size_t>(std::floor(x / seg_len));
        if (k >= n) k = n - 1;

        const size_t kL = (k + n - 1) % n;  // periodic left neighbor
        const size_t kR = (k + 1) % n;      // periodic right neighbor

        const Real xk  = seg_len * Real(k);
        const Real xk1 = xk + seg_len;

        const Real Tk = down_wall_segment_T[k];
        Real T_here = Tk;

        if (dx > Real(0.0))
        {
            const Real dx3 = dx * dx * dx;

            // Left transition band: [xk, xk + dx]
            if (x < xk + dx)
            {
                const Real T_left = down_wall_segment_T[kL];
                const Real t1 = (x - xk - Real(2.0) * dx);
                const Real t2 = (x - xk + dx);
                const Real poly = t1 * t2 * t2;  // (x-xk-2dx)*(x-xk+dx)^2
                T_here = T_left + ((T_left - Tk) / (Real(4.0) * dx3)) * poly;
            }
            // Right transition band: [xk1 - dx, xk1]
            else if (x > xk1 - dx)
            {
                const Real T_right = down_wall_segment_T[kR];
                const Real t1 = (x - xk1 - Real(2.0) * dx);
                const Real t2 = (x - xk1 + dx);
                const Real poly = t1 * t2 * t2;
                T_here = Tk + ((Tk - T_right) / (Real(4.0) * dx3)) * poly;
            }
            // Else plateau region: (xk + dx, xk1 - dx) => Tk
        }

        phi_[index_i] = T_here;
    }

protected:
    Vecd* pos_;
    Real* phi_;
};
