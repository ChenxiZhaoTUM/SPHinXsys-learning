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

class DirichletWallBoundaryInitialCondition : public LocalDynamics, public SphBasicGeometrySetting
{
  public:
    explicit DirichletWallBoundaryInitialCondition(SPHBody &sph_body)
        : LocalDynamics(sph_body),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          phi_(particles_->registerStateVariable<Real>(diffusion_species_name)) 
    {
        this->particles_->template addVariableToRestart<Real>(diffusion_species_name);
    };

    void update(size_t index_i, Real dt)
    {
        Real x = pos_[index_i][0];
        Real y = pos_[index_i][1];

        // --- Bottom wall (Dirichlet with segmented heating control)
        if (y <= -H / 2.0)
        {
            const size_t n = down_wall_segment_T.size();
            if (n > 0)
            {
                Real seg_len = L / Real(n);

                // choose smoothing half-width as a fraction of seg_len
                Real dx_eff = 0.1 * seg_len;
                Real dx_max = 0.49 * seg_len;
                if (dx_eff > dx_max)
                    dx_eff = dx_max;
                if (dx_eff < Real(0.0))
                    dx_eff = Real(0.0);

                // which segment is this x in?
                // floor(x / seg_len) gives [0, 1, ..., n-1] ideally,
                // but we clamp just in case x == L (right boundary).
                size_t k = static_cast<size_t>(std::floor(x / seg_len));
                if (k >= n)
                    k = n - 1;

                // segment k spans [xk, xk1]
                Real xk  = seg_len * Real(k);
                Real xk1 = xk + seg_len;

                Real Tk = down_wall_segment_T[k];

                bool has_left  = (k > 0);
                bool has_right = (k < n - 1);

                // define interior "plateau" region after cutting off smoothing bands
                Real mid_left  = xk  + (has_left  && dx_eff > 0.0 ? dx_eff : 0.0);
                Real mid_right = xk1 - (has_right && dx_eff > 0.0 ? dx_eff : 0.0);

                Real T_here = Tk; // default plateau value

                // CASE 1: left transition band [xk, xk+dx_eff], only if k>0
                if (has_left && dx_eff > 0.0 && x < (xk + dx_eff))
                {
                    Real T_left = down_wall_segment_T[k - 1];
                    // cubic "ease" from T_left -> Tk over [xk, xk+dx_eff]
                    // formula matches our Python smoothing construction:
                    // expr_left = T_left + ((T_left - Tk)/(4*dx^3)) * (x - xk - 2dx)*(x - xk + dx)^2
                    Real dx = dx_eff;
                    Real term1 = (x - xk - 2.0 * dx);
                    Real term2 = (x - xk + dx);
                    Real poly  = term1 * term2 * term2; // (x - xk - 2dx)*(x - xk + dx)^2
                    T_here = T_left + ((T_left - Tk) / (4.0 * std::pow(dx, 3))) * poly;
                }
                // CASE 2: plateau region [mid_left, mid_right]
                else if (x <= mid_right)
                {
                    // keep T_here = Tk (already set)
                }
                // CASE 3: right transition band [xk1 - dx_eff, xk1], only if k<n-1
                else if (has_right && dx_eff > 0.0)
                {
                    Real T_right = down_wall_segment_T[k + 1];
                    // cubic "ease" from Tk -> T_right over [xk1 - dx_eff, xk1]
                    // expr_right = Tk + ((Tk - T_right)/(4*dx^3)) * (x - xk1 - 2dx)*(x - xk1 + dx)^2
                    Real dx = dx_eff;
                    Real term1 = (x - xk1 - 2.0 * dx);
                    Real term2 = (x - xk1 + dx);
                    Real poly  = term1 * term2 * term2;
                    T_here = Tk + ((Tk - T_right) / (4.0 * std::pow(dx, 3))) * poly;
                }
                else
                {
                    // For completeness. If x > mid_right but !has_right, we just keep Tk.
                    // Similarly if dx_eff == 0 we just keep Tk.
                }

                phi_[index_i] = T_here;
            }
        }
        
        // --- Top wall (Dirichlet, fixed uniform up_temperature)
        if (pos_[index_i][1] >= H/2)
        {
            phi_[index_i] = up_temperature;
        }
    }

  protected:
    Vecd *pos_;
    Real *phi_;
};

