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
    Real L = 2.0;
    Real H = 1.0;
    Real particle_spacing_ref = H / 50.0;
    Real BW = particle_spacing_ref * 3.0;
    //----------------------------------------------------------------------
    // Temperature
    //----------------------------------------------------------------------
    Real up_temperature   = 1.0;
    Real down_temperature = 2.0;
    Real delta_T = down_temperature - up_temperature;
    //----------------------------------------------------------------------
    //	Basic parameters for material properties.
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

    Real rho0_f_one = 1000.0;
    Real rho0_f_two = rho_ratio * rho0_f_one;  // 0.33

    Real thermal_expansion_one = epsilon1 / delta_T;                 // 0.15
    Real thermal_expansion_two = beta_ratio * thermal_expansion_one; // 0.30

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


    size_t n_seg = 4;
    inline static StdVec<Real> down_wall_segment_T = StdVec<Real>(4, 2.0);
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
    
    // 15 * 10 probes
    StdVec<Vecd> createObservationPoints()
    {
        Real dx = L / 16;
        Real dy = H / 11;
        StdVec<Vecd> observation_points;
        for (size_t i = 0; i < 15; ++i)
        {
            for (size_t j = 0; j < 10; ++j)
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
class PhaseOneDiffusionBody : public MultiPolygonShape, public SphBasicGeometrySetting
{
  public:
    explicit PhaseOneDiffusionBody(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(createThermalDomainOne(), ShapeBooleanOps::add);
    }
};

class PhaseTwoDiffusionBody : public MultiPolygonShape, public SphBasicGeometrySetting
{
  public:
    explicit PhaseTwoDiffusionBody(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(createThermalDomainTwo(), ShapeBooleanOps::add);
    }
};


class WallBoundary : public MultiPolygonShape, public SphBasicGeometrySetting
{
  public:
    explicit WallBoundary(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(createOuterWall(), ShapeBooleanOps::add);
        multi_polygon_.addAPolygon(createThermalDomainOne(), ShapeBooleanOps::sub);
        multi_polygon_.addAPolygon(createThermalDomainTwo(), ShapeBooleanOps::sub);
    }
};

class UpDirichlet : public MultiPolygonShape, public SphBasicGeometrySetting
{
  public:
    explicit UpDirichlet(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(createUpWallDomain(), ShapeBooleanOps::add);
    }
};

class DownDirichlet : public MultiPolygonShape, public SphBasicGeometrySetting
{
  public:
    explicit DownDirichlet(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(createDownWallDomain(), ShapeBooleanOps::add);
    }
};

class NeumannBoundary : public MultiPolygonShape, public SphBasicGeometrySetting
{
  public:
    explicit NeumannBoundary(const std::string &shape_name) : MultiPolygonShape(shape_name)
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

        if (y >= H * Real(0.5))
        {
            phi_[index_i] = up_temperature;
            return;
        }

        if (y > -H * Real(0.5))
            return;

        const size_t n = down_wall_segment_T.size();
        if (n == 0)
            return;

        const Real seg_len = L / Real(n);

        Real dx = Real(0.1) * seg_len;
        dx = std::min(dx, Real(0.49) * seg_len);
        dx = std::max(dx, Real(0.0));

        Real x_clamped = x;
        if (x_clamped < Real(0.0)) x_clamped = Real(0.0);
        if (x_clamped >= L) x_clamped = L - Real(1.0e-12);

        size_t k = static_cast<size_t>(std::floor(x_clamped / seg_len));
        if (k >= n) k = n - 1;

        size_t kL = (k == 0) ? 0 : k - 1;
        size_t kR = (k + 1 >= n) ? n - 1 : k + 1;

        const Real xk  = seg_len * Real(k);
        const Real xk1 = xk + seg_len;

        const Real Tk = down_wall_segment_T[k];
        Real T_here = Tk;

        if (dx > Real(0.0))
        {
            const Real dx3 = dx * dx * dx;

            if (x_clamped < xk + dx)
            {
                const Real T_left = down_wall_segment_T[kL];
                const Real t1 = (x_clamped - xk - Real(2.0) * dx);
                const Real t2 = (x_clamped - xk + dx);
                const Real poly = t1 * t2 * t2;
                T_here = T_left + ((T_left - Tk) / (Real(4.0) * dx3)) * poly;
            }
            else if (x_clamped > xk1 - dx)
            {
                const Real T_right = down_wall_segment_T[kR];
                const Real t1 = (x_clamped - xk1 - Real(2.0) * dx);
                const Real t2 = (x_clamped - xk1 + dx);
                const Real poly = t1 * t2 * t2;
                T_here = Tk + ((Tk - T_right) / (Real(4.0) * dx3)) * poly;
            }
        }

        phi_[index_i] = T_here;
    }

protected:
    Vecd* pos_;
    Real* phi_;
};

class NeumannWallBoundaryInitialCondition : public LocalDynamics, public SphBasicGeometrySetting
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
