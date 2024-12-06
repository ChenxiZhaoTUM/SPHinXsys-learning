/**
 * @file 	elastic_tank_basic_case.h
 * @brief 	Sloshing in marine LNG fuel tank with elastic material under roll excitation
 * @author
 */

#ifndef LNG_ETANK_WATERSLOSHING_H
#define LNG_ETANK_WATERSLOSHING_H

#include "sphinxsys.h"
using namespace SPH;
#define PI (3.14159265358979323846)

// general parameters for geometry

Real DL = 1.0;              // tank length
Real DH = 0.7;                // tank height
Real DW = 0.1;                // tank width

Real BL = 6 * 1.0e-3;              // baffle length
Real BH = 0.194;                // baffle height
Real BaW = 0.095;                // baffle width

Real LL = 1.0;                // liquid length
Real LH = BH * 1.25;                // liquid height
Real LW = 0.1;                // liquid width

Real resolution_ref = BL/2;              /**< Global reference resolution. */
Real resolution_ref_solid = BL / 4;   // particle spacing
Real BW = resolution_ref * 4; // boundary width
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------

BoundingBox system_domain_bounds(Vecd(-0.5 * DL - BW, - BW, -0.5 * DW - BW), Vecd(0.5 * DL + BW, DH + BW, 0.5 * DW + BW));

//----------------------------------------------------------------------
//	Basic parameters for material properties.
//----------------------------------------------------------------------
Real rho0_f = 1000.0;                          /** Fluid density*/
Real rho0_a = 1.226;                           /** Air density*/
Real gravity_g = 9.81;                         /** Gravity force of fluid*/
Real U_max = 2.0 * sqrt(gravity_g * LH);   /** Characteristic velocity*/
Real c_f = 10.0 * U_max;                       /** Reference sound speed*/
Real mu_f = 653.9e-6;                          /** Water viscosity*/
Real mu_a = 20.88e-6;                          /** Air viscosity*/
Real viscous_dynamics = rho0_f * U_max * DL; /**< Dynamics viscosity. */

// Real rho0_s = 7890.0;								 /** Solid density*/
// Real poisson = 0.27;								 /** Poisson ratio*/
// Real Youngs_modulus = 135.0e9;

//Real rho0_s = 2800.0; /** Solid density*/
//Real poisson = 0.33;  /** Poisson ratio*/
//Real Youngs_modulus = 70.0e9;
//Real physical_viscosity = 1.3e4;
// Real physical_viscosity = sqrt(rho0_s * Youngs_modulus) * 0.03 * 0.03 / 0.24 / 4;
//
//----------------------------------------------------------------------
//	Define SPH bodies.
//----------------------------------------------------------------------
//	define the static solid wall boundary shape
class Tank : public ComplexShape
{
  public:
    explicit Tank(const std::string &shape_name) : ComplexShape(shape_name)
    {
        Vecd halfsize_outer(0.5 * DL + BW, 0.5 * DH + BW, 0.5 * DW + BW);
        Vecd halfsize_inner(0.5 * DL, 0.5 * DH, 0.5 * DW);
        Transform translation_wall(Vecd(0.0, 0.5 * DH, 0.0));
        add<TransformShape<GeometricShapeBox>>(Transform(translation_wall), halfsize_outer, "OuterWall");
        subtract<TransformShape<GeometricShapeBox>>(Transform(translation_wall), halfsize_inner, "InnerWall");
    }
};

class BaffleBlock : public ComplexShape
{
  public:
    explicit BaffleBlock(const std::string &shape_name) : ComplexShape(shape_name)
    {
        Vecd halfsize_baffle(0.5 * BL, 0.5 * BH, 0.5 * BaW);
        Transform translation_baffle(Vecd(0.0, 0.5 * BH, 0.0));
        add<TransformShape<GeometricShapeBox>>(Transform(translation_baffle), halfsize_baffle);
    }
};

//	define the water block shape
class WaterBlock : public ComplexShape
{
  public:
    explicit WaterBlock(const std::string &shape_name) : ComplexShape(shape_name)
    {
        Vecd halfsize_water(0.5 * LL, 0.5 * LH, 0.5 * LW);
        Transform translation_water(Vecd(0.0, 0.5 * LH, 0.0));
        add<TransformShape<GeometricShapeBox>>(Transform(translation_water), halfsize_water);

        Vecd halfsize_baffle(0.5 * BL, 0.5 * BH, 0.5 * BaW);
        Transform translation_baffle(Vecd(0.0, 0.5 * BH, 0.0));
        subtract<TransformShape<GeometricShapeBox>>(Transform(translation_baffle), halfsize_baffle);
    }
};

class AirBlock : public ComplexShape
{
  public:
    explicit AirBlock(const std::string &shape_name) : ComplexShape(shape_name)
    {
        Vecd halfsize_inner(0.5 * DL, 0.5 * DH, 0.5 * DW);
        Transform translation_wall(Vecd(0.0, 0.5 * DH, 0.0));
        add<TransformShape<GeometricShapeBox>>(Transform(translation_wall), halfsize_inner);

        Vecd halfsize_water(0.5 * LL, 0.5 * LH, 0.5 * LW);
        Transform translation_water(Vecd(0.0, 0.5 * LH, 0.0));
        subtract<TransformShape<GeometricShapeBox>>(Transform(translation_water), halfsize_water);
    }
};

class InitialDensity
    : public fluid_dynamics::FluidInitialCondition
{
  public:
    InitialDensity(SPHBody &sph_body)
        : fluid_dynamics::FluidInitialCondition(sph_body),
          fluid_particles_(dynamic_cast<BaseParticles *>(&sph_body.getBaseParticles())),
          p_(*fluid_particles_->getVariableByName<Real>("Pressure")), rho_(fluid_particles_->rho_){};

    void update(size_t index_i, Real dt)
    {
        p_[index_i] = rho0_f * gravity_g * (LH - pos_[index_i][1]);
        rho_[index_i] = p_[index_i] / pow(c_f, 2) + rho0_f;
    }

  protected:
    BaseParticles *fluid_particles_;
    StdLargeVec<Real> &p_, &rho_;
};

//----------------------------------------------------------------------
//	Define external excitation.
//----------------------------------------------------------------------
Real f = 0.9231;
Real omega = 2 * Pi * f;
//Real omega = f;
Real A = 0.01;

class VariableGravity : public Gravity
{
	Real time_ = 0;
public:
	VariableGravity() : Gravity(Vecd(0.0, -gravity_g, 0.0)) {};
	virtual Vecd InducedAcceleration(Vecd& position) override
	{
		
		time_ = GlobalStaticVariables::physical_time_;

		if (time_ < 0.1)
		{
			global_acceleration_[0] = A * omega * cos(omega * time_) / 0.1;
			global_acceleration_[1] = - gravity_g;
		}
		else
		{
			global_acceleration_[0] = - A * omega * omega * sin(omega * time_);
			global_acceleration_[1] = - gravity_g;
		}
		
		return global_acceleration_;
	}
};

//----------------------------------------------------------------------
//	Define observer particle generator.
//----------------------------------------------------------------------
Real h = 1.3 * resolution_ref;
Vecd probe_halfsize(0.5 * h, 0.5 * DH, 0.5 * h);

class E1ProbeShape : public ComplexShape
{
  public:
    explicit E1ProbeShape(const std::string &shape_name) : ComplexShape(shape_name)
    {
        Transform translation_probeE1(Vecd(-0.032, probe_halfsize[1], 0.0));
        add<TransformShape<GeometricShapeBox>>(Transform(translation_probeE1), probe_halfsize);
    }
};

class E2ProbeShape : public ComplexShape
{
  public:
    explicit E2ProbeShape(const std::string &shape_name) : ComplexShape(shape_name)
    {
        Transform translation_probeE2(Vecd(0.032, probe_halfsize[1], 0.0));
        add<TransformShape<GeometricShapeBox>>(Transform(translation_probeE2), probe_halfsize);
    }
};

class E3ProbeShape : public ComplexShape
{
  public:
    explicit E3ProbeShape(const std::string &shape_name) : ComplexShape(shape_name)
    {
        Transform translation_probeE3(Vecd(- 0.5 * DL + 0.5 * h, probe_halfsize[1], 0.0));
        add<TransformShape<GeometricShapeBox>>(Transform(translation_probeE3), probe_halfsize);
    }
};

class E4ProbeShape : public ComplexShape
{
  public:
    explicit E4ProbeShape(const std::string &shape_name) : ComplexShape(shape_name)
    {
        Transform translation_probeE4(Vecd(0.5 * DL - 0.5 * h, probe_halfsize[1], 0.0));
        add<TransformShape<GeometricShapeBox>>(Transform(translation_probeE4), probe_halfsize);
    }
};


 class PressureObserverParticleGenerator : public ObserverParticleGenerator
{
 public:
	explicit PressureObserverParticleGenerator(SPHBody& sph_body) : ObserverParticleGenerator(sph_body)
	{
		positions_.push_back(Vecd(0.5 * DL - h, 0.165, 0.0));
		positions_.push_back(Vecd(0.5 * DL - h, 0.220, 0.0));
		positions_.push_back(Vecd(h, 0.190, 0.0));
	}
};

//----------------------------------------------------------------------
//	Define constrain class for tank translation and rotation.
//----------------------------------------------------------------------
typedef DataDelegateSimple<SolidParticles> SolidDataSimple;

class QuantityMomentOfMomentum : public QuantitySummation<Vecd>
{
  protected:
    StdLargeVec<Real> &mass_;
    StdLargeVec<Vecd> &pos_;
    Vecd mass_center_;

  public:
    explicit QuantityMomentOfMomentum(SPHBody &sph_body, Vecd mass_center)
        : QuantitySummation<Vecd>(sph_body, "Velocity"),
          mass_center_(mass_center), mass_(this->particles_->mass_), pos_(this->particles_->pos_)
    {
        this->quantity_name_ = "Moment of Momentum";
    };
    virtual ~QuantityMomentOfMomentum(){};

    Vecd reduce(size_t index_i, Real dt = 0.0)
    {
        return (pos_[index_i] - mass_center_).cross(this->variable_[index_i]) * mass_[index_i];
    };
};

class QuantityMomentOfInertia : public QuantitySummation<Real>
{
  protected:
    StdLargeVec<Vecd> &pos_;
    Vecd mass_center_;
    Real p_1_;
    Real p_2_;

  public:
    explicit QuantityMomentOfInertia(SPHBody &sph_body, Vecd mass_center, Real position_1, Real position_2)
        : QuantitySummation<Real>(sph_body, "MassiveMeasure"),
          pos_(this->particles_->pos_), mass_center_(mass_center), p_1_(position_1), p_2_(position_2)
    {
        this->quantity_name_ = "Moment of Inertia";
    };
    virtual ~QuantityMomentOfInertia(){};

    Real reduce(size_t index_i, Real dt = 0.0)
    {
        if (p_1_ == p_2_)
        {
            return ((pos_[index_i] - mass_center_).norm() * (pos_[index_i] - mass_center_).norm() - (pos_[index_i][p_1_] - mass_center_[p_1_]) * (pos_[index_i][p_2_] - mass_center_[p_2_])) * this->variable_[index_i];
        }
        else
        {
            return -(pos_[index_i][p_1_] - mass_center_[p_1_]) * (pos_[index_i][p_2_] - mass_center_[p_2_]) * this->variable_[index_i];
        }
    };
};

class QuantityMassPosition : public QuantitySummation<Vecd>
{
  protected:
    StdLargeVec<Real> &mass_;

  public:
    explicit QuantityMassPosition(SPHBody &sph_body)
        : QuantitySummation<Vecd>(sph_body, "Position"),
          mass_(this->particles_->mass_)
    {
        this->quantity_name_ = "Mass*Position";
    };
    virtual ~QuantityMassPosition(){};

    Vecd reduce(size_t index_i, Real dt = 0.0)
    {
        return this->variable_[index_i] * mass_[index_i];
    };
};

class Constrain3DSolidBodyRotation : public LocalDynamics, public SolidDataSimple
{
  private:
    Vecd mass_center_;
    Matd moment_of_inertia_;
    Vecd angular_velocity_;
    Vecd linear_velocity_;
    ReduceDynamics<QuantityMomentOfMomentum> compute_total_moment_of_momentum_;
    StdLargeVec<Vecd> &vel_;
    StdLargeVec<Vecd> &pos_;

  protected:
    virtual void setupDynamics(Real dt = 0.0) override
    {
        angular_velocity_ = moment_of_inertia_.inverse() * compute_total_moment_of_momentum_.exec(dt);
    }

  public:
    explicit Constrain3DSolidBodyRotation(SPHBody &sph_body, Vecd mass_center, Matd inertia_tensor)
        : LocalDynamics(sph_body), SolidDataSimple(sph_body),
          vel_(particles_->vel_), pos_(particles_->pos_), compute_total_moment_of_momentum_(sph_body, mass_center),
          mass_center_(mass_center), moment_of_inertia_(inertia_tensor) {}

    virtual ~Constrain3DSolidBodyRotation(){};

    void update(size_t index_i, Real dt = 0.0)
    {
        linear_velocity_ = angular_velocity_.cross((pos_[index_i] - mass_center_));
        vel_[index_i] -= linear_velocity_;
    }
};

#endif // LNG_ETANK_WATERSLOSHING_H