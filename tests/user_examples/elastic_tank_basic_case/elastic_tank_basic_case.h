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

//----------------------------------------------------------------------
//	Set the file path to the data file.
//----------------------------------------------------------------------
std::string fuel_tank_outer = "./input/tank_outer_0.018.STL";
std::string fuel_tank_inner = "./input/3D_grotle_tank_inner.STL";
std::string water_05 = "./input/3D_grotle_water_0255.STL";
std::string air_05 = "./input/3D_grotle_air_0255.STL";
std::string probe_shape = "./input/base_case_probe_0.106.STL";

//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real resolution_ref = 0.0072;			  /** Initial particle spacing*/
Real length_scale = 1.0;							  /** Scale factor*/
Vecd translation(0, 0.12, 0);
BoundingBox system_domain_bounds(Vecd(-0.6, -0.2, -0.2), Vecd(0.6, 0.4, 0.2));

//----------------------------------------------------------------------
//	Basic parameters for material properties.
//----------------------------------------------------------------------
Real rho0_f = 1000.0;								 /** Fluid density*/
Real rho0_a = 1.226;								   /** Air density*/
Real gravity_g = 9.81;						/** Gravity force of fluid*/
Real U_max = 2.0 * sqrt(gravity_g*0.0612); /** Characteristic velocity*/
Real c_f = 10.0 * U_max;					 /** Reference sound speed*/
Real mu_f = 653.9e-6;							   /** Water viscosity*/
Real mu_a = 20.88e-6;								 /** Air viscosity*/
Real viscous_dynamics = rho0_f * U_max * 0.64; /**< Dynamics viscosity. */

//Real rho0_s = 7890.0;								 /** Solid density*/
//Real poisson = 0.27;								 /** Poisson ratio*/
//Real Youngs_modulus = 135.0e9;

Real rho0_s = 2800.0;								 /** Solid density*/
Real poisson = 0.33;								 /** Poisson ratio*/
Real Youngs_modulus = 70.0e9;
Real physical_viscosity = 1.3e4;
// Real physical_viscosity = sqrt(rho0_s * Youngs_modulus) * 0.03 * 0.03 / 0.24 / 4;
// 
//----------------------------------------------------------------------
//	Define SPH bodies.
//----------------------------------------------------------------------
class Tank : public ComplexShape
{
public:
	explicit Tank(const std::string& shape_name) :ComplexShape(shape_name)
	{
		add<TriangleMeshShapeSTL>(fuel_tank_outer, translation, length_scale, "OuterWall");
		subtract<TriangleMeshShapeSTL>(fuel_tank_inner, translation, length_scale, "InnerWall");
	}
};

class WaterBlock : public ComplexShape
{
public:
	explicit WaterBlock(const std::string& shape_name) : ComplexShape(shape_name)
	{
		add<TriangleMeshShapeSTL>(water_05, translation, length_scale);
	}
};

class AirBlock : public ComplexShape
{
public:
	explicit AirBlock(const std::string& shape_name) : ComplexShape(shape_name)
	{
		add<TriangleMeshShapeSTL>(air_05, translation, length_scale);
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
        p_[index_i] = rho0_f * gravity_g * (0.0612 - pos_[index_i][1]);
        rho_[index_i] = p_[index_i] / pow(c_f, 2) + rho0_f;
    }

  protected:
    BaseParticles *fluid_particles_;
    StdLargeVec<Real> &p_, &rho_;
};

//----------------------------------------------------------------------
//	Define external excitation.
//----------------------------------------------------------------------
class ExternalForceForVariableGravity
{
  public:
	ExternalForceForVariableGravity(){};
    virtual ~ExternalForceForVariableGravity(){};
    virtual Vecd InducedAccelerationForVariableGravity(Vecd &position, Vecd &velocity) = 0;
};

Real omega =  2 * PI * 0.5496;
Real Theta0 = - 3.0 * PI / 180.0;
class VariableGravity : public ExternalForceForVariableGravity
{
  protected:
    Vecd global_acceleration_;
    Vecd zero_potential_reference_;
	Real time_ = 0;

  public:
	  VariableGravity(Vecd gravity_vector = Vecd(0.0, 0.0, 0.0), Vecd reference_position = Vecd::Zero())
		  : ExternalForceForVariableGravity(), global_acceleration_(gravity_vector),
		  zero_potential_reference_(reference_position) {};
    virtual ~VariableGravity(){};

    /** This function can be used for runtime control of external force. */
	virtual Vecd InducedAccelerationForVariableGravity(Vecd &position, Vecd &velocity) override
	{
		time_= GlobalStaticVariables::physical_time_;
		Real Theta = Theta0 * sin(omega * (time_ - 1.0));
		Real ThetaV = Theta0 * omega * cos(omega * (time_ - 1.0));

		if (time_ < 1.0)
		{
			global_acceleration_[0] = 0.0;
			global_acceleration_[1] = -gravity_g;
		}
		else
		{
			global_acceleration_[0] = -gravity_g * sin(Theta) - ThetaV * ThetaV * position[0] + 2 * ThetaV * velocity[1];
			global_acceleration_[1] = -gravity_g * cos(Theta) + ThetaV * ThetaV * position[1] - 2 * ThetaV * velocity[0];
		}

		return global_acceleration_;
	}

	Real getPotential(Vecd& position, Vecd &velocity)
	{
		return InducedAccelerationForVariableGravity(position, velocity).dot(zero_potential_reference_ - position);
	}
};

class TimeStepInitializationForVariableGravity
    : public LocalDynamics,
      public GeneralDataDelegateSimple
{
  private:
	  SharedPtrKeeper<VariableGravity> variable_gravity_ptr_keeper_;

  protected:
	  VariableGravity* variable_gravity_;
	  StdLargeVec<Vecd> &pos_, &acc_prior_, &vel_;

  public:
	  TimeStepInitializationForVariableGravity(SPHBody& sph_body, SharedPtr<VariableGravity> variable_gravity_ptr = makeShared<VariableGravity>(Vecd::Zero()))
		  : LocalDynamics(sph_body), GeneralDataDelegateSimple(sph_body), variable_gravity_(variable_gravity_ptr_keeper_.assignPtr(variable_gravity_ptr)),
		  pos_(particles_->pos_), acc_prior_(particles_->acc_prior_), vel_(particles_->vel_) {};

    virtual ~TimeStepInitializationForVariableGravity(){};

	void update(size_t index_i, Real dt = 0.0)
	{
		acc_prior_[index_i] = variable_gravity_->InducedAccelerationForVariableGravity(pos_[index_i], vel_[index_i]);
	}
};

//----------------------------------------------------------------------
//	Define external excitation 02.
//----------------------------------------------------------------------
class VariableGravitySecond : public Gravity
{
	Real time_ = 0;

public:
	VariableGravitySecond() : Gravity(Vecd(0.0, -gravity_g, 0.0)) {};
	virtual Vecd InducedAcceleration(Vecd& position) override
	{
		time_= GlobalStaticVariables::physical_time_;
		Real Theta = Theta0 * sin(omega * (time_ - 1.0));
		Real ThetaV = Theta0 * omega * cos(omega * (time_ - 1.0));

		Real alpha = std::atan2(position[1], position[0]);
		Real distance = std::sqrt(pow(position[0], 2) + pow(position[1], 2));
		Real Vx = Theta * distance * std::sin(alpha);
		Real Vy = Theta * distance * std::cos(alpha);

		if (time_ < 1.0)
		{
			global_acceleration_[0] = 0.0;
			global_acceleration_[1] = -gravity_g;
		}
		else
		{
			global_acceleration_[0] = -gravity_g * sin(Theta) - ThetaV * ThetaV * position[0] + 2 * ThetaV * Vy;
			global_acceleration_[1] = -gravity_g * cos(Theta) + ThetaV * ThetaV * position[1] - 2 * ThetaV * Vx;
		}

		return global_acceleration_;
	}
};

//----------------------------------------------------------------------
//	Define observer particle generator.
//----------------------------------------------------------------------
//class TankObserverParticleGenerator : public ObserverParticleGenerator
//{
//public:
//	explicit TankObserverParticleGenerator(SPHBody& sph_body) : ObserverParticleGenerator(sph_body)
//	{
//		positions_.push_back(Vecd(-0.198, 0.0, 0.0));
//	}
//};

class ProbeShape : public ComplexShape
{
public:
	explicit ProbeShape(const std::string& shape_name) : ComplexShape(shape_name)
	{
		Vec3d translation_probe(0.0, 0.0, 0.0);
		add<TriangleMeshShapeSTL>(probe_shape, translation_probe, length_scale);
	}
};

//----------------------------------------------------------------------
//	Define constrain class for tank translation and rotation.
//----------------------------------------------------------------------
typedef DataDelegateSimple<SolidParticles> SolidDataSimple;

class QuantityMomentOfMomentum : public QuantitySummation<Vecd>
{
protected:
	StdLargeVec<Real>& mass_;
	StdLargeVec<Vecd>& pos_;
	Vecd mass_center_;

public:
	explicit QuantityMomentOfMomentum(SPHBody& sph_body, Vecd mass_center)
		: QuantitySummation<Vecd>(sph_body, "Velocity"),
		mass_center_(mass_center), mass_(this->particles_->mass_), pos_(this->particles_->pos_)
	{
		this->quantity_name_ = "Moment of Momentum";
	};
	virtual ~QuantityMomentOfMomentum() {};

	Vecd reduce(size_t index_i, Real dt = 0.0)
	{
		return (pos_[index_i] - mass_center_).cross(this->variable_[index_i]) * mass_[index_i];
	};
};

class QuantityMomentOfInertia : public QuantitySummation<Real>
{
protected:
	StdLargeVec<Vecd>& pos_;
	Vecd mass_center_;
	Real p_1_;
	Real p_2_;

public:
	explicit QuantityMomentOfInertia(SPHBody& sph_body, Vecd mass_center, Real position_1, Real position_2)
		: QuantitySummation<Real>(sph_body, "MassiveMeasure"),
		pos_(this->particles_->pos_), mass_center_(mass_center), p_1_(position_1), p_2_(position_2)
	{
		this->quantity_name_ = "Moment of Inertia";
	};
	virtual ~QuantityMomentOfInertia() {};

	Real reduce(size_t index_i, Real dt = 0.0)
	{
		if (p_1_ == p_2_)
		{
			return  ((pos_[index_i] - mass_center_).norm() * (pos_[index_i] - mass_center_).norm()
				- (pos_[index_i][p_1_] - mass_center_[p_1_]) * (pos_[index_i][p_2_] - mass_center_[p_2_])) * this->variable_[index_i];
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
	StdLargeVec<Real>& mass_;


public:
	explicit QuantityMassPosition(SPHBody& sph_body)
		: QuantitySummation<Vecd>(sph_body, "Position"),
		mass_(this->particles_->mass_)
	{
		this->quantity_name_ = "Mass*Position";
	};
	virtual ~QuantityMassPosition() {};

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
	StdLargeVec<Vecd>& vel_;
	StdLargeVec<Vecd>& pos_;

protected:
	virtual void setupDynamics(Real dt = 0.0) override
	{
		angular_velocity_ = moment_of_inertia_.inverse() * compute_total_moment_of_momentum_.exec(dt);
	}

public:
	explicit Constrain3DSolidBodyRotation(SPHBody& sph_body, Vecd mass_center, Matd inertia_tensor)
		: LocalDynamics(sph_body), SolidDataSimple(sph_body),
		vel_(particles_->vel_), pos_(particles_->pos_), compute_total_moment_of_momentum_(sph_body, mass_center),
		mass_center_(mass_center), moment_of_inertia_(inertia_tensor) {}

	virtual ~Constrain3DSolidBodyRotation() {};

	void update(size_t index_i, Real dt = 0.0)
	{
		linear_velocity_ = angular_velocity_.cross((pos_[index_i] - mass_center_));
		vel_[index_i] -= linear_velocity_;
	}
};

#endif // LNG_ETANK_WATERSLOSHING_H