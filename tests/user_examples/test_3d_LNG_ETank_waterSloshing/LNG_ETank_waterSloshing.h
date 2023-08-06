/**
 * @file 	LNG_ETank_waterSloshing.h
 * @brief 	Sloshing in marine LNG fuel tank under roll excitation
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
std::string fuel_tank_outer = "./input/3D_grotle_tank_outer_03.STL";
std::string fuel_tank_inner = "./input/3D_grotle_tank_inner.STL";
std::string water_05 = "./input/3D_grotle_water_0255.STL";
std::string air_05 = "./input/3D_grotle_air_0255.STL";

//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real resolution_ref = 0.006;			  /** Initial particle spacing*/
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

Real rho0_s = 7890.0;								 /** Solid density*/
Real poisson = 0.27;								 /** Poisson ratio*/
Real Youngs_modulus = 1.35e9;

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

//----------------------------------------------------------------------
//	Define external excitation.
//----------------------------------------------------------------------
/** Roll sloshing */
Real omega = 2 * PI * 0.5496;
Real Theta0 = 3.0 * PI / 180.0;

class Sloshing
	: public fluid_dynamics::FluidInitialCondition
{
public:
	Sloshing(SPHBody &sph_body)
		: FluidInitialCondition(sph_body),
		acc_prior_(particles_->acc_prior_),
		vel_(particles_->vel_)
	{};

protected:
	StdLargeVec<Vecd> &acc_prior_;
	StdLargeVec<Vecd> &vel_;
	Real time_ = 0;

	void update(size_t index_i, Real dt)
	{
		time_= GlobalStaticVariables::physical_time_;
		Real Theta = Theta0 * sin(omega * (time_ - 0.5));
		Real ThetaV = Theta0 * omega * cos(omega * (time_ - 0.5));
		//Real ThetaA = -Theta0 * omega* omega*sin(omega*GlobalStaticVariables::physical_time_);


		if (time_ < 0.5)
		{
			acc_prior_[index_i][0] = 0.0;
			acc_prior_[index_i][1] = -gravity_g;
		}

		if (time_ >= 0.5)
		{
			acc_prior_[index_i][0] = -gravity_g * sin(Theta) + ThetaV * ThetaV * pos_[index_i][0] - 2 * ThetaV * vel_[index_i][1];
			acc_prior_[index_i][1] = -gravity_g * cos(Theta) - ThetaV * ThetaV * pos_[index_i][1] + 2 * ThetaV * vel_[index_i][0];
		}
	}
};

//----------------------------------------------------------------------
//	Define observer particle generator.
//----------------------------------------------------------------------
class TankObserverParticleGenerator : public ObserverParticleGenerator
{
public:
	explicit TankObserverParticleGenerator(SPHBody& sph_body) : ObserverParticleGenerator(sph_body)
	{
		positions_.push_back(Vecd(-0.198, 0.0, 0.0));
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