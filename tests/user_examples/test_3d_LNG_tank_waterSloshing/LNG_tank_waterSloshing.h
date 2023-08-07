/**
 * @file 	LNG_tank.h
 * @brief 	Sloshing in marine LNG fuel tank under roll excitation
 * @author	
 */

#ifndef LNG_TANK_H
#define LNG_TANK_H

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
Real mu_f = 1.01e-3;							   /** Water viscosity*/
Real mu_a = 17.9e-6;								 /** Air viscosity*/

//----------------------------------------------------------------------
//	Define SPH bodies.
//----------------------------------------------------------------------
class Tank : public ComplexShape
{
public:
	explicit Tank(const std::string &shape_name) :ComplexShape(shape_name)
	{
		add<TriangleMeshShapeSTL>(fuel_tank_outer, translation, length_scale, "OuterWall");
		subtract<TriangleMeshShapeSTL>(fuel_tank_inner, translation, length_scale, "InnerWall");
	}
};

class WaterBlock : public ComplexShape
{
public:
	explicit WaterBlock(const std::string &shape_name) : ComplexShape(shape_name)
	{
		add<TriangleMeshShapeSTL>(water_05, translation, length_scale);
	}
};

class AirBlock : public ComplexShape
{
public:
	explicit AirBlock(const std::string &shape_name) : ComplexShape(shape_name)
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
	explicit TankObserverParticleGenerator(SPHBody &sph_body) : ObserverParticleGenerator(sph_body)
	{
		positions_.push_back(Vecd(-0.198, 0.0, 0.0));
	}
};

#endif // LNG_TANK_H