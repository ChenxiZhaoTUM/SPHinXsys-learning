/**
 * @file 	fsi2.h
 * @brief 	This is the case file for the test of fluid - structure interaction.
 * @details  We consider a flow - induced vibration of an elastic beam behind a cylinder in 2D.
 * @author 	Xiangyu Hu, Chi Zhang and Luhui Han
 */

#ifndef FSI2_CASE_H
#define FSI2_CASE_H

#include "sphinxsys.h"
using namespace SPH;
//----------------------------------------------------------------------
//	Rectangular Tank
//----------------------------------------------------------------------
//Real DL = 5.0;                         /**< Channel length. */
//Real DH = 5.0;                          /**< Channel height. */
//Real resolution_ref = 0.1;              /**< Global reference resolution. */
//
//Real BW = resolution_ref * 4.0;         /**< Boundary width, determined by specific layer of boundary particles. */
//
///** Domain bounds of the system. */
//BoundingBox system_domain_bounds(Vec2d(-DL - BW, -BW), Vec2d(DL + BW, DH + BW));

//----------------------------------------------------------------------
//	Global parameters on the fluid properties
//----------------------------------------------------------------------
//Real rho0_f = 1000.0;                                         /**< Density. */
//Real gravity_g = 9.81;					                      /**< Gravity. */
//Real U_max = 2.0 * sqrt(gravity_g * 0.5 * DH); /**< Characteristic velocity. */
//Real c_f = 10.0 * U_max;				 /**< Reference sound speed. */
//Real mu_water = 653.9e-6;
//Real viscous_dynamics = rho0_f * U_max * 2 * DL; /**< Dynamics viscosity. */

//----------------------------------------------------------------------
//	Vertical Tank
//----------------------------------------------------------------------
Real resolution_ref = 0.001;              /**< Global reference resolution. */
std::string water = "./input/water.dat";
std::string tank_inner = "./input/tank_inner.dat";
std::string tank_outer = "./input/tank_outer.dat";
BoundingBox system_domain_bounds(Vec2d(-0.2, -0.090), Vec2d(0.2, 1.0));

//----------------------------------------------------------------------
//	Global parameters on the fluid properties
//----------------------------------------------------------------------
Real rho0_f = 1000.0;                                         /**< Density. */
Real gravity_g = 9.81;					                      /**< Gravity. */
Real U_max = 2.0 * sqrt(gravity_g * 0.5); /**< Characteristic velocity. */
Real c_f = 10.0 * U_max;				 /**< Reference sound speed. */
Real mu_water = 653.9e-6;
Real viscous_dynamics = rho0_f * U_max * 0.28; /**< Dynamics viscosity. */

//----------------------------------------------------------------------
//	Global parameters on the solid properties
//----------------------------------------------------------------------
Real rho0_s = 7890; /**< Reference density.*/
Real poisson = 0.27; /**< Poisson ratio.*/
Real Ae = 135.0e9;    /**< Normalized Youngs Modulus. */
Real Youngs_modulus = Ae;
//----------------------------------------------------------------------
//	define geometry of SPH bodies
//----------------------------------------------------------------------
///** create a water block shape */
//std::vector<Vecd> createWaterBlockShape()
//{
//    // geometry
//    std::vector<Vecd> water_block_shape;
//    water_block_shape.push_back(Vecd(-DL, 0.0));
//    water_block_shape.push_back(Vecd(-DL, DH / 2));
//    water_block_shape.push_back(Vecd(DL, DH / 2));
//    water_block_shape.push_back(Vecd(DL, 0.0));
//    water_block_shape.push_back(Vecd(-DL, 0.0));
//
//    return water_block_shape;
//}
///** create outer wall shape */
//std::vector<Vecd> createOuterWallShape()
//{
//    std::vector<Vecd> outer_wall_shape;
//    outer_wall_shape.push_back(Vecd(-DL - BW, -BW));
//    outer_wall_shape.push_back(Vecd(-DL - BW, DH + BW));
//    outer_wall_shape.push_back(Vecd(DL + BW, DH + BW));
//    outer_wall_shape.push_back(Vecd(DL + BW, -BW));
//    outer_wall_shape.push_back(Vecd(-DL - BW, -BW));
//
//    return outer_wall_shape;
//}
///**
// * @brief create inner wall shape
// */
//std::vector<Vecd> createInnerWallShape()
//{
//    std::vector<Vecd> inner_wall_shape;
//    inner_wall_shape.push_back(Vecd(-DL, 0.0));
//    inner_wall_shape.push_back(Vecd(-DL, DH));
//    inner_wall_shape.push_back(Vecd(DL, DH));
//    inner_wall_shape.push_back(Vecd(DL, 0.0));
//    inner_wall_shape.push_back(Vecd(-DL, 0.0));
//
//    return inner_wall_shape;
//}

//----------------------------------------------------------------------
//	Define case dependent geometries
//----------------------------------------------------------------------
//class WaterBlock : public MultiPolygonShape
//{
//  public:
//    explicit WaterBlock(const std::string &shape_name) : MultiPolygonShape(shape_name)
//    {
//        multi_polygon_.addAPolygon(createWaterBlockShape(), ShapeBooleanOps::add);
//    }
//};
//class WallBoundary : public MultiPolygonShape
//{
//  public:
//    explicit WallBoundary(const std::string &shape_name) : MultiPolygonShape(shape_name)
//    {
//        multi_polygon_.addAPolygon(createOuterWallShape(), ShapeBooleanOps::add);
//        multi_polygon_.addAPolygon(createInnerWallShape(), ShapeBooleanOps::sub);
//    }
//};

class WaterBlock : public MultiPolygonShape
{
public:
	explicit WaterBlock(const std::string &shape_name) : MultiPolygonShape(shape_name)
	{
		multi_polygon_.addAPolygonFromFile(water, ShapeBooleanOps::add);
	}
};

class WallBoundary : public MultiPolygonShape
{	
public:
	explicit WallBoundary(const std::string &shape_name) : MultiPolygonShape(shape_name)
	{
		multi_polygon_.addAPolygonFromFile(tank_outer, ShapeBooleanOps::add);
		multi_polygon_.addAPolygonFromFile(tank_inner, ShapeBooleanOps::sub);
	}
};

typedef DataDelegateSimple<SolidParticles> SolidDataSimple;

template <typename VariableType>
class QuantityMomentOfMomentum : public QuantitySummation<VariableType>
{
protected:
	StdLargeVec<Vecd> &vel_;
	StdLargeVec<Vecd> &pos_;
	Vecd mass_center_;

public:
	explicit QuantityMomentOfMomentum(SPHBody &sph_body, Vecd mass_center= Vecd::Zero())
		: QuantitySummation<VariableType>(sph_body, "MassiveMeasure"),
		mass_center_(mass_center), vel_(this->particles_->vel_), pos_(this->particles_->pos_)
	{
		this->quantity_name_ = "Moment of Momentum";
	};
	virtual ~QuantityMomentOfMomentum() {};

	VariableType reduce(size_t index_i, Real dt = 0.0)
	{
		return ((pos_[index_i][0] - mass_center_[0]) * vel_[index_i][1] - (pos_[index_i][1] - mass_center_[1]) * vel_[index_i][0])* this->variable_[index_i];
	};
};

template <typename VariableType>
class QuantityMomentOfInertia : public QuantitySummation<VariableType>
{
protected:
	StdLargeVec<Vecd> &pos_;
	Vecd mass_center_;

public:
	explicit QuantityMomentOfInertia(SPHBody &sph_body, Vecd mass_center = Vecd::Zero())
		: QuantitySummation<VariableType>(sph_body, "MassiveMeasure"),
		pos_(this->particles_->pos_), mass_center_(mass_center)
	{
		this->quantity_name_ = "Moment of Inertia";
	};
	virtual ~QuantityMomentOfInertia() {};

	VariableType reduce(size_t index_i, Real dt = 0.0)
	{
		return (pos_[index_i] - mass_center_).norm() *(pos_[index_i] - mass_center_).norm() * this->variable_[index_i];
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
	virtual ~QuantityMassPosition() {};

	Vecd reduce(size_t index_i, Real dt = 0.0)
	{
		return this->variable_[index_i] * mass_[index_i];
	};
};

class Constrain2DSolidBodyRotation : public LocalDynamics, public SolidDataSimple
{
private:
	Vecd mass_center_;
	Real moment_of_inertia_;
	Real angular_velocity_;
	ReduceDynamics<QuantityMomentOfMomentum<Real>> compute_total_moment_of_momentum_;
	StdLargeVec<Vecd> &vel_;
	StdLargeVec<Vecd> &pos_;

protected:
	virtual void setupDynamics(Real dt = 0.0) override
	{
		angular_velocity_ = compute_total_moment_of_momentum_.exec(dt) / moment_of_inertia_;
	}

public:
	explicit Constrain2DSolidBodyRotation(SPHBody &sph_body, Vecd mass_center = Vecd::Zero(), Real moment_of_inertia=Real(0.0))
		: LocalDynamics(sph_body), SolidDataSimple(sph_body),
		vel_(particles_->vel_), pos_(particles_->pos_), compute_total_moment_of_momentum_(sph_body, mass_center),
		mass_center_(mass_center), moment_of_inertia_(moment_of_inertia), angular_velocity_(Real(0.0)) {};
	virtual ~Constrain2DSolidBodyRotation() {};

	void update(size_t index_i, Real dt = 0.0)
	{
		Real x = pos_[index_i][0] - mass_center_[0];
		Real y = pos_[index_i][1] - mass_center_[1];
		Real local_radius = sqrt(pow(y, 2.0) + pow(x, 2.0));
		Real angular = atan2(y, x);
		Vecd linear_velocity_;

		linear_velocity_[1] = angular_velocity_ * local_radius * cos(angular);
		linear_velocity_[0] = -angular_velocity_ * local_radius * sin(angular);
		vel_[index_i] -= linear_velocity_;
	}
};
#endif // FSI2_CASE_H
