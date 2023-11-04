/**
 * @file 	two_phase_dambreak.h
 * @brief 	Numerical parameters and body definition for 2D two-phase dambreak flow.
 * @author 	Chi Zhang and Xiangyu Hu
 */
#include "sphinxsys.h" // SPHinXsys Library.

#define PI (3.14159265358979323846)
using namespace SPH;   // Namespace cite here.
//----------------------------------------------------------------------
//	Vertical Tank
//----------------------------------------------------------------------
//Real particle_spacing_ref = 0.0045;              /**< Global reference resolution. */
//std::string water = "./input/water_profile.dat";
//std::string tank_inner = "./input/tank_inner_profile.dat";
//std::string tank_outer = "./input/tank_outer_0.018_profile.dat";
//Vecd translation(0, 0.12);
//BoundingBox system_domain_bounds(Vecd(-0.6, -0.2), Vecd(0.6, 0.4));

////----------------------------------------------------------------------
////	Basic geometry parameters and numerical setup.
////----------------------------------------------------------------------
Real DL = 5.3;                      /**< Tank length. */
Real DH = 2.0;                      /**< Tank height. */
Real LL = 5.3;                      /**< Liquid column length. */
Real LH = 1.0;                      /**< Liquid column height. */
Real particle_spacing_ref = 0.05;   /**< Initial reference particle spacing. */
Real BW = particle_spacing_ref * 3; /**< Extending width for BCs. */
/** Domain bounds of the system. */
BoundingBox system_domain_bounds(Vec2d(-BW, -BW), Vec2d(DL + BW, DH + BW));

//----------------------------------------------------------------------
//	Material properties of the fluid.
//----------------------------------------------------------------------
Real rho0_f = 1000.0;                       /**< Reference density of water. */
Real rho0_a = 1.226;                     /**< Reference density of air. */
Real gravity_g = 9.81;                    /**< Gravity force of fluid. */
Real U_ref = 2.0 * sqrt(gravity_g * LH); /**< Characteristic velocity. */
//Real U_ref = 2.0 * sqrt(gravity_g * 0.0612); /**< Characteristic velocity. */
Real c_f = 10.0 * U_ref;                 /**< Reference sound speed. */
Real mu_f = 653.9e-6;
Real mu_a = 20.88e-6;
Real viscous_dynamics = rho0_f * U_ref * 0.64; /**< Dynamics viscosity. */

//----------------------------------------------------------------------
//	Global parameters on the solid properties
//----------------------------------------------------------------------
Real rho0_s = 2800.0; /**< Reference density.*/
Real poisson = 0.33; /**< Poisson ratio.*/
Real Ae = 70.0e9; /**< Normalized Youngs Modulus. */
Real Youngs_modulus = Ae;
Real physical_viscosity = 1.3e4;

//----------------------------------------------------------------------
//	Geometric elements used in shape modeling.
//----------------------------------------------------------------------
std::vector<Vecd> createWaterBlockShape()
{
    std::vector<Vecd> water_block_shape;
    water_block_shape.push_back(Vecd(0.0, 0.0));
    water_block_shape.push_back(Vecd(0.0, LH));
    water_block_shape.push_back(Vecd(LL, LH));
    water_block_shape.push_back(Vecd(LL, 0.0));
    water_block_shape.push_back(Vecd(0.0, 0.0));
    return water_block_shape;
}

std::vector<Vecd> createOuterWallShape()
{
    std::vector<Vecd> outer_wall_shape;
    outer_wall_shape.push_back(Vecd(-BW, -BW));
    outer_wall_shape.push_back(Vecd(-BW, DH + BW));
    outer_wall_shape.push_back(Vecd(DL + BW, DH + BW));
    outer_wall_shape.push_back(Vecd(DL + BW, -BW));
    outer_wall_shape.push_back(Vecd(-BW, -BW));

    return outer_wall_shape;
}

std::vector<Vecd> createInnerWallShape()
{
    std::vector<Vecd> inner_wall_shape;
    inner_wall_shape.push_back(Vecd(0.0, 0.0));
    inner_wall_shape.push_back(Vecd(0.0, DH));
    inner_wall_shape.push_back(Vecd(DL, DH));
    inner_wall_shape.push_back(Vecd(DL, 0.0));
    inner_wall_shape.push_back(Vecd(0.0, 0.0));

    return inner_wall_shape;
}
//----------------------------------------------------------------------
//	cases-dependent geometric shape for water block.
//----------------------------------------------------------------------
class WaterBlock : public MultiPolygonShape
{
  public:
    explicit WaterBlock(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(createWaterBlockShape(), ShapeBooleanOps::add);
    }
};
//class WaterBlock : public MultiPolygonShape
//{
//public:
//	explicit WaterBlock(const std::string &shape_name) : MultiPolygonShape(shape_name)
//	{
//		multi_polygon_.addAPolygonFromFile(water, ShapeBooleanOps::add, translation);
//	}
//};

//----------------------------------------------------------------------
//	cases-dependent geometric shape for air block.
//----------------------------------------------------------------------
class AirBlock : public MultiPolygonShape
{
  public:
    explicit AirBlock(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(createInnerWallShape(), ShapeBooleanOps::add);
        multi_polygon_.addAPolygon(createWaterBlockShape(), ShapeBooleanOps::sub);
    }
};

//class AirBlock : public MultiPolygonShape
//{
//public:
//	explicit AirBlock(const std::string &shape_name) : MultiPolygonShape(shape_name)
//	{
//		multi_polygon_.addAPolygonFromFile(tank_inner, ShapeBooleanOps::add, translation);
//		multi_polygon_.addAPolygonFromFile(water, ShapeBooleanOps::sub, translation);
//	}
//};
//----------------------------------------------------------------------
//	Wall boundary shape definition.
//----------------------------------------------------------------------
class WallBoundary : public ComplexShape
{
  public:
    explicit WallBoundary(const std::string &shape_name) : ComplexShape(shape_name)
    {
        add<MultiPolygonShape>(MultiPolygon(createOuterWallShape()), "OuterWall");
        subtract<MultiPolygonShape>(MultiPolygon(createInnerWallShape()), "InnerWall");
    }
};


//class WallBoundary : public MultiPolygonShape
//{	
//public:
//	explicit WallBoundary(const std::string &shape_name) : MultiPolygonShape(shape_name)
//	{
//		multi_polygon_.addAPolygonFromFile(tank_outer, ShapeBooleanOps::add, translation);
//		multi_polygon_.addAPolygonFromFile(tank_inner, ShapeBooleanOps::sub, translation);
//	}
//};
//----------------------------------------------------------------------
//	Define external excitation 02.
//----------------------------------------------------------------------
Real omega =  2 * PI * 0.5496;
Real Theta0 = - 3.0 * PI / 180.0;

class VariableGravitySecond : public Gravity
{
	Real time_ = 0;

public:
	VariableGravitySecond() : Gravity(Vecd(0.0, 0.0)) {};
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
