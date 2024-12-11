/**
 * @file 	two_phase_dambreak.h
 * @brief 	Numerical parameters and body definition for 2D two-phase dambreak flow.
 * @author 	Chi Zhang and Xiangyu Hu
 */
#include "sphinxsys.h" // SPHinXsys Library.
using namespace SPH;   // Namespace cite here.
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real DL = 1.0;                      /**< Tank length. */
Real DH = 0.7;                      /**< Tank height. */
Real BL = 6 * 1.0e-3;              // baffle length
Real BH = 0.194;                   // baffle height
Real LL = DL;                      /**< Liquid column length. */
Real LH = BH * 1.25;                      /**< Liquid column height. */
Real resolution_ref_solid = BL/8.0;   /**< Initial reference particle spacing. */
Real BW = resolution_ref_solid * 10.0; /**< Extending width for BCs. */

//Real particle_spacing_ref = resolution_ref_solid * 2.0;   /**< Initial reference particle spacing. */
//Real particle_spacing_ref = resolution_ref_solid * 2.5;   /**< Initial reference particle spacing. */
//Real particle_spacing_ref = resolution_ref_solid * 4;   /**< Initial reference particle spacing. */
Real particle_spacing_ref = resolution_ref_solid * 8;   /**< Initial reference particle spacing. */

/** Domain bounds of the system. */
BoundingBox system_domain_bounds(Vecd(-0.5 * DL - BW, - BW), Vecd(0.5 * DL + BW, DH + BW));
//----------------------------------------------------------------------
//	Material properties of the fluid.
//----------------------------------------------------------------------
Real rho0_f = 1000.0;                       /**< Reference density of water. */
Real rho0_a = 1.226;                     /**< Reference density of air. */
Real gravity_g = 9.81;                    /**< Gravity force of fluid. */
Real U_ref = 2.0 * sqrt(gravity_g * LH); /**< Characteristic velocity. */
Real c_f = 10.0 * U_ref;                 /**< Reference sound speed. */
Real mu_water = 653.9e-6;
Real mu_air = 20.88e-6;
//----------------------------------------------------------------------
//	Geometric elements used in shape modeling.
//----------------------------------------------------------------------
std::vector<Vecd> createWaterBlockShape()
{
    std::vector<Vecd> water_block_shape;
    water_block_shape.push_back(Vecd(-0.5 * LL, 0.0));
    water_block_shape.push_back(Vecd(-0.5 * LL, LH));
    water_block_shape.push_back(Vecd(0.5 * LL, LH));
    water_block_shape.push_back(Vecd(0.5 * LL, 0.0));
    water_block_shape.push_back(Vecd(-0.5 * LL, 0.0));

    return water_block_shape;
}

std::vector<Vecd> createOuterWallShape()
{
    std::vector<Vecd> outer_wall_shape;
    outer_wall_shape.push_back(Vecd(-0.5 * DL - BW, -BW));
    outer_wall_shape.push_back(Vecd(-0.5 * DL - BW, DH + BW));
    outer_wall_shape.push_back(Vecd(0.5 * DL + BW, DH + BW));
    outer_wall_shape.push_back(Vecd(0.5 * DL + BW, -BW));
    outer_wall_shape.push_back(Vecd(-0.5 * DL - BW, -BW));

    return outer_wall_shape;
}

std::vector<Vecd> createInnerWallShape()
{
    std::vector<Vecd> inner_wall_shape;
    inner_wall_shape.push_back(Vecd(-0.5 * DL, 0.0));
    inner_wall_shape.push_back(Vecd(-0.5 * DL, DH));
    inner_wall_shape.push_back(Vecd(0.5 * DL, DH));
    inner_wall_shape.push_back(Vecd(0.5 * DL, 0.0));
    inner_wall_shape.push_back(Vecd(-0.5 * DL, 0.0));

    return inner_wall_shape;
}

Real slot_scale = 1.0e-3;
std::vector<Vecd> createSlotShape()
{
    std::vector<Vecd> slot_shape;
    slot_shape.push_back(Vecd(-26.5, 0.0) * slot_scale);
    slot_shape.push_back(Vecd(-26.5, 6.0) * slot_scale);
    slot_shape.push_back(Vecd(-19.0, 6.0) * slot_scale);
    slot_shape.push_back(Vecd(-19.0, 12.5) * slot_scale);
    slot_shape.push_back(Vecd(-11.0, 12.5) * slot_scale);
    slot_shape.push_back(Vecd(-11.0, 26.0) * slot_scale);
    slot_shape.push_back(Vecd(-3.0, 26.0) * slot_scale);
    slot_shape.push_back(Vecd(-3.0, 6.0) * slot_scale);
    slot_shape.push_back(Vecd(3.0, 6.0) * slot_scale);
    slot_shape.push_back(Vecd(3.0, 26.0) * slot_scale);
    slot_shape.push_back(Vecd(11.0, 26.0) * slot_scale);
    slot_shape.push_back(Vecd(11.0, 12.5) * slot_scale);
    slot_shape.push_back(Vecd(19.0, 12.5) * slot_scale);
    slot_shape.push_back(Vecd(19.0, 6.0) * slot_scale);
    slot_shape.push_back(Vecd(26.5, 6.0) * slot_scale);
    slot_shape.push_back(Vecd(26.5, 0.0) * slot_scale);
    slot_shape.push_back(Vecd(-26.5, 0.0) * slot_scale);

    return slot_shape;
}

std::vector<Vecd> createBaffleShape()
{
    std::vector<Vecd> baffle_shape;

    baffle_shape.push_back(Vecd(-0.5 * BL, 0.0));
    baffle_shape.push_back(Vecd(-0.5 * BL, BH));
    baffle_shape.push_back(Vecd(0.5 * BL, BH));
    baffle_shape.push_back(Vecd(0.5 * BL, 0.0));
    baffle_shape.push_back(Vecd(-0.5 * BL, 0.0));

    return baffle_shape;
}
//----------------------------------------------------------------------
//	cases-dependent geometric shape for baffle block.
//----------------------------------------------------------------------
class BaffleBlock : public MultiPolygonShape
{
  public:
    explicit BaffleBlock(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(createBaffleShape(), ShapeBooleanOps::add);
    }
};
//----------------------------------------------------------------------
//	cases-dependent geometric shape for water block.
//----------------------------------------------------------------------
class WaterBlock : public MultiPolygonShape
{
  public:
    explicit WaterBlock(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygon(createWaterBlockShape(), ShapeBooleanOps::add);
        multi_polygon_.addAPolygon(createBaffleShape(), ShapeBooleanOps::sub);
        multi_polygon_.addAPolygon(createSlotShape(), ShapeBooleanOps::sub);
    }
};
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
        add<MultiPolygonShape>(MultiPolygon(createSlotShape()), "Slot");
    }
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
public:
	VariableGravity() : Gravity(Vecd(0.0, 0.0)) {};
	Vecd InducedAcceleration(const Vecd &position, Real physical_time) const
	{
		Vecd acceleration = Vecd::Zero();

		if (physical_time < 0.1)
		{
			acceleration[0] = A * omega * cos(omega * physical_time) / 0.1;
			acceleration[1] = - gravity_g;
		}
		else
		{
			acceleration[0] = - A * omega * omega * sin(omega * physical_time);
			acceleration[1] = - gravity_g;
		}
		
		return acceleration;
	}
};

class InitialDensity
    : public fluid_dynamics::FluidInitialCondition
{
  public:
      InitialDensity(SPHBody &sph_body)
        : FluidInitialCondition(sph_body),
          p_(particles_->registerStateVariable<Real>("Pressure")), rho_(particles_->registerStateVariable<Real>("Density")) {};

    void update(size_t index_i, Real dt)
    {
        p_[index_i] = rho0_f * gravity_g * (LH - pos_[index_i][1]);
        rho_[index_i] = p_[index_i] / pow(c_f, 2) + rho0_f;
    }

  protected:
    Real *rho_, *p_;
};

//----------------------------------------------------------------------
//	Observation.
//----------------------------------------------------------------------
Real h = 1.3 * particle_spacing_ref;
MultiPolygon createWaveProbeShapeE1()
{
    std::vector<Vecd> pnts;
    pnts.push_back(Vecd(-0.032 - 0.5 * h, 0.0));
    pnts.push_back(Vecd(-0.032 - 0.5 * h, DH));
    pnts.push_back(Vecd(-0.032 + 0.5 * h, DH));
    pnts.push_back(Vecd(-0.032 + 0.5 * h, 0.0));
    pnts.push_back(Vecd(-0.032 - 0.5 * h, 0.0));

    MultiPolygon multi_polygon;
    multi_polygon.addAPolygon(pnts, ShapeBooleanOps::add);
    return multi_polygon;
}

MultiPolygon createWaveProbeShapeE2()
{
    std::vector<Vecd> pnts;
    pnts.push_back(Vecd(0.032 - 0.5 * h, 0.0));
    pnts.push_back(Vecd(0.032 - 0.5 * h, DH));
    pnts.push_back(Vecd(0.032 + 0.5 * h, DH));
    pnts.push_back(Vecd(0.032 + 0.5 * h, 0.0));
    pnts.push_back(Vecd(0.032 - 0.5 * h, 0.0));

    MultiPolygon multi_polygon;
    multi_polygon.addAPolygon(pnts, ShapeBooleanOps::add);
    return multi_polygon;
}

MultiPolygon createWaveProbeShapeE3()
{
    std::vector<Vecd> pnts;
    pnts.push_back(Vecd(-0.5 * DL + 0.032 - 0.5 * h, 0.0));
    pnts.push_back(Vecd(-0.5 * DL + 0.032 - 0.5 * h, DH));
    pnts.push_back(Vecd(-0.5 * DL + 0.032 + 0.5 * h, DH));
    pnts.push_back(Vecd(-0.5 * DL + 0.032 + 0.5 * h, 0.0));
    pnts.push_back(Vecd(-0.5 * DL + 0.032 - 0.5 * h, 0.0));

    MultiPolygon multi_polygon;
    multi_polygon.addAPolygon(pnts, ShapeBooleanOps::add);
    return multi_polygon;
}

MultiPolygon createWaveProbeShapeE4()
{
    std::vector<Vecd> pnts;
    pnts.push_back(Vecd(0.5 * DL - 0.032 - 0.5 * h, 0.0));
    pnts.push_back(Vecd(0.5 * DL - 0.032 - 0.5 * h, DH));
    pnts.push_back(Vecd(0.5 * DL - 0.032 + 0.5 * h, DH));
    pnts.push_back(Vecd(0.5 * DL - 0.032 + 0.5 * h, 0.0));
    pnts.push_back(Vecd(0.5 * DL - 0.032 - 0.5 * h, 0.0));

    MultiPolygon multi_polygon;
    multi_polygon.addAPolygon(pnts, ShapeBooleanOps::add);
    return multi_polygon;
}

StdVec<Vecd> water_pressure_observation_location = {Vecd(0.5 * DL - h, 0.165), Vecd(0.5 * DL - h, 0.220), Vecd(0.5 * BL + h, 0.190)};
StdVec<Vecd> baffle_displacement_observation_location = {Vecd(0.0, 0.195), Vecd(0.0, 0.145), Vecd(0.0, 0.095)};

//Real h_solid = 1.3 * resolution_ref_solid;
//StdVec<Vecd> tank_pressure_observation_location = {Vecd(0.5 * DL + h_solid, 0.165), Vecd(0.5 * DL + h_solid, 0.220), Vecd(0.5 * BL - h_solid, 0.190)};
//StdVec<Vecd> baffle_pressure_observation_location = {Vecd(0.5 * BL - h_solid, 0.190)};