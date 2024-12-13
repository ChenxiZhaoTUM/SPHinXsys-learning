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
Real DW = 0.1;                      /**< Tank width. */

Real BL = 6 * 1.0e-3;              // baffle length
Real BH = 0.194;                   // baffle height
Real BaW = 0.095;                   // baffle width

Real LL = DL;                      /**< Liquid column length. */
Real LH = BH * 1.25;                      /**< Liquid column height. */
Real LW = DW;                       // liquid width

Real resolution_ref_solid = BL/8.0;   /**< Initial reference particle spacing. */
Real BW = resolution_ref_solid * 10.0; /**< Extending width for BCs. */

Real particle_spacing_ref = resolution_ref_solid * 2.0;   /**< Initial reference particle spacing. */
//Real particle_spacing_ref = resolution_ref_solid * 2.5;   /**< Initial reference particle spacing. */
//Real particle_spacing_ref = resolution_ref_solid * 4;   /**< Initial reference particle spacing. */
//Real particle_spacing_ref = resolution_ref_solid * 8;   /**< Initial reference particle spacing. */

/** Domain bounds of the system. */
BoundingBox system_domain_bounds(Vecd(-0.5 * DL - BW, - BW, -0.5 * DW - BW), Vecd(0.5 * DL + BW, DH + BW, 0.5 * DW + BW));
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
class BaffleBlock : public ComplexShape
{
  public:
    explicit BaffleBlock(const std::string &shape_name = "BaffleShape") : ComplexShape(shape_name)
    {
        Vecd halfsize_baffle(0.5 * BL, 0.5 * BH, 0.5 * BaW);
        Transform translation_baffle(Vecd(0.0, 0.5 * BH, 0.0));
        add<TransformShape<GeometricShapeBox>>(Transform(translation_baffle), halfsize_baffle);

        Vecd slot_1(11.75e-3, 3.0e-3, 0.5 * BaW);
        Transform slot_1_left(Vecd(-14.75e-3, 3.0e-3, 0.0));
        Transform slot_1_right(Vecd(14.75e-3, 3.0e-3, 0.0));
        add<TransformShape<GeometricShapeBox>>(Transform(slot_1_left), slot_1);
        add<TransformShape<GeometricShapeBox>>(Transform(slot_1_right), slot_1);

        Vecd slot_2(8.0e-3, 3.25e-3, 0.5 * BaW);
        Transform slot_2_left(Vecd(-11e-3, 9.25e-3, 0.0));
        Transform slot_2_right(Vecd(11e-3, 9.25e-3, 0.0));
        add<TransformShape<GeometricShapeBox>>(Transform(slot_2_left), slot_2);
        add<TransformShape<GeometricShapeBox>>(Transform(slot_2_right), slot_2);

        Vecd slot_3(4.0e-3, 6.75e-3, 0.5 * BaW);
        Transform slot_3_left(Vecd(-7e-3, 19.25e-3, 0.0));
        Transform slot_3_right(Vecd(7e-3, 19.25e-3, 0.0));
        add<TransformShape<GeometricShapeBox>>(Transform(slot_3_left), slot_3);
        add<TransformShape<GeometricShapeBox>>(Transform(slot_3_right), slot_3);
    }
};

class Tank : public ComplexShape
{
  public:
    explicit Tank(const std::string &shape_name = "TankShape") : ComplexShape(shape_name)
    {
        Vecd halfsize_outer(0.5 * DL + BW, 0.5 * DH + BW, 0.5 * DW + BW);
        Vecd halfsize_inner(0.5 * DL, 0.5 * DH, 0.5 * DW);
        Transform translation_wall(Vecd(0.0, 0.5 * DH, 0.0));
        add<TransformShape<GeometricShapeBox>>(Transform(translation_wall), halfsize_outer, "OuterWall");
        subtract<TransformShape<GeometricShapeBox>>(Transform(translation_wall), halfsize_inner, "InnerWall");
    }
};

class AirBlock : public ComplexShape
{
  public:
    explicit AirBlock(const std::string &shape_name = "AirShape") : ComplexShape(shape_name)
    {
        Vecd halfsize_air(0.5 * DL, 0.5 * (DH-LH), 0.5 * DW);
        Transform translation_air(Vecd(0.0, 0.5 * (DH+LH), 0.0));
        add<TransformShape<GeometricShapeBox>>(Transform(translation_air), halfsize_air);
    }
};

class WaterBlock : public ComplexShape
{
  public:
    explicit WaterBlock(const std::string &shape_name = "WaterShape") : ComplexShape(shape_name)
    {
        Vecd halfsize_inner(0.5 * DL, 0.5 * DH, 0.5 * DW);
        Transform translation_wall(Vecd(0.0, 0.5 * DH, 0.0));
        add<TransformShape<GeometricShapeBox>>(Transform(translation_wall), halfsize_inner);

        subtract<AirBlock>();
        subtract<BaffleBlock>();
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
	VariableGravity() : Gravity(Vecd::Zero()) {};
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
Vecd probe_halfsize(0.5 * h, 0.5 * DH, 0.5 * h);

class E3ProbeShape : public ComplexShape
{
  public:
    explicit E3ProbeShape(const std::string &shape_name) : ComplexShape(shape_name)
    {
        Transform translation_probeE3(Vecd(- 0.5 * DL + 0.032, probe_halfsize[1], 0.0));
        add<TransformShape<GeometricShapeBox>>(Transform(translation_probeE3), probe_halfsize);
    }
};

class E4ProbeShape : public ComplexShape
{
  public:
    explicit E4ProbeShape(const std::string &shape_name) : ComplexShape(shape_name)
    {
        Transform translation_probeE4(Vecd(0.5 * DL - 0.032, probe_halfsize[1], 0.0));
        add<TransformShape<GeometricShapeBox>>(Transform(translation_probeE4), probe_halfsize);
    }
};

StdVec<Vecd> water_pressure_observation_location = {Vecd(0.5 * DL - h, 0.165, 0.0), Vecd(0.5 * DL - h, 0.220, 0.0), Vecd(0.5 * BL + h, 0.190, 0.0)};
