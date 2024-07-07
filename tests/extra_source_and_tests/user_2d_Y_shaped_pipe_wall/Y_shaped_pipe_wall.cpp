/**
 * @file 	T_shaped_pipe.cpp
 * @brief 	This is the benchmark test of multi-inlet and multi-outlet.
 * @details We consider a flow with one inlet and two outlets in a T-shaped pipe in 2D.
 * @author 	Xiangyu Hu, Shuoguo Zhang
 */

#include "sphinxsys.h"
using namespace SPH;
//----------------------------------------------------------------------
//	Set the file path to the data file.
//----------------------------------------------------------------------
std::string wall_inner_path = "./input/Yshape_inner_boundary_for_code.dat";
std::string wall_outer_path = "./input/Yshape_outer_boundary_for_code.dat";
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
BoundingBox system_domain_bounds(Vec2d(-3.0, -12.0), Vecd(44.0, 16.0));
Real resolution_ref = 0.125;  /**< Initial reference particle spacing. */
Real BW = resolution_ref * 4; /**< Reference size of the emitter. */
// Real DL_sponge = resolution_ref * 20; /**< Reference size of the emitter buffer to impose inflow condition. */
//-------------------------------------------------------
//----------------------------------------------------------------------
//	Global parameters on the fluid properties
//----------------------------------------------------------------------
Real rho0_f = 1.0; /**< Reference density of fluid. */
Real U_f = 1.0;    /**< Characteristic velocity. */
/** Reference sound speed needs to consider the flow speed in the narrow channels. */
Real c_f = 10.0 * U_f * SMAX(Real(1), Real(7.0) / (Real(1.6) + Real(2.55)));
Real Re = 100.0;                           /**< Reynolds number. */
Real mu_f = rho0_f * U_f * Real(7.0) / Re; /**< Dynamics viscosity. */
//----------------------------------------------------------------------
//	Define case dependent body shapes.
//----------------------------------------------------------------------
class WaterBlock : public MultiPolygonShape
{
  public:
    explicit WaterBlock(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygonFromFile(wall_inner_path, ShapeBooleanOps::add);
    }
};

class WallBoundary : public MultiPolygonShape
{
  public:
    explicit WallBoundary(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygonFromFile(wall_outer_path, ShapeBooleanOps::add);
        multi_polygon_.addAPolygonFromFile(wall_inner_path, ShapeBooleanOps::sub);
    }
};
//----------------------------------------------------------------------
//	Inflow velocity
//----------------------------------------------------------------------
struct InflowVelocity
{
    Real u_ref_, t_ref_;
    AlignedBoxShape &aligned_box_;
    Vecd halfsize_;

    template <class BoundaryConditionType>
    InflowVelocity(BoundaryConditionType &boundary_condition)
        : u_ref_(U_f), t_ref_(2.0),
          aligned_box_(boundary_condition.getAlignedBox()),
          halfsize_(aligned_box_.HalfSize()) {}

    Vecd operator()(Vecd &position, Vecd &velocity)
    {
        Vecd target_velocity = velocity;
        // Real run_time = GlobalStaticVariables::physical_time_;
        // Real u_ave = run_time < t_ref_ ? 0.5 * u_ref_ * (1.0 - cos(Pi * run_time / t_ref_)) : u_ref_;
        // target_velocity[0] = 1.5 * u_ave * SMAX(0.0, 1.0 - position[1] * position[1] / halfsize_[1] / halfsize_[1]);
        target_velocity[0] = 1;
        target_velocity[1] = 0;

        return target_velocity;
    }
};
//-----------------------------------------------------------------------------------------------------------
//	Main program starts here.
//-----------------------------------------------------------------------------------------------------------
int main(int ac, char *av[])
{
    //----------------------------------------------------------------------
    //	Build up the environment of a SPHSystem with global controls.
    //----------------------------------------------------------------------
    SPHSystem sph_system(system_domain_bounds, resolution_ref);
    sph_system.setRunParticleRelaxation(false); // Tag for run particle relaxation for body-fitted distribution
    sph_system.setReloadParticles(true);        // Tag for computation with save particles distribution
#ifdef BOOST_AVAILABLE
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment(); // handle command line arguments
#endif
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.cd
    //----------------------------------------------------------------------
    FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBody"));
    water_block.defineMaterial<WeaklyCompressibleFluid>(rho0_f, c_f, mu_f);
    ParticleBuffer<ReserveSizeFactor> inlet_particle_buffer(0.5);
    water_block.generateParticlesWithReserve<BaseParticles, Lattice>(inlet_particle_buffer);
    SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("WallBoundary"));
    wall_boundary.defineBodyLevelSetShape()->writeLevelSet(sph_system);
    wall_boundary.defineMaterial<Solid>();
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? wall_boundary.generateParticles<BaseParticles, Reload>(wall_boundary.getName())
        : wall_boundary.generateParticles<BaseParticles, Lattice>();

    // test buffer location
    /*Vec2d test_half = Vec2d(0.5 * 3.0, 0.5 * BW);
    Vec2d test_translation = Vec2d(32.716, 13.854);
    Real test_rotation = -0.8506;
    SolidBody test_up(
        sph_system, makeShared<AlignedBoxShape>(Transform(Rotation2d(test_rotation), Vec2d(test_translation)), test_half, "TestBody"));
    test_up.defineParticlesAndMaterial<SolidParticles, Solid>();
    test_up.generateParticles<Lattice>();*/

    //Vec2d emitter_halfsize = Vec2d(0.5 * BW, 0.5 * 7.0);
    //Vec2d emitter_translation = Vec2d(-0.5 * BW, 0.0);
    //BodyAlignedBoxByParticle emitter(water_block, makeShared<AlignedBoxShape>(Transform(Vec2d(emitter_translation)), emitter_halfsize));

    //Vec2d disposer_up_halfsize = Vec2d(0.5 * 3.0, 0.5 * BW);
    Vec2d disposer_up_halfsize = Vec2d(0.5 * BW, 0.5 * 10.0);
    Vec2d disposer_up_translation = Vec2d(33.0, 14.2);
    Real disposer_up_rotation = 0.722;
    BodyAlignedBoxByCell disposer_up(
        water_block, makeShared<AlignedBoxShape>(Transform(Rotation2d(disposer_up_rotation), Vec2d(disposer_up_translation)), disposer_up_halfsize));

    BodyAlignedBoxByCell out_up_detection(
        wall_boundary, makeShared<AlignedBoxShape>(Transform(Rotation2d(disposer_up_rotation), Vec2d(disposer_up_translation)), disposer_up_halfsize));

    /*Vec2d disposer_down_halfsize = Vec2d(0.5 * 4.0, 0.5 * BW);
    Vec2d disposer_down_translation = Vec2d(42.0, -9.7);
    Real disposer_down_rotation = 4.3807;
    BodyAlignedBoxByCell disposer_down(
        water_block, makeShared<AlignedBoxShape>(Transform(Rotation2d(disposer_down_rotation), Vec2d(disposer_down_translation)), disposer_down_halfsize));*/
    
    SolidBody test_up(
        sph_system, makeShared<AlignedBoxShape>(Transform(Rotation2d(disposer_up_rotation), Vec2d(disposer_up_translation)), disposer_up_halfsize, "TestBody"));
    test_up.generateParticles<BaseParticles, Lattice>();
        
    //----------------------------------------------------------------------
    //	SPH Particle relaxation section
    //----------------------------------------------------------------------
    /** check whether run particle relaxation for body fitted particle distribution. */
    if (sph_system.RunParticleRelaxation())
    {
        InnerRelation wall_inner(wall_boundary);
        using namespace relax_dynamics;
        SimpleDynamics<RandomizeParticlePosition> random_particles(wall_boundary);
        RelaxationStepInner relaxation_step_inner(wall_inner);

        // here, need a class to switch particles in aligned box to ghost particles (not real particles)
        SimpleDynamics<AlignedBoxParticlesDetection> out_up_particles_detection(out_up_detection, xAxis);

        /** Write the body state to Vtp file. */
        BodyStatesRecordingToVtp write_wall_state_to_vtp({wall_boundary});
        /** Write the particle reload files. */
        ReloadParticleIO write_particle_reload_files(wall_boundary);
        //----------------------------------------------------------------------
        //	Physics relaxation starts here.
        //----------------------------------------------------------------------
        random_particles.exec(0.25);
        relaxation_step_inner.SurfaceBounding().exec();
        write_wall_state_to_vtp.writeToFile(0.0);
        //----------------------------------------------------------------------
        // From here the time stepping begins.
        //----------------------------------------------------------------------
        int ite = 0;
        int relax_step = 1000;
        while (ite < relax_step)
        {
            relaxation_step_inner.exec();
            ite++;
            if (ite % 100 == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "Relaxation steps N = " << ite << "\n";
                write_wall_state_to_vtp.writeToFile(ite);
            }
        }

        std::cout << "The physics relaxation process of wall particles finish !" << std::endl;

        out_up_particles_detection.exec();
        write_particle_reload_files.writeToFile(0);

        return 0;
    }

    BodyStatesRecordingToVtp write_body_states(sph_system);
    write_body_states.writeToFile();
    return 0;
}
