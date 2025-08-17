/**
 * @file 	case_3d_stenosed_VIPO_rigid_shell.cpp
 * @brief 
 * @details
 * @author 
 */
#include "sphinxsys.h"
#include "bidirectional_buffer.h"
#include "density_correciton.h"
#include "density_correciton.hpp"
#include "kernel_summation.h"
#include "kernel_summation.hpp"
#include "pressure_boundary.h"
#include "particle_generation_and_detection.h"
#include "windkessel_bc.h"
#include "hemodynamic_indices.h"

using namespace SPH;
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
std::string full_path_to_file = "./input/stenosed_fluid.stl";
Real scale = 0.001;
Real diameter = 6.0 * scale;
Real fluid_radius = 0.5 * diameter;
Real full_length = 60 * scale;
//----------------------------------------------------------------------
//	Geometry parameters for shell.
//----------------------------------------------------------------------
int number_of_particles = 20;
Real resolution_ref = diameter / number_of_particles;
Real shell_resolution = resolution_ref;
Real shell_thickness = 1.5 * scale;
Vec3d translation_fluid(0., 0., 0.);
//----------------------------------------------------------------------
//	Geometry parameters for boundary condition.
//----------------------------------------------------------------------
Vec3d emitter_halfsize(resolution_ref * 2, fluid_radius, fluid_radius);
Vec3d emitter_translation(resolution_ref * 2, 0., 0.);
Vec3d disposer_halfsize(resolution_ref * 2, fluid_radius * 1.1, fluid_radius * 1.1);
Vec3d disposer_translation(full_length - disposer_halfsize[0], 0., 0.);
//----------------------------------------------------------------------
//	Domain bounds of the system.
//----------------------------------------------------------------------
BoundingBox system_domain_bounds(Vec3d(0, -0.5 * diameter, -0.5 * diameter) - Vec3d(resolution_ref * 4.0, shell_resolution, shell_resolution),
                                 Vec3d(100 * scale, 0.5 * diameter, 0.5 * diameter) + Vec3d(resolution_ref * 4.0, shell_resolution, shell_resolution));
//----------------------------------------------------------------------
//	define the imported model.
//----------------------------------------------------------------------
class ShellShape : public ComplexShape
{
public:
    explicit ShellShape(const std::string &shape_name) : ComplexShape(shape_name),
        mesh_shape_(new TriangleMeshShapeSTL(full_path_to_file, translation_fluid, scale))
    {
        add<ExtrudeShape<TriangleMeshShapeSTL>>(shell_resolution/2, full_path_to_file, translation_fluid, scale);
    }

    TriangleMeshShapeSTL* getMeshShape() const
    {
        return mesh_shape_.get();
    }

private:
    std::unique_ptr<TriangleMeshShapeSTL> mesh_shape_;
};

class WaterBlock : public ComplexShape
{
public:
    explicit WaterBlock(const std::string &shape_name) : ComplexShape(shape_name)
    {
        add<TriangleMeshShapeSTL>(full_path_to_file, translation_fluid, scale);
    }
};
//----------------------------------------------------------------------
//	Shell particle generation.
//----------------------------------------------------------------------
class FromSTLFile;
template <>
class ParticleGenerator<SurfaceParticles, FromSTLFile> : public ParticleGenerator<SurfaceParticles>
{
    Real mesh_total_area_;
    Real particle_spacing_;
    const Real thickness_;
    Real avg_particle_volume_;
    size_t planned_number_of_particles_;

    TriangleMeshShapeSTL* mesh_shape_;
    Shape &initial_shape_;

public:
    explicit ParticleGenerator(SPHBody &sph_body, SurfaceParticles &surface_particles, TriangleMeshShapeSTL* mesh_shape, Real shell_thickness) 
        : ParticleGenerator<SurfaceParticles>(sph_body, surface_particles),
        mesh_total_area_(0),
        particle_spacing_(sph_body.sph_adaptation_->ReferenceSpacing()),
        thickness_(shell_thickness),
        avg_particle_volume_(pow(particle_spacing_, Dimensions - 1) * thickness_),
        planned_number_of_particles_(0),
        mesh_shape_(mesh_shape), initial_shape_(sph_body.getInitialShape()) 
    {
        if (!mesh_shape_)
        {
            std::cerr << "Error: Mesh shape is not set!" << std::endl;
            return;
        }

        if (!initial_shape_.isValid())
        {
            std::cout << "\n BaseParticleGeneratorLattice Error: initial_shape_ is invalid." << std::endl;
            std::cout << __FILE__ << ':' << __LINE__ << std::endl;
            throw;
        }
    }

    virtual void prepareGeometricData() override
    {
        
        // Preload vertex positions
        std::vector<std::array<Real, 3>> vertex_positions;
        int num_vertices = mesh_shape_->getTriangleMesh()->getNumVertices();
        vertex_positions.reserve(num_vertices);
        for (int i = 0; i < num_vertices; i++)
        {
            const auto &p = mesh_shape_->getTriangleMesh()->getVertexPosition(i);
            vertex_positions.push_back({Real(p[0]), Real(p[1]), Real(p[2])});
        }

        // Preload face
        std::vector<std::array<int, 3>> faces;
        int num_faces = mesh_shape_->getTriangleMesh()->getNumFaces();
        std::cout << "num_faces calculation = " << num_faces << std::endl;
        faces.reserve(num_faces);
        for (int i = 0; i < num_faces; i++)
        {
            auto f1 = mesh_shape_->getTriangleMesh()->getFaceVertex(i, 0);
            auto f2 = mesh_shape_->getTriangleMesh()->getFaceVertex(i, 1);
            auto f3 = mesh_shape_->getTriangleMesh()->getFaceVertex(i, 2);
            faces.push_back({f1, f2, f3});
        }

        // Calculate total volume
        std::vector<Real> face_areas(num_faces);
        for (int i = 0; i < num_faces; ++i)
        {
            Vec3d vertices[3];
            for (int j = 0; j < 3; ++j)
            {
                const auto& pos = vertex_positions[faces[i][j]];
                vertices[j] = Vec3d(pos[0], pos[1], pos[2]);
            }

            Real each_area = calculateEachFaceArea(vertices);
            face_areas[i] = each_area;
            mesh_total_area_ += each_area;
        }

        Real number_of_particles = mesh_total_area_ * thickness_ / avg_particle_volume_ + 0.5;
        planned_number_of_particles_ = int(number_of_particles);
        std::cout << "planned_number_of_particles calculation = " << planned_number_of_particles_ << std::endl;

        // initialize a uniform distribution between 0 (inclusive) and 1 (exclusive)
        std::mt19937_64 rng;
        std::uniform_real_distribution<Real> unif(0, 1);

        // Calculate the interval based on the number of particles.
        Real interval = planned_number_of_particles_ / (num_faces + TinyReal);  // if planned_number_of_particles_ >= num_faces, every face will generate particles
        if (interval <= 0)
            interval = 1; // It has to be lager than 0.

        for (int i = 0; i < num_faces; ++i)
        {
            Vec3d vertices[3];
            for (int j = 0; j < 3; ++j)
            {
                const auto& pos = vertex_positions[faces[i][j]];
                vertices[j] = Vec3d(pos[0], pos[1], pos[2]);
            }

            Real random_real = unif(rng);
            if (random_real <= interval && base_particles_.TotalRealParticles() < planned_number_of_particles_)
            {
                // Generate particle at the center of this triangle face
                // generateParticleAtFaceCenter(vertices);
                
                // Generate particles on this triangle face, unequal
                int particles_per_face = std::max(1, int(planned_number_of_particles_ * (face_areas[i] / mesh_total_area_)));
                generateParticlesOnFace(vertices, particles_per_face);
            }
        }

        std::cout << "Shell particle generation finish!" << std::endl;
    }

private:
    Real calculateEachFaceArea(const Vec3d vertices[3])
    {
        Vec3d edge1 = vertices[1] - vertices[0];
        Vec3d edge2 = vertices[2] - vertices[0];
        Real area = 0.5 * edge1.cross(edge2).norm();
        return area;
    }

    void generateParticleAtFaceCenter(const Vec3d vertices[3])
    {
        Vec3d face_center = (vertices[0] + vertices[1] + vertices[2]) / 3.0;

        addPositionAndVolumetricMeasure(face_center, avg_particle_volume_/thickness_);
        addSurfaceProperties(initial_shape_.findNormalDirection(face_center), thickness_);
    }

    void generateParticlesOnFace(const Vec3d vertices[3], int particles_per_face)
    {
        for (int k = 0; k < particles_per_face; ++k)
        {
            Real u = static_cast<Real>(rand()) / static_cast<Real>(RAND_MAX);
            Real v = static_cast<Real>(rand()) / static_cast<Real>(RAND_MAX);

            if (u + v > 1.0) {
                u = 1.0 - u;
                v = 1.0 - v;
            }

            Vec3d particle_position = (1 - u - v) * vertices[0] + u * vertices[1] + v * vertices[2];
            Vec3d normal = initial_shape_.findNormalDirection(particle_position);
            Vec3d offset_position = particle_position + 0.5 * particle_spacing_ * normal;

            addPositionAndVolumetricMeasure(offset_position, avg_particle_volume_ / thickness_);
            addSurfaceProperties(normal, thickness_);
        }
    }
};
//----------------------------------------------------------------------
//	Material parameters.
//----------------------------------------------------------------------
Real rho0_f = 1060.0; /**< Reference density of fluid. */
Real mu_f = 0.004;
Real Re = 2100;
/**< Characteristic velocity. Average velocity */
Real U_f = Re * mu_f / rho0_f / diameter;
Real U_max = 2 * U_f;
Real c_f = 10.0 * U_max; /**< Reference sound speed. */

StdVec<Vecd> createAxialObservationPoints(
    double full_length, Vec3d translation = Vec3d(0.0, 0.0, 0.0))
{
    StdVec<Vecd> observation_points;
    int nx = 51;
    for (int i = 0; i < nx; i++)
    {
        double x = full_length / (nx - 1) * i;
        Vec3d point_coordinate(x, 0.0, 0.0);
        observation_points.emplace_back(point_coordinate + translation);
    }
    return observation_points;
};

StdVec<Vecd> createRadialObservationPoints(
    double full_length, double diameter, int number_of_particles = 50,
    Vecd translation = Vecd(0.0, 0.0, 0.0))
{
    StdVec<Vecd> observation_points;
    double x = full_length / 2.0;
    double R = diameter / 2.0;

    for (int i = 0; i <= number_of_particles; ++i)
    {
        double z = -R + (2.0 * R) * i / double(number_of_particles);
        observation_points.emplace_back(Vecd(x, 0.0, z) + translation);
    }

    return observation_points;
};
//----------------------------------------------------------------------
//	Inflow velocity
//----------------------------------------------------------------------
struct InflowVelocity
{
    Real u_ave;

    template <class BoundaryConditionType>
    InflowVelocity(BoundaryConditionType &boundary_condition)
        : u_ave(0.0) {}

    Vecd operator()(Vecd &position, Vecd &velocity, Real current_time)
    {
        Vecd target_velocity = velocity;

        u_ave = 0.2339;
        Real a[8] = {-0.0176, -0.0657, -0.0280, 0.0068, 0.0075, 0.0115, 0.0040, 0.0035};
        Real b[8] = {0.1205, 0.0171, -0.0384, -0.0152, -0.0122, 0.0002, 0.0033, 0.0060};
        Real w = 2 * Pi / 1;
        for (size_t i = 0; i < 8; i++)
        {
            u_ave = SMAX(u_ave + a[i] * cos(w * (i + 1) * current_time) + b[i] * sin(w * (i + 1) * current_time),
                        0.0);
        }
            
        target_velocity[0] = SMAX(1.5 * u_ave * (1.0 - (position[1] * position[1] + position[2] * position[2]) / fluid_radius / fluid_radius),
                                  0.);
        target_velocity[1] = 0.0;
        target_velocity[2] = 0.0;

        return target_velocity;
    }
};

int main(int ac, char *av[])
{
    //----------------------------------------------------------------------
    //  Build up -- a SPHSystem --
    //----------------------------------------------------------------------
    SPHSystem system(system_domain_bounds, resolution_ref);
    system.setRunParticleRelaxation(false); // Tag for run particle relaxation for body-fitted distribution
    system.setReloadParticles(true);       // Tag for computation with save particles distribution
#ifdef BOOST_AVAILABLE
    system.handleCommandlineOptions(ac, av); // handle command line arguments
#endif
    IOEnvironment io_environment(system);

    //----------------------------------------------------------------------
    //	Creating bodies with corresponding materials and particles.
    //----------------------------------------------------------------------
    FluidBody water_block(system, makeShared<WaterBlock>("WaterBody"));
    water_block.defineBodyLevelSetShape(2.0)->correctLevelSetSign()->writeLevelSet(system);
    water_block.defineClosure<WeaklyCompressibleFluid, Viscosity>(ConstructArgs(rho0_f, c_f), mu_f);
    ParticleBuffer<ReserveSizeFactor> in_outlet_particle_buffer(0.5);
    //water_block.generateParticlesWithReserve<BaseParticles, Lattice>(in_outlet_particle_buffer);
    (!system.RunParticleRelaxation() && system.ReloadParticles())
    ? water_block.generateParticlesWithReserve<BaseParticles, Reload>(in_outlet_particle_buffer, water_block.getName())
    : water_block.generateParticles<BaseParticles, Lattice>();

    ShellShape body_from_mesh("BodyFromMesh");
    TriangleMeshShapeSTL* mesh_shape = body_from_mesh.getMeshShape();
    SolidBody shell_boundary(system, makeShared<ShellShape>("ShellBody"));
    shell_boundary.defineAdaptation<SPHAdaptation>(1.15, resolution_ref/shell_resolution);
    shell_boundary.defineBodyLevelSetShape(2.0)->correctLevelSetSign()->writeLevelSet(system);
    shell_boundary.defineMaterial<Solid>();
    (!system.RunParticleRelaxation() && system.ReloadParticles())
        ? shell_boundary.generateParticles<SurfaceParticles, Reload>(shell_boundary.getName())
        : shell_boundary.generateParticles<SurfaceParticles, FromSTLFile>(mesh_shape, shell_thickness);

    ObserverBody observer_axial(system, "fluid_observer_axial");
    observer_axial.generateParticles<ObserverParticles>(createAxialObservationPoints(full_length));
    ObserverBody observer_radial(system, "fluid_observer_radial");
    observer_radial.generateParticles<ObserverParticles>(createRadialObservationPoints(full_length, diameter));
    //----------------------------------------------------------------------
    //	SPH Particle relaxation section
    //----------------------------------------------------------------------
    /** check whether run particle relaxation for body fitted particle distribution. */
    if (system.RunParticleRelaxation() && !system.ReloadParticles())
    {
        InnerRelation shell_inner(shell_boundary);
        InnerRelation blood_inner(water_block);

        BodyAlignedBoxByCell inlet_detection_box(shell_boundary,
                                             makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(Pi, Vecd(0., 1.0, 0.)), Vec3d(-emitter_halfsize[0] + shell_resolution/2, 0., 0.)), emitter_halfsize));
        //BodyAlignedBoxByCell outlet_detection_box(shell_boundary,
        //                                        makeShared<AlignedBoxShape>(xAxis, Transform(Vec3d(full_length + disposer_halfsize[0] - shell_resolution/2, 0., 0.)), disposer_halfsize));
        
        BodyAlignedBoxByCell redundant_detection_box_for_shell(shell_boundary,
                                                makeShared<AlignedBoxShape>(xAxis, Transform(Vec3d(80 * scale + shell_resolution/2, 0., 0.)), Vec3d( 20 * scale + shell_resolution, fluid_radius * 1.1, fluid_radius * 1.1)));
        BodyAlignedBoxByCell redundant_detection_box_for_water(water_block,
                                                makeShared<AlignedBoxShape>(xAxis, Transform(Vec3d(80 * scale, 0., 0.)), Vec3d( 20 * scale, fluid_radius * 1.1, fluid_radius * 1.1)));

        //RealBody test_body_in(
        //system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(Pi, Vecd(0., 1.0, 0.)), Vec3d(-emitter_halfsize[0] + shell_resolution/2, 0., 0.)), emitter_halfsize, "TestBodyIn"));
        //test_body_in.generateParticles<BaseParticles, Lattice>();

        //RealBody test_body_out(
        //system, makeShared<AlignedBoxShape>(xAxis, Transform(Vec3d(full_length + disposer_halfsize[0] - shell_resolution/2, 0., 0.)), disposer_halfsize, "TestBodyOut"));
        //test_body_out.generateParticles<BaseParticles, Lattice>();

        //----------------------------------------------------------------------
        //	Methods used for particle relaxation.
        //----------------------------------------------------------------------
        using namespace relax_dynamics;
        /** A  Physics relaxation step. */
        SurfaceRelaxationStep relaxation_step_inner(shell_inner);
        ShellNormalDirectionPrediction shell_normal_prediction(shell_inner, shell_resolution * 1.0);

        RelaxationStepInner relaxation_step_inner_blood(blood_inner);

        // here, need a class to switch particles in aligned box to ghost particles (not real particles)
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> inlet_particles_detection(inlet_detection_box);
        //SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet_particles_detection(outlet_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> redundant_shell_particles_detection(redundant_detection_box_for_shell);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> redundant_water_particles_detection(redundant_detection_box_for_water);

        /** Write the body state to Vtp file. */
        BodyStatesRecordingToVtp write_shell_to_vtp({shell_boundary});
        write_shell_to_vtp.addToWrite<Vecd>(shell_boundary, "NormalDirection");
        BodyStatesRecordingToVtp write_blood_to_vtp({water_block});
        BodyStatesRecordingToVtp write_all_bodies_to_vtp({system});
        /** Write the particle reload files. */
        ReloadParticleIO write_particle_reload_files({ &shell_boundary, &water_block });
        ParticleSorting particle_sorting(shell_boundary);
        ParticleSorting particle_sorting_water(water_block);
        //----------------------------------------------------------------------
        //	Particle relaxation starts here.
        //----------------------------------------------------------------------
        relaxation_step_inner.getOnSurfaceBounding().exec();
        relaxation_step_inner_blood.SurfaceBounding().exec();
        write_shell_to_vtp.writeToFile(0.0);
        write_blood_to_vtp.writeToFile(0.0);
        shell_boundary.updateCellLinkedList();
        //----------------------------------------------------------------------
        //	Particle relaxation time stepping start here.
        //----------------------------------------------------------------------
        int ite_p = 0;
        while (ite_p < 2000)
        {
            relaxation_step_inner.exec();
            relaxation_step_inner_blood.exec();
            ite_p += 1;
            if (ite_p % 500 == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "Relaxation steps for the imported model N = " << ite_p << "\n";
                write_shell_to_vtp.writeToFile(ite_p);
                write_blood_to_vtp.writeToFile(ite_p);
            }
        }
        std::cout << "The physics relaxation process of imported model finish !" << std::endl;

        shell_normal_prediction.smoothing_normal_exec();

        inlet_particles_detection.exec();
        particle_sorting.exec();
        shell_boundary.updateCellLinkedList();
        /*outlet_particles_detection.exec();
        particle_sorting.exec();
        shell_boundary.updateCellLinkedList();*/
        redundant_shell_particles_detection.exec();
        particle_sorting.exec();
        shell_boundary.updateCellLinkedList();

        redundant_water_particles_detection.exec();
        particle_sorting_water.exec();
        water_block.updateCellLinkedList();

        write_all_bodies_to_vtp.writeToFile(ite_p);
        write_particle_reload_files.writeToFile(0);

        return 0;
    }
    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the the range of bodies to build neighbor particle lists.
    //  Generally, we first define all the inner relations, then the contact relations.
    //----------------------------------------------------------------------
    InnerRelation water_block_inner(water_block);
    InnerRelation shell_boundary_inner(shell_boundary);
    ShellInnerRelationWithContactKernel wall_curvature_inner(shell_boundary, water_block);
    ContactRelationFromShellToFluid water_shell_contact(water_block, {&shell_boundary}, {false});
    ContactRelationFromFluidToShell shell_water_contact(shell_boundary, {&water_block}, {false});
    ContactRelation fluid_observer_contact_axial(observer_axial, {&water_block});
    ContactRelation fluid_observer_contact_radial(observer_radial, {&water_block});
    ContactRelation shell_observer_contact_axial(observer_axial, {&shell_boundary});
    //----------------------------------------------------------------------
    // Combined relations built from basic relations
    // which is only used for update configuration.
    //----------------------------------------------------------------------
    ComplexRelation water_block_complex(water_block_inner, water_shell_contact);

    //----------------------------------------------------------------------
    // Define the numerical methods used in the simulation.
    // Note that there may be data dependence on the sequence of constructions.
    // Generally, the geometric models or simple objects without data dependencies,
    // such as gravity, should be initiated first.
    // Then the major physical particle dynamics model should be introduced.
    // Finally, the auxillary models such as time step estimator, initial condition,
    // boundary condition and other constraints should be defined.
    //----------------------------------------------------------------------
    InteractionDynamics<thin_structure_dynamics::ShellCorrectConfiguration> wall_corrected_configuration(shell_boundary_inner);
    SimpleDynamics<thin_structure_dynamics::AverageShellCurvature> shell_curvature(wall_curvature_inner);

    InteractionWithUpdate<LinearGradientCorrectionMatrixComplex> kernel_correction_complex(water_block_inner, water_shell_contact);
    InteractionDynamics<NablaWVComplex> kernel_summation(water_block_inner, water_shell_contact);
    InteractionWithUpdate<SpatialTemporalFreeSurfaceIndicationComplex> boundary_indicator(water_block_inner, water_shell_contact);
    Dynamics1Level<fluid_dynamics::Integration1stHalfCorrectionWithWallRiemann> pressure_relaxation(water_block_inner, water_shell_contact);
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallNoRiemann> density_relaxation(water_block_inner, water_shell_contact);
    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> viscous_acceleration(water_block_inner, water_shell_contact);
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionComplex<BulkParticles>> transport_velocity_correction(water_block_inner, water_shell_contact);

    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep> get_fluid_advection_time_step_size(water_block, U_max);
    ReduceDynamics<fluid_dynamics::AcousticTimeStep> get_fluid_time_step_size(water_block);
    //----------------------------------------------------------------------
    //	Boundary conditions.
    //----------------------------------------------------------------------
    BodyAlignedBoxByCell left_buffer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Vec3d(emitter_translation)), emitter_halfsize));
    //fluid_dynamics::BidirectionalBufferWindkessel<fluid_dynamics::NonPrescribedPressureForFlowRate> left_bidirection_buffer(left_buffer, in_outlet_particle_buffer);
    fluid_dynamics::BidirectionalBuffer<fluid_dynamics::NonPrescribedPressure> left_bidirection_buffer(left_buffer, in_outlet_particle_buffer);
    BodyAlignedBoxByCell right_buffer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(Pi, Vecd(0., 1.0, 0.)), Vec3d(disposer_translation)), disposer_halfsize));
    fluid_dynamics::BidirectionalBufferWindkessel<fluid_dynamics::ResistanceBCPressure> right_bidirection_buffer(right_buffer, in_outlet_particle_buffer);

    InteractionWithUpdate<fluid_dynamics::DensitySummationPressureComplex> update_fluid_density(water_block_inner, water_shell_contact);
    //SimpleDynamics<fluid_dynamics::PressureCondition<fluid_dynamics::NonPrescribedPressureForFlowRate>> left_pressure_condition(left_buffer);
    SimpleDynamics<fluid_dynamics::PressureCondition<fluid_dynamics::NonPrescribedPressure>> left_pressure_condition(left_buffer);
    SimpleDynamics<fluid_dynamics::ResistanceBoundaryCondition> right_pressure_condition(right_buffer);
    SimpleDynamics<fluid_dynamics::InflowVelocityCondition<InflowVelocity>> inflow_velocity_condition(left_buffer);

    ReduceDynamics<fluid_dynamics::SectionTransientFlowRate> compute_inlet_transient_flow_rate(left_buffer, Pi*fluid_radius*fluid_radius);
    ReduceDynamics<fluid_dynamics::SectionTransientFlowRate> compute_outlet_transient_flow_rate(right_buffer, Pi*fluid_radius*fluid_radius);
    
    InteractionWithUpdate<solid_dynamics::WallShearStress> viscous_force_from_fluid(shell_water_contact);
    InteractionWithUpdate<solid_dynamics::PressureForceFromFluid<decltype(density_relaxation)>> pressure_force_on_shell(shell_water_contact);
    SimpleDynamics<solid_dynamics::HemodynamicIndiceCalculation> hemodynamic_indice_calculation(shell_boundary, 1.0);
    //----------------------------------------------------------------------
    //	Define the configuration related particles dynamics.
    //----------------------------------------------------------------------
    ParticleSorting particle_sorting(water_block);
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations, observations
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp body_states_recording(system);
    body_states_recording.addToWrite<int>(water_block, "Indicator");
    body_states_recording.addToWrite<Real>(water_block, "Pressure");
    body_states_recording.addToWrite<Real>(water_block, "Density");
    body_states_recording.addToWrite<int>(water_block, "BufferParticleIndicator");
    body_states_recording.addToWrite<Vecd>(shell_boundary, "NormalDirection");
    body_states_recording.addToWrite<Real>(shell_boundary, "Average1stPrincipleCurvature");
    body_states_recording.addToWrite<Real>(shell_boundary, "Average2ndPrincipleCurvature");
    body_states_recording.addToWrite<Vecd>(shell_boundary, "WallShearStress");
    body_states_recording.addToWrite<Real>(shell_boundary, "TimeAveragedWallShearStress");
    body_states_recording.addToWrite<Real>(shell_boundary, "OscillatoryShearIndex");
    ObservedQuantityRecording<Vecd> write_shell_WSS_axial("WallShearStress", shell_observer_contact_axial);
    AxialVelocityRecording write_fluid_velocity_axial(fluid_observer_contact_axial);
    AxialVelocityRecording write_fluid_velocity_radial(fluid_observer_contact_radial);
    ReducedQuantityRecording<QuantitySummation<Vecd>> write_total_viscous_force_on_wall(shell_boundary, "ViscousForceFromFluid");
    ReducedQuantityRecording<QuantitySummation<Vecd>> write_total_pressure_force_on_wall(shell_boundary, "PressureForceFromFluid");
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    system.initializeSystemCellLinkedLists();
    system.initializeSystemConfigurations();
    wall_corrected_configuration.exec();
    shell_curvature.exec();
    water_block_complex.updateConfiguration();
    shell_water_contact.updateConfiguration();
    boundary_indicator.exec();
    left_bidirection_buffer.tag_buffer_particles.exec();
    right_bidirection_buffer.tag_buffer_particles.exec();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    Real &physical_time = *system.getSystemVariableDataByName<Real>("PhysicalTime");
    size_t number_of_iterations = 0;
    int screen_output_interval = 100;
    Real end_time = 2.0;               /**< End time. */
    Real Output_Time = 0.01; /**< Time stamps for output of body states. */
    Real dt = 0.0;                     /**< Default acoustic time step sizes. */
    Real accumulated_time = 0.01;
    int updateP_n = 0;
    //----------------------------------------------------------------------
    //	Statistics for CPU time
    //----------------------------------------------------------------------
    TickCount t1 = TickCount::now();
    TimeInterval interval;
    TimeInterval interval_computing_time_step;
    TimeInterval interval_computing_pressure_relaxation;
    TimeInterval interval_updating_configuration;
    TickCount time_instance;

    //----------------------------------------------------------------------
    //	First output before the main loop.
    //----------------------------------------------------------------------
    body_states_recording.writeToFile(0);
    //right_pressure_condition.getTargetPressure()->setWindkesselParams(0.0, accumulated_time);
    right_pressure_condition.getTargetPressure()->setWindkesselParams(5.0E6, accumulated_time);
    //----------------------------------------------------------------------
    //	Main loop starts here.
    //----------------------------------------------------------------------
    while (physical_time < end_time)
    {
        Real integration_time = 0.0;
        /** Integrate time (loop) until the next output time. */
        while (integration_time < Output_Time)
        {
            /** Acceleration due to viscous force and gravity. */
            time_instance = TickCount::now();
            Real Dt = get_fluid_advection_time_step_size.exec();
            update_fluid_density.exec();
            kernel_correction_complex.exec();
            viscous_acceleration.exec();
            transport_velocity_correction.exec();

            /** FSI for viscous force. */
            viscous_force_from_fluid.exec();
            hemodynamic_indice_calculation.exec(Dt);

            interval_computing_time_step += TickCount::now() - time_instance;
            /** Dynamics including pressure relaxation. */
            time_instance = TickCount::now();
            Real relaxation_time = 0.0;
            while (relaxation_time < Dt)
            {
                dt = SMIN(get_fluid_time_step_size.exec(),
                          Dt - relaxation_time);
                pressure_relaxation.exec(dt);
                pressure_force_on_shell.exec();

                // boundary condition implementation
                kernel_summation.exec();
                left_pressure_condition.exec(dt);
                if (physical_time >= updateP_n * accumulated_time)
                {
                    right_pressure_condition.getTargetPressure()->updateNextPressure();

                    ++updateP_n;
                }
                right_pressure_condition.exec(dt);
                inflow_velocity_condition.exec();
                
                density_relaxation.exec(dt);

                relaxation_time += dt;
                integration_time += dt;
                physical_time += dt;
            }
            interval_computing_pressure_relaxation += TickCount::now() - time_instance;
            
            if (number_of_iterations % screen_output_interval == 0)
            {
                std::cout << std::fixed << std::setprecision(9)
                          << "N=" << number_of_iterations
                          << "	Time = " << physical_time
                          << "	Dt = " << Dt << "	dt = " << dt << "\n";
            }
            number_of_iterations++;

            time_instance = TickCount::now();

            compute_inlet_transient_flow_rate.exec();
            compute_outlet_transient_flow_rate.exec();

            /** Water block configuration and periodic condition. */
            left_bidirection_buffer.injection.exec();
            right_bidirection_buffer.injection.exec();
            
            left_bidirection_buffer.deletion.exec();
            right_bidirection_buffer.deletion.exec();

            if (number_of_iterations % 100 == 0 && number_of_iterations != 1)
            {
                particle_sorting.exec();
            }

            water_block.updateCellLinkedList();
            water_block_complex.updateConfiguration();
            shell_water_contact.updateConfiguration();

            interval_updating_configuration += TickCount::now() - time_instance;
            boundary_indicator.exec();

            left_bidirection_buffer.tag_buffer_particles.exec();
            right_bidirection_buffer.tag_buffer_particles.exec();

            //body_states_recording.writeToFile();
        }
        TickCount t2 = TickCount::now();
        body_states_recording.writeToFile();

        /** Update observer and write output of observer. */
        fluid_observer_contact_axial.updateConfiguration();
        fluid_observer_contact_radial.updateConfiguration();
        write_fluid_velocity_axial.writeToFile(number_of_iterations);
        write_fluid_velocity_radial.writeToFile(number_of_iterations);

        shell_observer_contact_axial.updateConfiguration();
        write_shell_WSS_axial.writeToFile(number_of_iterations);
    
        write_total_viscous_force_on_wall.writeToFile(number_of_iterations);
        write_total_pressure_force_on_wall.writeToFile(number_of_iterations);

        TickCount t3 = TickCount::now();
        interval += t3 - t2;
    }
    TickCount t4 = TickCount::now();

    TimeInterval tt;
    tt = t4 - t1 - interval;
    std::cout << "Total wall time for computation: " << tt.seconds()
              << " seconds." << std::endl;
    std::cout << std::fixed << std::setprecision(9)
              << "interval_computing_time_step ="
              << interval_computing_time_step.seconds() << "\n";
    std::cout << std::fixed << std::setprecision(9)
              << "interval_computing_pressure_relaxation = "
              << interval_computing_pressure_relaxation.seconds() << "\n";
    std::cout << std::fixed << std::setprecision(9)
              << "interval_updating_configuration = "
              << interval_updating_configuration.seconds() << "\n";
    return 0;
}
