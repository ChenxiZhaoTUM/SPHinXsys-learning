/**
 * @file 	carotid_VIPO_shell.cpp
 * @brief 	Carotid artery with shell, imposed velocity inlet and pressure outlet condition.
 */

#include "sphinxsys.h"
#include "bidirectional_buffer.h"
#include "density_correciton.h"
#include "density_correciton.hpp"
#include "kernel_summation.h"
#include "kernel_summation.hpp"
#include "pressure_boundary.h"
#include "arbitrary_shape_buffer.h"
#include "arbitrary_shape_buffer_3d.h"

using namespace SPH;
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
std::string full_path_to_file = "./input/aorta_blood_domain.stl";
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Vec3d translation(0.0, 0.0, 0.0);
Real scaling = 1.0E-2;
BoundingBox system_domain_bounds(Vecd(-6.0, -4.0, -2.0)*scaling, Vecd(3.0, 10.0, 15.0)*scaling);
Real dp_0 = 0.06 * scaling;
Real shell_resolution = dp_0 / 2;  /*thickness = 1.0 * shell_resolution*/
//Real shell_resolution = dp_0;  /*thickness = 1.0 * shell_resolution*/
StdVec<Vecd> observer_location = {Vecd(-1.24, 4.41, 5.18) * scaling};
//----------------------------------------------------------------------
//	define the imported model.
//----------------------------------------------------------------------
class ShellShape : public ComplexShape
{
public:
    explicit ShellShape(const std::string &shape_name) : ComplexShape(shape_name),
        mesh_shape_(new TriangleMeshShapeSTL(full_path_to_file, translation, scaling))
    {
        //add<ExtrudeShape<TriangleMeshShapeSTL>>(thickness, full_path_to_file, translation, scaling);
        add<TriangleMeshShapeSTL>(full_path_to_file, translation, scaling);
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
        //add<TriangleMeshShapeSTL>(full_path_to_file, translation, scaling);
        add<ExtrudeShape<TriangleMeshShapeSTL>>(-shell_resolution/2, full_path_to_file, translation, scaling);
    }
};

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
    explicit ParticleGenerator(SPHBody& sph_body, SurfaceParticles &surface_particles, TriangleMeshShapeSTL* mesh_shape) 
        : ParticleGenerator<SurfaceParticles>(sph_body, surface_particles),
        mesh_total_area_(0),
        particle_spacing_(sph_body.sph_adaptation_->ReferenceSpacing()),
        thickness_(particle_spacing_),
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

            addPositionAndVolumetricMeasure(particle_position, avg_particle_volume_/thickness_);
            addSurfaceProperties(initial_shape_.findNormalDirection(particle_position), thickness_);
        }
    }
};
//----------------------------------------------------------------------
//	Buffer location.
//----------------------------------------------------------------------
struct RotationResult
{
    Vec3d axis;
    Real angle;
};

RotationResult RotationCalculator(Vecd target_normal, Vecd standard_direction)
{
    target_normal.normalize();

    Vec3d axis = standard_direction.cross(target_normal);
    Real angle = std::acos(standard_direction.dot(target_normal));

    if (axis.norm() < 1e-6)
    {
        if (standard_direction.dot(target_normal) < 0)
        {
            axis = Vec3d(1, 0, 0);
            angle = M_PI;
        }
        else
        {
            axis = Vec3d(0, 0, 1);
            angle = 0;
        }
    }
    else
    {
        axis.normalize();
    }

    return {axis, angle};
}

Vecd standard_direction(1, 0, 0);

// inlet (-0.9768, 4.6112, 3.0052), (0.1000, 0.1665, 0.9810)
Real A_in = 5.9765 * scaling * scaling;
Vec3d inlet_half = Vec3d(2.0 * dp_0, 1.8 * scaling, 1.8 * scaling);
Vec3d inlet_vector(0.1000, 0.1665, 0.9810);
Vec3d inlet_normal = inlet_vector.normalized();
Vec3d inlet_center = Vec3d(-0.9768, 4.6112, 3.0052) * scaling - inlet_normal * (1.0 * dp_0 + 1.0 * (dp_0 - shell_resolution));
Vec3d inlet_cut_translation = inlet_center - inlet_normal * (2.0 * dp_0 + 1.0 * (dp_0 - shell_resolution));
Vec3d inlet_buffer_translation = inlet_center + inlet_normal * (2.0 * dp_0);
RotationResult inlet_rotation_result = RotationCalculator(inlet_normal, standard_direction);
Rotation3d inlet_emitter_rotation(inlet_rotation_result.angle, inlet_rotation_result.axis);
Rotation3d inlet_disposer_rotation(inlet_rotation_result.angle + Pi, inlet_rotation_result.axis);

// outlet1 (-1.2562, 4.4252, 10.0148), (0.6420, 0.4110, 0.6472)
Real A_out1 = 0.2688 * scaling * scaling;
Vec3d outlet_1_half = Vec3d(2.0 * dp_0, 1.0 * scaling, 1.0 * scaling);
Vec3d outlet_1_vector(0.6420, 0.4110, 0.6472);
Vec3d outlet_1_normal = outlet_1_vector.normalized();
Vec3d outlet_1_center = Vec3d(-1.2562, 4.4252, 10.0148) * scaling + outlet_1_normal * (2.0 * dp_0);
Vec3d outlet_1_cut_translation = outlet_1_center + outlet_1_normal * (0.5 * dp_0 + 1.0 * (dp_0 - shell_resolution));
Vec3d outlet_1_buffer_translation = outlet_1_center - outlet_1_normal * (2.0 * dp_0);
RotationResult outlet_1_rotation_result = RotationCalculator(outlet_1_normal, standard_direction);
Rotation3d outlet_1_disposer_rotation(outlet_1_rotation_result.angle, outlet_1_rotation_result.axis);
Rotation3d outlet_1_emitter_rotation(outlet_1_rotation_result.angle + Pi, outlet_1_rotation_result.axis);

// outlet2 (-2.6303, 3.0594, 10.6919), (-0.0988, 0.0485, 0.9939)
Real A_out2 = 0.2230 * scaling * scaling;
Vec3d outlet_2_half = Vec3d(2.0 * dp_0, 1.0 * scaling, 1.0 * scaling);
Vec3d outlet_2_vector(-0.0988, 0.0485, 0.9939);
Vec3d outlet_2_normal = outlet_2_vector.normalized();
Vec3d outlet_2_center = Vec3d(-2.6303, 3.0594, 10.6919) * scaling + outlet_2_normal * (2.0 * dp_0);
Vec3d outlet_2_cut_translation = outlet_2_center + outlet_2_normal * (1.0 * dp_0 + 1.0 * (dp_0 - shell_resolution));
Vec3d outlet_2_buffer_translation = outlet_2_center - outlet_2_normal * (2.0 * dp_0);
RotationResult outlet_2_rotation_result = RotationCalculator(outlet_2_normal, standard_direction);
Rotation3d outlet_2_disposer_rotation(outlet_2_rotation_result.angle, outlet_2_rotation_result.axis);
Rotation3d outlet_2_emitter_rotation(outlet_2_rotation_result.angle + Pi, outlet_2_rotation_result.axis);

// outlet3 (-2.8585, 1.8357, 9.8034), (-0.1471, -0.1813, 0.9724)
Real A_out3 = 0.3948 * scaling * scaling;
Vec3d outlet_3_half = Vec3d(2.0 * dp_0, 0.4 * scaling, 0.4 * scaling);
Vec3d outlet_3_vector(-0.1471, -0.1813, 0.9724);
Vec3d outlet_3_normal = outlet_3_vector.normalized();
Vec3d outlet_3_center = Vec3d(-2.8585, 1.8357, 9.8034) * scaling + outlet_3_normal * (2.0 * dp_0);
Vec3d outlet_3_cut_translation = outlet_3_center + outlet_3_normal * (1.0 * dp_0 + 1.0 * (dp_0 - shell_resolution));
Vec3d outlet_3_buffer_translation = outlet_3_center - outlet_3_normal * (2.0 * dp_0);
RotationResult outlet_3_rotation_result = RotationCalculator(outlet_3_normal, standard_direction);
Rotation3d outlet_3_disposer_rotation(outlet_3_rotation_result.angle, outlet_3_rotation_result.axis);
Rotation3d outlet_3_emitter_rotation(outlet_3_rotation_result.angle + Pi, outlet_3_rotation_result.axis);

// outlet4 (-1.0946, 1.0386, 9.5016), (0.5675, 0.4280, 0.7034)
Real A_out4 = 0.5134 * scaling * scaling;
Vec3d outlet_4_half = Vec3d(2.0 * dp_0, 1.0 * scaling, 1.0 * scaling);
Vec3d outlet_4_vector(0.5675, 0.4280, 0.7034);
Vec3d outlet_4_normal = outlet_4_vector.normalized();
Vec3d outlet_4_center = Vec3d(-1.0946, 1.0386, 9.5016) * scaling + outlet_4_normal * (2.0 * dp_0);
Vec3d outlet_4_cut_translation = outlet_4_center + outlet_4_normal * (0.5 * dp_0 + 1.0 * (dp_0 - shell_resolution));
Vec3d outlet_4_buffer_translation = outlet_4_center - outlet_4_normal * (2.0 * dp_0);
RotationResult outlet_4_rotation_result = RotationCalculator(outlet_4_normal, standard_direction);
Rotation3d outlet_4_disposer_rotation(outlet_4_rotation_result.angle, outlet_4_rotation_result.axis);
Rotation3d outlet_4_emitter_rotation(outlet_4_rotation_result.angle + Pi, outlet_4_rotation_result.axis);

// outlet5 (-1.6791, -0.8069, 0.5017), (0.0327, -0.0729, 0.9968)
Real A_out5 = 2.67 * scaling * scaling;
Vec3d outlet_5_half = Vec3d(2.0 * dp_0, 2.0 * scaling, 2.0 * scaling);
Vec3d outlet_5_vector(-0.0327, 0.0729, -0.9968);
Vec3d outlet_5_normal = outlet_5_vector.normalized();
Vec3d outlet_5_center = Vec3d(-1.6791, -0.8069, 0.5017) * scaling + outlet_5_normal * (2.0 * dp_0);
Vec3d outlet_5_cut_translation = outlet_5_center + outlet_5_normal * (1.0 * dp_0 + 1.0 * (dp_0 - shell_resolution));
Vec3d outlet_5_buffer_translation = outlet_5_center - outlet_5_normal * (2.0 * dp_0);
RotationResult outlet_5_rotation_result = RotationCalculator(outlet_5_normal, standard_direction);
Rotation3d outlet_5_disposer_rotation(outlet_5_rotation_result.angle, outlet_5_rotation_result.axis);
Rotation3d outlet_5_emitter_rotation(outlet_5_rotation_result.angle + Pi, outlet_5_rotation_result.axis);

//----------------------------------------------------------------------
//	Global parameters on the fluid properties
//----------------------------------------------------------------------
Real rho0_f = 1060; /**< Reference density of fluid. */
Real U_f = 2.0;    /**< Characteristic velocity. */
/** Reference sound speed needs to consider the flow speed in the narrow channels. */
//Real c_f = 10.0 * U_f * SMAX(Real(1), A_in / (A_out1 + A_out2 + A_out3 + A_out4 + A_out5));
Real c_f = 10.0 * U_f;
Real mu_f = 0.00355; /**< Dynamics viscosity. */

//Real rho0_s = 1120;                /** Normalized density. */
//Real Youngs_modulus = 1.08e6;    /** Normalized Youngs Modulus. */
//Real poisson = 0.49;               /** Poisson ratio. */
//Real physical_viscosity = 0.25 * sqrt(rho0_s * Youngs_modulus) * 55.0 * scaling; /** physical damping */
//----------------------------------------------------------------------
//	Inflow velocity
//----------------------------------------------------------------------
struct InflowVelocity
{
    Real u_ave;

    template <class BoundaryConditionType>
    InflowVelocity(BoundaryConditionType &boundary_condition)
        : u_ave(0.0) {}

    Vecd operator()(Vecd &position, Vecd &velocity)
    {
        Vecd target_velocity = velocity;
        Real run_time = GlobalStaticVariables::physical_time_;

        u_ave = 0.3782;
        Real a[8] = {-0.1812, 0.1276, -0.08981, 0.04347, -0.05412, 0.02642, 0.008946, -0.009005};
        Real b[8] = {-0.07725, 0.01466, 0.004295, -0.06679, 0.05679, -0.01878, 0.01869, -0.01888};
        for (size_t i = 0; i < 8; i++)
        {
            u_ave = u_ave + a[i] * cos(8.302 * (i + 1) * run_time) + b[i] * sin(8.302 * (i + 1) * run_time);
        }

        target_velocity[0] = u_ave;
        target_velocity[1] = 0.0;
        target_velocity[2] = 0.0;

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
    SPHSystem sph_system(system_domain_bounds, dp_0);
    sph_system.setRunParticleRelaxation(true); // Tag for run particle relaxation for body-fitted distribution
    sph_system.setReloadParticles(false);       // Tag for computation with save particles distribution
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment();
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.cd
    //----------------------------------------------------------------------
    ShellShape body_from_mesh("BodyFromMesh");
    TriangleMeshShapeSTL* mesh_shape = body_from_mesh.getMeshShape();
    SolidBody shell_body(sph_system, makeShared<ShellShape>("ShellBody"));
    shell_body.defineAdaptation<SPHAdaptation>(1.15, dp_0/shell_resolution);
    shell_body.defineBodyLevelSetShape(2.0)->correctLevelSetSign()->writeLevelSet(sph_system);
    shell_body.defineMaterial<Solid>();
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? shell_body.generateParticles<SurfaceParticles, Reload>(shell_body.getName())
        : shell_body.generateParticles<SurfaceParticles, FromSTLFile>(mesh_shape);

    FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBody"));
    water_block.defineBodyLevelSetShape()->cleanLevelSet();
    water_block.defineMaterial<WeaklyCompressibleFluid>(rho0_f, c_f, mu_f);
    ParticleBuffer<ReserveSizeFactor> in_outlet_particle_buffer(0.5);
    //water_block.generateParticlesWithReserve<BaseParticles, Lattice>(in_outlet_particle_buffer);
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
    ? water_block.generateParticlesWithReserve<BaseParticles, Reload>(in_outlet_particle_buffer, water_block.getName())
    : water_block.generateParticles<BaseParticles, Lattice>();
    
    ObserverBody velocity_observer(sph_system, "VelocityObserver");
    velocity_observer.generateParticles<ObserverParticles>(observer_location);
    //----------------------------------------------------------------------
    //	SPH Particle relaxation section
    //----------------------------------------------------------------------
    /** check whether run particle relaxation for body fitted particle distribution. */
    if (sph_system.RunParticleRelaxation())
    {
        InnerRelation shell_inner(shell_body);
        InnerRelation blood_inner(water_block);

        BodyAlignedCylinderByCell inlet_detection_cylinder(shell_body,
                                                           makeShared<AlignedCylinderShape>(xAxis, Transform(Rotation3d(inlet_emitter_rotation), Vec3d(inlet_cut_translation)), inlet_half[1], inlet_half[0]));
        BodyAlignedBoxByCell outlet01_detection_box(shell_body,
                                                makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_1_emitter_rotation), Vec3d(outlet_1_cut_translation)), outlet_1_half));
        BodyAlignedBoxByCell outlet02_detection_box(shell_body,
                                                makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_2_emitter_rotation), Vec3d(outlet_2_cut_translation)), outlet_2_half));
        BodyAlignedBoxByCell outlet03_detection_box(shell_body,
                                                    makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_3_emitter_rotation), Vec3d(outlet_3_cut_translation)), outlet_3_half));
        BodyAlignedBoxByCell outlet04_detection_box(shell_body,
                                                    makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_4_emitter_rotation), Vec3d(outlet_4_cut_translation)), outlet_4_half));
        BodyAlignedBoxByCell outlet05_detection_box(shell_body,
                                                    makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_5_emitter_rotation), Vec3d(outlet_5_cut_translation)), outlet_5_half));


        RealBody test_body_in(
            sph_system, makeShared<AlignedCylinderShape>(xAxis, Transform(Rotation3d(inlet_emitter_rotation), Vec3d(inlet_cut_translation)), inlet_half[1], inlet_half[0], "TestBodyIn"));
        test_body_in.generateParticles<BaseParticles, Lattice>();

        RealBody test_body_out_1(
        sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_1_emitter_rotation), Vec3d(outlet_1_cut_translation)), outlet_1_half, "TestBodyOut01"));
        test_body_out_1.generateParticles<BaseParticles, Lattice>();

        RealBody test_body_out_2(
        sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_2_emitter_rotation), Vec3d(outlet_2_cut_translation)), outlet_2_half, "TestBodyOut02"));
        test_body_out_2.generateParticles<BaseParticles, Lattice>();

        RealBody test_body_out_3(
            sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_3_emitter_rotation), Vec3d(outlet_3_cut_translation)), outlet_3_half, "TestBodyOut03"));
        test_body_out_3.generateParticles<BaseParticles, Lattice>();

        RealBody test_body_out_4(
            sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_4_emitter_rotation), Vec3d(outlet_4_cut_translation)), outlet_4_half, "TestBodyOut04"));
        test_body_out_4.generateParticles<BaseParticles, Lattice>();

        RealBody test_body_out_5(
            sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_5_emitter_rotation), Vec3d(outlet_5_cut_translation)), outlet_5_half, "TestBodyOut05"));
        test_body_out_5.generateParticles<BaseParticles, Lattice>();
        //----------------------------------------------------------------------
        //	Methods used for particle relaxation.
        //----------------------------------------------------------------------
        using namespace relax_dynamics;
        /** A  Physics relaxation step. */
        SurfaceRelaxationStep relaxation_step_inner(shell_inner);
        ShellNormalDirectionPrediction shell_normal_prediction(shell_inner, shell_resolution * 1.0);

        RelaxationStepInner relaxation_step_inner_blood(blood_inner);

        // here, need a class to switch particles in aligned box to ghost particles (not real particles)
        SimpleDynamics<DeleteParticlesInCylinder> inlet_particles_detection(inlet_detection_cylinder);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet01_particles_detection(outlet01_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet02_particles_detection(outlet02_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet03_particles_detection(outlet03_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet04_particles_detection(outlet04_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet05_particles_detection(outlet05_detection_box);

        /** Write the body state to Vtp file. */
        BodyStatesRecordingToVtp write_shell_to_vtp({shell_body});
        write_shell_to_vtp.addToWrite<Vecd>(shell_body, "NormalDirection");
        BodyStatesRecordingToVtp write_blood_to_vtp({water_block});
        BodyStatesRecordingToVtp write_all_bodies_to_vtp({sph_system});
        /** Write the particle reload files. */
        ReloadParticleIO write_particle_reload_files({ &shell_body, &water_block });
        //----------------------------------------------------------------------
        //	Particle relaxation starts here.
        //----------------------------------------------------------------------
        relaxation_step_inner.getOnSurfaceBounding().exec();
        relaxation_step_inner_blood.SurfaceBounding().exec();
        write_shell_to_vtp.writeToFile(0.0);
        write_blood_to_vtp.writeToFile(0.0);
        shell_body.updateCellLinkedList();
        //----------------------------------------------------------------------
        //	Particle relaxation time stepping start here.
        //----------------------------------------------------------------------
        int ite_p = 0;
        while (ite_p < 5000)
        {
            relaxation_step_inner.exec();
            relaxation_step_inner_blood.exec();
            ite_p += 1;
            if (ite_p % 100 == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "Relaxation steps for the imported model N = " << ite_p << "\n";

                if (ite_p % 1000 == 0)
                {
                    write_shell_to_vtp.writeToFile(ite_p);
                    write_blood_to_vtp.writeToFile(ite_p);
                }
            }
        }
        std::cout << "The physics relaxation process of imported model finish !" << std::endl;

        shell_normal_prediction.smoothing_normal_exec();

        inlet_particles_detection.exec();
        shell_body.updateCellLinkedListWithParticleSort(100);
        outlet01_particles_detection.exec();
        shell_body.updateCellLinkedListWithParticleSort(100);
        outlet02_particles_detection.exec();
        shell_body.updateCellLinkedListWithParticleSort(100);
        outlet03_particles_detection.exec();
        shell_body.updateCellLinkedListWithParticleSort(100);
        outlet04_particles_detection.exec();
        shell_body.updateCellLinkedListWithParticleSort(100);
        outlet05_particles_detection.exec();
        shell_body.updateCellLinkedListWithParticleSort(100);

        write_all_bodies_to_vtp.writeToFile(ite_p);
        write_particle_reload_files.writeToFile(0);

        return 0;
    }
    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the the range of bodies to build neighbor particle lists.
    //  Generally, we first define all the inner relations, then the contact relations.
    //  At last, we define the complex relaxations by combining previous defined
    //  inner and contact relations.
    //----------------------------------------------------------------------
    InnerRelation water_block_inner(water_block);
    InnerRelation shell_inner(shell_body);
    ContactRelationFromShellToFluid water_shell_contact(water_block, {&shell_body}, {false});
    //ContactRelationFromFluidToShell shell_water_contact(shell_body, {&water_block}, {false});
    ShellInnerRelationWithContactKernel shell_curvature_inner(shell_body, water_block);
    ContactRelation velocity_observer_contact(velocity_observer, {&water_block});
    //----------------------------------------------------------------------
    // Combined relations built from basic relations
    // which is only used for update configuration.
    //----------------------------------------------------------------------
    ComplexRelation water_block_complex(water_block_inner, {&water_shell_contact});
    //----------------------------------------------------------------------
    //	Define the main numerical methods used in the simulation.
    //	Note that there may be data dependence on the constructors of these methods.
    //----------------------------------------------------------------------
    // shell dynamics
    //InteractionDynamics<thin_structure_dynamics::ShellCorrectConfiguration> shell_corrected_configuration(shell_inner);
    //Dynamics1Level<thin_structure_dynamics::ShellStressRelaxationFirstHalf> shell_stress_relaxation_first(shell_inner, 3, true);
    //Dynamics1Level<thin_structure_dynamics::ShellStressRelaxationSecondHalf> shell_stress_relaxation_second(shell_inner);
    //ReduceDynamics<thin_structure_dynamics::ShellAcousticTimeStepSize> shell_time_step_size(shell_body);
    SimpleDynamics<thin_structure_dynamics::AverageShellCurvature> shell_average_curvature(shell_curvature_inner);
    //SimpleDynamics<thin_structure_dynamics::UpdateShellNormalDirection> shell_update_normal(shell_body);

    ///** Exert constrain on shell. */
    //BoundaryGeometry boundary_geometry(shell_body, "BoundaryGeometry");
    //SimpleDynamics<thin_structure_dynamics::ConstrainShellBodyRegion> constrain_holder(boundary_geometry);

    // fluid dynamics
    InteractionDynamics<NablaWVComplex> kernel_summation(water_block_inner, water_shell_contact);
    InteractionWithUpdate<SpatialTemporalFreeSurfaceIndicationComplex> boundary_indicator(water_block_inner, water_shell_contact);
    Dynamics1Level<fluid_dynamics::Integration1stHalfWithWallRiemann> pressure_relaxation(water_block_inner, water_shell_contact);
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallRiemann> density_relaxation(water_block_inner, water_shell_contact);
    ReduceDynamics<fluid_dynamics::AdvectionTimeStepSize> get_fluid_advection_time_step_size(water_block, U_f);
    ReduceDynamics<fluid_dynamics::AcousticTimeStepSize> get_fluid_time_step_size(water_block);
    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> viscous_acceleration(water_block_inner, water_shell_contact);
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionComplex<BulkParticles>> transport_velocity_correction(water_block_inner, water_shell_contact);

    // add buffers
    // disposer
    BodyAlignedCylinderByCell inlet_disposer(water_block, makeShared<AlignedCylinderShape>(xAxis, Transform(Rotation3d(inlet_disposer_rotation), Vec3d(inlet_buffer_translation)), inlet_half[1], inlet_half[0]));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletionArb<AlignedCylinderShape>> inlet_disposer_outflow_deletion(inlet_disposer);
    BodyAlignedBoxByCell disposer_1(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_1_disposer_rotation), Vec3d(outlet_1_buffer_translation)), outlet_1_half));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletionWithWindkessel> disposer_outflow_deletion_1(disposer_1, "out01");
    BodyAlignedBoxByCell disposer_2(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_2_disposer_rotation), Vec3d(outlet_2_buffer_translation)), outlet_2_half));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletionWithWindkessel> disposer_outflow_deletion_2(disposer_2, "out02");
    BodyAlignedBoxByCell disposer_3(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_3_disposer_rotation), Vec3d(outlet_3_buffer_translation)), outlet_3_half));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletionWithWindkessel> disposer_outflow_deletion_3(disposer_3, "out03");
    BodyAlignedBoxByCell disposer_4(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_4_disposer_rotation), Vec3d(outlet_4_buffer_translation)), outlet_4_half));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletionWithWindkessel> disposer_outflow_deletion_4(disposer_4, "out04");
    BodyAlignedBoxByCell disposer_5(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_5_disposer_rotation), Vec3d(outlet_5_buffer_translation)), outlet_5_half));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletionWithWindkessel> disposer_outflow_deletion_5(disposer_5, "out05");

    // bidirectional buffer
    BodyAlignedCylinderByCell inlet_emitter(water_block, makeShared<AlignedCylinderShape>(xAxis, Transform(Rotation3d(inlet_emitter_rotation), Vec3d(inlet_buffer_translation)), inlet_half[1], inlet_half[0]));
    fluid_dynamics::NonPrescribedPressureBidirectionalBufferArb<AlignedCylinderShape> inlet_emitter_inflow_injection(inlet_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell outflow_emitter_1(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_1_emitter_rotation), Vec3d(outlet_1_buffer_translation)), outlet_1_half));
    fluid_dynamics::WindkesselOutletBidirectionalBuffer outflow_injection_1(outflow_emitter_1, "out01", in_outlet_particle_buffer);
    BodyAlignedBoxByCell outflow_emitter_2(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_2_emitter_rotation), Vec3d(outlet_2_buffer_translation)), outlet_2_half));
    fluid_dynamics::WindkesselOutletBidirectionalBuffer outflow_injection_2(outflow_emitter_2, "out02", in_outlet_particle_buffer);
    BodyAlignedBoxByCell outflow_emitter_3(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_3_emitter_rotation), Vec3d(outlet_3_buffer_translation)), outlet_3_half));
    fluid_dynamics::WindkesselOutletBidirectionalBuffer outflow_injection_3(outflow_emitter_3, "out03", in_outlet_particle_buffer);
    BodyAlignedBoxByCell outflow_emitter_4(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_4_emitter_rotation), Vec3d(outlet_4_buffer_translation)), outlet_4_half));
    fluid_dynamics::WindkesselOutletBidirectionalBuffer outflow_injection_4(outflow_emitter_4, "out04", in_outlet_particle_buffer);
    BodyAlignedBoxByCell outflow_emitter_5(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_5_emitter_rotation), Vec3d(outlet_5_buffer_translation)), outlet_5_half));
    fluid_dynamics::WindkesselOutletBidirectionalBuffer outflow_injection_5(outflow_emitter_5, "out05", in_outlet_particle_buffer);

    InteractionWithUpdate<fluid_dynamics::DensitySummationPressureComplex> update_fluid_density(water_block_inner, water_shell_contact);
    SimpleDynamics<fluid_dynamics::InflowVelocityConditionArb<InflowVelocity, AlignedCylinderShape>> emitter_buffer_inflow_condition(inlet_emitter);
    SimpleDynamics<fluid_dynamics::WindkesselBoundaryCondition> outflow_pressure_condition1(outflow_emitter_1, "out01");
    SimpleDynamics<fluid_dynamics::WindkesselBoundaryCondition> outflow_pressure_condition2(outflow_emitter_2, "out02");
    SimpleDynamics<fluid_dynamics::WindkesselBoundaryCondition> outflow_pressure_condition3(outflow_emitter_3, "out03");
    SimpleDynamics<fluid_dynamics::WindkesselBoundaryCondition> outflow_pressure_condition4(outflow_emitter_4, "out04");
    SimpleDynamics<fluid_dynamics::WindkesselBoundaryCondition> outflow_pressure_condition5(outflow_emitter_5, "out05");
    
    // FSI
    /*InteractionWithUpdate<solid_dynamics::ViscousForceFromFluid> viscous_force_on_shell(shell_water_contact);
    InteractionWithUpdate<solid_dynamics::PressureForceFromFluid<decltype(density_relaxation)>> pressure_force_on_shell(shell_water_contact);
    solid_dynamics::AverageVelocityAndAcceleration average_velocity_and_acceleration(shell_body);*/
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations, observations
    //	and regression tests of the simulation.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp body_states_recording(sph_system);
    body_states_recording.addToWrite<Real>(water_block, "Pressure");
    body_states_recording.addToWrite<int>(water_block, "Indicator");
    body_states_recording.addToWrite<Real>(water_block, "Density");
    body_states_recording.addToWrite<int>(water_block, "BufferParticleIndicator");
    body_states_recording.addToWrite<Vecd>(shell_body, "NormalDirection");
    //body_states_recording.addToWrite<Vecd>(shell_body, "PressureForceFromFluid");
    body_states_recording.addToWrite<Real>(shell_body, "Average1stPrincipleCurvature");
    body_states_recording.addToWrite<Real>(shell_body, "Average2ndPrincipleCurvature");
    
    /*RegressionTestDynamicTimeWarping<ObservedQuantityRecording<Vecd>>
        write_centerline_velocity("Velocity", velocity_observer_contact);*/
    ObservedQuantityRecording<Vecd> write_centerline_velocity("Velocity", velocity_observer_contact);
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    sph_system.initializeSystemConfigurations();
    //shell_corrected_configuration.exec();
    shell_average_curvature.exec();
    //constrain_holder.exec();
    water_block_complex.updateConfiguration();
    //shell_water_contact.updateConfiguration();
    boundary_indicator.exec();
    inlet_emitter_inflow_injection.tag_buffer_particles.exec();
    outflow_injection_1.tag_buffer_particles.exec();
    outflow_injection_2.tag_buffer_particles.exec();
    outflow_injection_3.tag_buffer_particles.exec();
    outflow_injection_4.tag_buffer_particles.exec();
    outflow_injection_5.tag_buffer_particles.exec();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    size_t number_of_iterations = sph_system.RestartStep();
    int screen_output_interval = 100;
    int observation_sample_interval = screen_output_interval * 2;
    Real end_time = 2.0;   /**< End time. */
    Real Output_Time = end_time/200; /**< Time stamps for output of body states. */
    Real dt = 0.0;          /**< Default acoustic time step sizes. */
    Real dt_s = 0.0; /**< Default acoustic time step sizes for solid. */
    //----------------------------------------------------------------------
    //	Statistics for CPU time
    //----------------------------------------------------------------------
    TickCount t1 = TickCount::now();
    TimeInterval interval;
    TimeInterval interval_computing_time_step;
    TimeInterval interval_computing_pressure_relaxation;
    TimeInterval interval_updating_configuration;
    TickCount time_instance;
    Real accumulated_time = 0.006;
    int updateP_n = 0;

    //----------------------------------------------------------------------
    //	First output before the main loop.
    //----------------------------------------------------------------------
    //body_states_recording.writeToFile();
    write_centerline_velocity.writeToFile(number_of_iterations);

    //----------------------------------------------------------------------
    //	Windkessel parameters.
    //----------------------------------------------------------------------
    //outflow_pressure_condition1.getTargetPressure()->setWindkesselParams(1.18E8, 1.84E9, 7.7E-10, accumulated_time, 0.0000098);
    //outflow_pressure_condition2.getTargetPressure()->setWindkesselParams(1.04E8, 1.63E9, 8.74E-10, accumulated_time, 0.00001);
    //outflow_pressure_condition3.getTargetPressure()->setWindkesselParams(1.18E8, 1.84E9, 7.7E-10, accumulated_time, 0.0000068);
    //outflow_pressure_condition4.getTargetPressure()->setWindkesselParams(9.7E7, 1.52E9, 9.34E-10, accumulated_time, 0.0000118);
    //outflow_pressure_condition5.getTargetPressure()->setWindkesselParams(1.88E7, 2.95E8, 4.82E-9, accumulated_time, 0.000096);

    outflow_pressure_condition1.getTargetPressure()->setWindkesselParams(1.18E8, 1.84E9, 7.7E-10, accumulated_time, 0);
    outflow_pressure_condition2.getTargetPressure()->setWindkesselParams(1.04E8, 1.63E9, 8.74E-10, accumulated_time, 0);
    outflow_pressure_condition3.getTargetPressure()->setWindkesselParams(1.18E8, 1.84E9, 7.7E-10, accumulated_time, 0);
    outflow_pressure_condition4.getTargetPressure()->setWindkesselParams(9.7E7, 1.52E9, 9.34E-10, accumulated_time, 0);
    outflow_pressure_condition5.getTargetPressure()->setWindkesselParams(1.88E7, 2.95E8, 4.82E-9, accumulated_time, 0);
    //----------------------------------------------------------------------
    //	Main loop starts here.
    //----------------------------------------------------------------------
    while (GlobalStaticVariables::physical_time_ < end_time)
    {
        Real integration_time = 0.0;
        /** Integrate time (loop) until the next output time. */
        while (integration_time < Output_Time)
        {
            time_instance = TickCount::now();

            Real Dt = get_fluid_advection_time_step_size.exec();
            //std::cout << "Dt = " << Dt << std::endl;
            update_fluid_density.exec();
            viscous_acceleration.exec();
            transport_velocity_correction.exec();
            /** FSI for viscous force. */
            //viscous_force_on_shell.exec();

            interval_computing_time_step += TickCount::now() - time_instance;
            time_instance = TickCount::now();
            Real relaxation_time = 0.0;
            while (relaxation_time < Dt)
            {
                dt = SMIN(get_fluid_time_step_size.exec(), Dt);
                //std::cout << "dt = " << dt << std::endl;

                pressure_relaxation.exec(dt);
                /** FSI for pressure force. */
                //pressure_force_on_shell.exec();

                kernel_summation.exec();
                emitter_buffer_inflow_condition.exec();

                // windkessel model implementation
                if (GlobalStaticVariables::physical_time_ >= updateP_n * accumulated_time)
                {
                    outflow_pressure_condition1.getTargetPressure()->updateNextPressure();
                    outflow_pressure_condition2.getTargetPressure()->updateNextPressure();
                    outflow_pressure_condition3.getTargetPressure()->updateNextPressure();
                    outflow_pressure_condition4.getTargetPressure()->updateNextPressure();
                    outflow_pressure_condition5.getTargetPressure()->updateNextPressure();

                    ++updateP_n;
                }
                outflow_pressure_condition1.exec(dt);
                outflow_pressure_condition2.exec(dt);
                outflow_pressure_condition3.exec(dt);
                outflow_pressure_condition4.exec(dt);
                outflow_pressure_condition5.exec(dt);    

                density_relaxation.exec(dt);

                /*Real dt_s_sum = 0.0;
                average_velocity_and_acceleration.initialize_displacement_.exec();
                while (dt_s_sum < dt)
                {
                    dt_s = shell_time_step_size.exec();
                    if (dt - dt_s_sum < dt_s)
                        dt_s = dt - dt_s_sum;
                    shell_stress_relaxation_first.exec(dt_s);

                    constrain_holder.exec(dt_s);

                    shell_stress_relaxation_second.exec(dt_s);
                    dt_s_sum += dt_s;
                }
                average_velocity_and_acceleration.update_averages_.exec(dt);*/

                relaxation_time += dt;
                integration_time += dt;
                GlobalStaticVariables::physical_time_ += dt;
            }
            interval_computing_pressure_relaxation += TickCount::now() - time_instance;

            if (number_of_iterations % screen_output_interval == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "N=" << number_of_iterations << "	Time = "
                          << GlobalStaticVariables::physical_time_
                          << "	Dt = " << Dt << "	dt = " << dt << "	dt_s = " << dt_s << "\n";
            
                if (number_of_iterations % observation_sample_interval == 0 && number_of_iterations != sph_system.RestartStep())
                {
                    write_centerline_velocity.writeToFile(number_of_iterations);
                }
            
            }
            number_of_iterations++;

            time_instance = TickCount::now();
            /** Water block configuration and periodic condition. */
            inlet_emitter_inflow_injection.injection.exec();
            outflow_injection_1.injection.exec();
            outflow_injection_2.injection.exec();
            outflow_injection_3.injection.exec();
            outflow_injection_4.injection.exec();
            outflow_injection_5.injection.exec();
            inlet_disposer_outflow_deletion.exec();
            disposer_outflow_deletion_1.exec();
            disposer_outflow_deletion_2.exec();
            disposer_outflow_deletion_3.exec();
            disposer_outflow_deletion_4.exec();
            disposer_outflow_deletion_5.exec();

            water_block.updateCellLinkedListWithParticleSort(100);
            //shell_update_normal.exec();
            //shell_body.updateCellLinkedList();
            //shell_curvature_inner.updateConfiguration();
            //shell_average_curvature.exec();
            //shell_water_contact.updateConfiguration();
            water_block_complex.updateConfiguration();

            interval_updating_configuration += TickCount::now() - time_instance;
            boundary_indicator.exec();

            inlet_emitter_inflow_injection.tag_buffer_particles.exec();
            outflow_injection_1.tag_buffer_particles.exec();
            outflow_injection_2.tag_buffer_particles.exec();
            outflow_injection_3.tag_buffer_particles.exec();
            outflow_injection_4.tag_buffer_particles.exec();
            outflow_injection_5.tag_buffer_particles.exec();
        }
        TickCount t2 = TickCount::now();
        //body_states_recording.writeToFile();
        velocity_observer_contact.updateConfiguration();
        TickCount t3 = TickCount::now();
        interval += t3 - t2;
    }
    TickCount t4 = TickCount::now();

    TimeInterval tt;
    tt = t4 - t1 - interval;
    std::cout << "Total wall time for computation: " << tt.seconds()
              << " seconds." << std::endl;
    std::cout << std::fixed << std::setprecision(9) << "interval_computing_time_step ="
              << interval_computing_time_step.seconds() << "\n";
    std::cout << std::fixed << std::setprecision(9) << "interval_computing_pressure_relaxation = "
              << interval_computing_pressure_relaxation.seconds() << "\n";
    std::cout << std::fixed << std::setprecision(9) << "interval_updating_configuration = "
              << interval_updating_configuration.seconds() << "\n";

    return 0;
}
