﻿/**
 * @file 	particle_relaxation_single_resolution.cpp
 * @brief 	This is the test of using levelset to generate particles with single resolution and relax particles.
 * @details We use this case to test the particle generation and relaxation for a complex geometry.
 *			Before particle generation, we clean the sharp corners of the model.
 * @author 	Yongchuan Yu and Xiangyu Hu
 */

#include "sphinxsys.h"
#include <numeric>

using namespace SPH;
//----------------------------------------------------------------------
//	Setting for the geometry.
//	To use this, please commenting the setting for the first geometry.
//----------------------------------------------------------------------
std::string full_path_to_file = "./input/carotid_fluid_geo.stl";
// std::string full_path_to_file = "./input/carotid_solid_geo.stl";
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Vec3d translation(0.0, 0.0, 0.0);
//Real scaling = pow(10, -3);
Real scaling = 1.0;
Vec3d domain_lower_bound(-6.0 * scaling, -4.0 * scaling, -32.5 * scaling);
Vec3d domain_upper_bound(12.0 * scaling, 10.0 * scaling, 23.5 * scaling);
//----------------------------------------------------------------------
//	Below are common parts for the two test geometries.
//----------------------------------------------------------------------
BoundingBox system_domain_bounds(domain_lower_bound, domain_upper_bound);
Real dp_0 = 0.15 * scaling;
Real thickness = 1.0 * dp_0;
//Real level_set_refinement_ratio = dp_0 / (0.1 * thickness);

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

            if (initial_shape_.checkContain(particle_position))
            {
                addPositionAndVolumetricMeasure(particle_position, avg_particle_volume_/thickness_);
                addSurfaceProperties(initial_shape_.findNormalDirection(particle_position), thickness_);
            }
        }
    }
};

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

// inlet R=2.9293, (1.5611, 5.8559, -30.8885), (0.1034, -0.0458, 0.9935)
Vec3d inlet_half = Vec3d(1.5 * dp_0, 3.5 * scaling, 3.5 * scaling);
Vec3d inlet_normal(-0.1034, 0.0458, -0.9935);
Vec3d inlet_translation = Vec3d(1.5611, 5.8559, -30.8885) * scaling + inlet_normal * 1.0 * dp_0;
Vec3d inlet_standard_direction(1, 0, 0);
RotationResult inlet_rotation_result = RotationCalculator(inlet_normal, inlet_standard_direction);
Rotation3d inlet_rotation(inlet_rotation_result.angle, inlet_rotation_result.axis);

// outlet1 R=1.9416, (-2.6975, -0.4330, 21.7855), (-0.3160, -0.0009, 0.9488)
Vec3d outlet_01_half = Vec3d(1.5 * dp_0, 2.4 * scaling, 2.4 * scaling);
Vec3d outlet_01_normal(-0.3160, -0.0009, 0.9488);
Vec3d outlet_01_translation = Vec3d(-2.6975, -0.4330, 21.7855) * scaling + outlet_01_normal * 1.0 * dp_0;
Vec3d outlet_01_standard_direction(1, 0, 0);
RotationResult outlet_01_rotation_result = RotationCalculator(outlet_01_normal, outlet_01_standard_direction);
Rotation3d outlet_01_rotation(outlet_01_rotation_result.angle, outlet_01_rotation_result.axis);

// outlet2 R=1.3261, (9.0220, 0.9750, 18.6389), (-0.0399, 0.0693, 0.9972)
Vec3d outlet_02_half = Vec3d(1.5 * dp_0, 2.0 * scaling, 2.0 * scaling);
Vec3d outlet_02_normal(-0.0399, 0.0693, 0.9972);
Vec3d outlet_02_translation = Vec3d(9.0220, 0.9750, 18.6389) * scaling + outlet_02_normal * 1.0 * dp_0;
Vec3d outlet_02_standard_direction(1, 0, 0);
RotationResult outlet_02_rotation_result = RotationCalculator(outlet_02_normal, outlet_02_standard_direction);
Rotation3d outlet_02_rotation(outlet_02_rotation_result.angle, outlet_02_rotation_result.axis);

//----------------------------------------------------------------------
//	define the imported model.
//----------------------------------------------------------------------
class SolidBodyFromMesh : public ComplexShape
{
public:
    explicit SolidBodyFromMesh(const std::string &shape_name) : ComplexShape(shape_name),
        mesh_shape_(new TriangleMeshShapeSTL(full_path_to_file, translation, scaling))
    {
        //add<TriangleMeshShapeSTL>(full_path_to_file, translation, scaling);
        add<ExtrudeShape<TriangleMeshShapeSTL>>(thickness, full_path_to_file, translation, scaling);
        subtract<TransformShape<GeometricShapeBox>>(Transform(Rotation3d(inlet_rotation), Vec3d(inlet_translation)), inlet_half);
        subtract<TransformShape<GeometricShapeBox>>(Transform(Rotation3d(outlet_01_rotation), Vec3d(outlet_01_translation)), outlet_01_half);
        subtract<TransformShape<GeometricShapeBox>>(Transform(Rotation3d(outlet_02_rotation), Vec3d(outlet_02_translation)), outlet_02_half);
    }

    TriangleMeshShapeSTL* getMeshShape() const
    {
        return mesh_shape_.get();
    }

private:
    std::unique_ptr<TriangleMeshShapeSTL> mesh_shape_;
};

class ShellShape : public ComplexShape
{
public:
    explicit ShellShape(const std::string &shape_name) : ComplexShape(shape_name)
    {
        add<ExtrudeShape<TriangleMeshShapeSTL>>(thickness, full_path_to_file, translation, scaling);
        subtract<TriangleMeshShapeSTL>(full_path_to_file, translation, scaling);
    }
};
//-----------------------------------------------------------------------------------------------------------
//	Main program starts here.
//-----------------------------------------------------------------------------------------------------------
int main(int ac, char *av[])
{
    //----------------------------------------------------------------------
    //	Build up -- a SPHSystem
    //----------------------------------------------------------------------
    SPHSystem sph_system(system_domain_bounds, dp_0);
    sph_system.setRunParticleRelaxation(true); // Tag for run particle relaxation for body-fitted distribution
    sph_system.setReloadParticles(false);
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment();
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.
    //----------------------------------------------------------------------
    SolidBodyFromMesh solid_body_from_mesh("SolidBodyFromMesh");
    TriangleMeshShapeSTL* mesh_shape = solid_body_from_mesh.getMeshShape();

    RealBody imported_model(sph_system, makeShared<SolidBodyFromMesh>("ShellShape"));
    imported_model.defineAdaptation<SPHAdaptation>(1.15, 1.0);
    //imported_model.defineBodyLevelSetShape(2.0)->correctLevelSetSign()->writeLevelSet(sph_system);
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? imported_model.generateParticles<SurfaceParticles, Reload>(imported_model.getName())
        : imported_model.generateParticles<SurfaceParticles, FromSTLFile>(mesh_shape);
    
    //auto shell_particles = dynamic_cast<SurfaceParticles *>(&imported_model.getBaseParticles());
    //// test volume
    //StdLargeVec<Real> &Vol_ = *shell_particles->getVariableDataByName<Real>("VolumetricMeasure");
    //Real total_volume = std::accumulate(Vol_.begin(), Vol_.end(), 0.0);
    //std::cout << "total_volume: " << total_volume << std::endl;

    // aligned box for detect useless inlet and outlet partcles for wall
    /*RealBody test_body_in(
        sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(inlet_rotation), Vec3d(inlet_translation)), inlet_half, "TestBodyIn"));
    test_body_in.generateParticles<BaseParticles, Lattice>();*/
    BodyAlignedBoxByCell inlet_detection_box(imported_model,
                                             makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(inlet_rotation), Vec3d(inlet_translation)), inlet_half));

    /*RealBody test_body_out01(
        sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_01_rotation), Vec3d(outlet_01_translation)), outlet_01_half, "TestBodyOut01"));
    test_body_out01.generateParticles<BaseParticles, Lattice>();*/
    BodyAlignedBoxByCell outlet01_detection_box(imported_model,
                                                makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_01_rotation), Vec3d(outlet_01_translation)), outlet_01_half));

    /*RealBody test_body_out02(
        sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_02_rotation), Vec3d(outlet_02_translation)), outlet_02_half, "TestBodyOut02"));
    test_body_out02.generateParticles<BaseParticles, Lattice>();*/
    BodyAlignedBoxByCell outlet02_detection_box(imported_model,
                                                makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_02_rotation), Vec3d(outlet_02_translation)), outlet_02_half));

    if (sph_system.RunParticleRelaxation())
    {
        //----------------------------------------------------------------------
        //	Define body relation map.
        //	The contact map gives the topological connections between the bodies.
        //	Basically the the range of bodies to build neighbor particle lists.
        //  Generally, we first define all the inner relations, then the contact relations.
        //  At last, we define the complex relaxations by combining previous defined
        //  inner and contact relations.
        //----------------------------------------------------------------------
        InnerRelation imported_model_inner(imported_model);
        //----------------------------------------------------------------------
        //	Methods used for particle relaxation.
        //----------------------------------------------------------------------
        using namespace relax_dynamics;
        //SimpleDynamics<RandomizeParticlePosition> random_imported_model_particles(imported_model);
        /** A  Physics relaxation step. */
        SurfaceRelaxationStep relaxation_step_inner(imported_model_inner);
        //ShellNormalDirectionPrediction shell_normal_prediction(imported_model_inner, thickness, cos(Pi / 3.75));

        // here, need a class to switch particles in aligned box to ghost particles (not real particles)
        /*SimpleDynamics<relax_dynamics::ParticlesInAlignedBoxDetectionByCell> inlet_particles_detection(inlet_detection_box);
        SimpleDynamics<relax_dynamics::ParticlesInAlignedBoxDetectionByCell> outlet01_particles_detection(outlet01_detection_box);
        SimpleDynamics<relax_dynamics::ParticlesInAlignedBoxDetectionByCell> outlet02_particles_detection(outlet02_detection_box);*/

        /** Write the body state to Vtp file. */
        BodyStatesRecordingToVtp write_imported_model_to_vtp({imported_model});
        /** Write the particle reload files. */
        ReloadParticleIO write_particle_reload_files(imported_model);
        //----------------------------------------------------------------------
        //	Particle relaxation starts here.
        //----------------------------------------------------------------------
        //random_imported_model_particles.exec(0.25);
        relaxation_step_inner.getOnSurfaceBounding().exec();
        write_imported_model_to_vtp.writeToFile(0.0);
        imported_model.updateCellLinkedList();
        // write_cell_linked_list.writeToFile(0.0);
        //----------------------------------------------------------------------
        //	Particle relaxation time stepping start here.
        //----------------------------------------------------------------------
        int ite_p = 0;
        while (ite_p < 5000)
        {
            relaxation_step_inner.exec();
            ite_p += 1;
            if (ite_p % 500 == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "Relaxation steps for the imported model N = " << ite_p << "\n";
                write_imported_model_to_vtp.writeToFile(ite_p);
            }
        }
        std::cout << "The physics relaxation process of imported model finish !" << std::endl;

        /*inlet_particles_detection.exec();
        imported_model.updateCellLinkedListWithParticleSort(100);
        outlet01_particles_detection.exec();
        imported_model.updateCellLinkedListWithParticleSort(100);
        outlet02_particles_detection.exec();
        imported_model.updateCellLinkedListWithParticleSort(100);*/
        write_imported_model_to_vtp.writeToFile(ite_p);
        write_particle_reload_files.writeToFile(0);

        //shell_normal_prediction.exec();

        return 0;
    }

    //imported_model.updateCellLinkedList();

    //// here, need a class to switch particles in aligned box to ghost particles (not real particles)
    //SimpleDynamics<relax_dynamics::ParticlesInAlignedBoxDetectionByCell> inlet_particles_detection(inlet_detection_box);
    //SimpleDynamics<relax_dynamics::ParticlesInAlignedBoxDetectionByCell> outlet01_particles_detection(outlet01_detection_box);
    //SimpleDynamics<relax_dynamics::ParticlesInAlignedBoxDetectionByCell> outlet02_particles_detection(outlet02_detection_box);

    BodyStatesRecordingToVtp write_body_states(sph_system);
    write_body_states.addToWrite<Real>(imported_model, "VolumetricMeasure");

    //inlet_particles_detection.exec();
    //imported_model.updateCellLinkedListWithParticleSort(100);

    //outlet01_particles_detection.exec();
    //imported_model.updateCellLinkedListWithParticleSort(100);

    //outlet02_particles_detection.exec();
    //imported_model.updateCellLinkedListWithParticleSort(100);

    write_body_states.writeToFile();

    return 0;
}