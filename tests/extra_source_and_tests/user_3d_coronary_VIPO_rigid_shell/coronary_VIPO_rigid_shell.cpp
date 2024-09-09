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
std::string full_path_to_file = "./input/normal_fluid_repair.stl";
std::string full_vtp_file_path = "./input/converted_normal1_mesh.vtp";
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Vec3d translation(0.0, 0.0, 0.0);
Real scaling = 0.03935;
Vec3d domain_lower_bound(-375.0 * scaling, 100.0 * scaling, -340 * scaling);
Vec3d domain_upper_bound(-100.0 * scaling, 360.0 * scaling, 0.0 * scaling);
BoundingBox system_domain_bounds(domain_lower_bound, domain_upper_bound);
Real dp_0 = 0.05;
Real shell_resolution = dp_0 / 2;
Real thickness = 1.0 * shell_resolution;
//----------------------------------------------------------------------
//	define the imported model.
//----------------------------------------------------------------------
class SolidBodyFromMesh : public ComplexShape
{
public:
    explicit SolidBodyFromMesh(const std::string &shape_name) : ComplexShape(shape_name),
        mesh_shape_(new TriangleMeshShapeSTL(full_path_to_file, translation, scaling))
    {
        if (!mesh_shape_->isValid())
        {
            std::cerr << "Error: Failed to load the mesh from file: " << full_path_to_file << std::endl;
            throw std::runtime_error("Mesh loading failed");
        }

        add<TriangleMeshShapeSTL>(full_path_to_file, translation, scaling);
        //add<ExtrudeShape<TriangleMeshShapeSTL>>(thickness, full_path_to_file, translation, scaling);
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
        add<ExtrudeShape<TriangleMeshShapeSTL>>(-shell_resolution/2, full_path_to_file, translation, scaling);
    }
};

class FromVTPFile;
template <>
class ParticleGenerator<SurfaceParticles, FromVTPFile> : public ParticleGenerator<SurfaceParticles>
{
    Real mesh_total_area_;
    Real particle_spacing_;
    const Real thickness_;
    Real avg_particle_volume_;
    size_t planned_number_of_particles_;

    std::vector<Vec3d> vertex_positions_;
    std::vector<std::array<int, 3>> faces_;
    Shape &initial_shape_;

public:
    explicit ParticleGenerator(SPHBody& sph_body, SurfaceParticles &surface_particles, const std::string& vtp_file_path) 
        : ParticleGenerator<SurfaceParticles>(sph_body, surface_particles),
        mesh_total_area_(0),
        particle_spacing_(sph_body.sph_adaptation_->ReferenceSpacing()),
        thickness_(particle_spacing_),
        avg_particle_volume_(pow(particle_spacing_, Dimensions - 1) * thickness_),
        planned_number_of_particles_(0),
        initial_shape_(sph_body.getInitialShape()) 
    {
        if (!readVTPFile(vtp_file_path))
        {
            std::cerr << "Error: VTP file could not be read!" << std::endl;
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
        
        int num_faces = faces_.size();
        std::cout << "num_faces calculation = " << num_faces << std::endl;

        // Calculate total volume
        std::vector<Real> face_areas(num_faces);
        for (int i = 0; i < num_faces; ++i)
        {
            Vec3d vertices[3];
            for (int j = 0; j < 3; ++j)
            {
                const auto& pos = vertex_positions_[faces_[i][j]];
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
                const auto& pos = vertex_positions_[faces_[i][j]];
                vertices[j] = Vec3d(pos[0], pos[1], pos[2]);
            }

            Real random_real = unif(rng);
            if (random_real <= interval && base_particles_.TotalRealParticles() < planned_number_of_particles_)
            {
                // Generate particle at the center of this triangle face
                //generateParticleAtFaceCenter(vertices);
                
                // Generate particles on this triangle face, unequal
                int particles_per_face = std::max(1, int(planned_number_of_particles_ * (face_areas[i] / mesh_total_area_)));
                generateParticlesOnFace(vertices, particles_per_face);
            }
        }
    }

private:
    bool readVTPFile(const std::string& vtp_file)
    {
        std::ifstream file(vtp_file);
        if (!file.is_open())
        {
            std::cerr << "Could not open file: " << vtp_file << std::endl;
            return false;
        }

        std::string line;
        bool reading_points = false;
        bool reading_faces = false;
        bool reading_points_data = false;
        bool reading_faces_data = false;

        vertex_positions_.reserve(50000);
        faces_.reserve(50000);

        int read_face_num = 0;

        while (std::getline(file, line))
        {
            if (line.find("<Points>") != std::string::npos)
            {
                reading_points = true;
                continue;
            }
            if (line.find("</Points>") != std::string::npos)
            {
                reading_points = false;
                continue;
            }
            if (line.find("<Polys>") != std::string::npos)
            {
                reading_faces = true;
                continue;
            }
            if (line.find("</Polys>") != std::string::npos)
            {
                reading_faces = false;
                continue;
            }
            if (reading_points && line.find("<DataArray") != std::string::npos)
            {
                reading_points_data = true;
                continue;
            }
            if (reading_faces && line.find("<DataArray type=\"Int32\" Name=\"connectivity\"") != std::string::npos)
            {
                reading_faces_data = true;
                continue;
            }
            if (reading_points_data && line.find("</DataArray>") != std::string::npos)
            {
                reading_points_data = false;
                continue;
            }
            if (reading_faces_data && line.find("</DataArray>") != std::string::npos)
            {
                reading_faces_data = false;
                continue;
            }

            if (reading_points_data)
            {
                std::istringstream iss(line);
                Real x, y, z;
                if (iss >> x >> y >> z)
                {
                    vertex_positions_.push_back({x, y, z});
                }
            }

            if (reading_faces_data)
            {
                std::istringstream iss(line);
                int v1, v2, v3;
                if (iss >> v1 >> v2 >> v3)
                {
                    faces_.push_back({v1, v2, v3});
                }
            }
        }

        std::cout << "Read VTP file successfully!" << std::endl;

        // for debug
        // std::cout << "faces_[39097][0] = " << faces_[39097][0] << ", faces_[39097][1] = " << faces_[39097][1] << ", faces_[39097][2] = " << faces_[39097][2] << std::endl;
        // std::cout << "faces_[39098][0] = " << faces_[39098][0] << ", faces_[39098][1] = " << faces_[39098][1] << ", faces_[39098][2] = " << faces_[39098][2] << std::endl;

        return true;
    }

    Real calculateEachFaceArea(const Vec3d vertices[3])
    {
        Vec3d edge1 = vertices[1] - vertices[0];
        Vec3d edge2 = vertices[2] - vertices[0];
        return 0.5 * edge1.cross(edge2).norm();
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
int simTK_resolution(20);
Vecd standard_direction(1, 0, 0);

// inlet: R=41.7567, (-203.6015, 204.1509, -135.3577), (0.2987, 0.1312, 0.9445)
Real DW_in = 41.7567 * 2 * scaling;
Vec3d inlet_half = Vec3d(2.0 * dp_0, 44.0 * scaling, 44.0 * scaling);
Vec3d inlet_normal(-0.2987, -0.1312, -0.9445);
Vec3d inlet_cut_translation = Vec3d(-203.6015, 204.1509, -135.3577) * scaling + inlet_normal * (1.0 * dp_0 + 1.0 * (dp_0 - shell_resolution));
Vec3d inlet_buffer_translation = Vec3d(-203.6015, 204.1509, -135.3577) * scaling - inlet_normal * 2.0 * dp_0;
RotationResult inlet_rotation_result = RotationCalculator(inlet_normal, standard_direction);
Rotation3d inlet_rotation(inlet_rotation_result.angle, inlet_rotation_result.axis);
Rotation3d inlet_disposer_rotation(inlet_rotation_result.angle, inlet_rotation_result.axis);
Rotation3d inlet_emitter_rotation(inlet_rotation_result.angle + Pi, inlet_rotation_result.axis);

// outlet main: R=36.1590, (-172.2628, 205.9036, -19.8868), (0.2678, 0.3191, -0.9084)
Real DW_out_main = 36.1590 * 2 * scaling;
Vec3d outlet_half_main = Vec3d(2.0 * dp_0, 36.5 * scaling, 36.5 * scaling);
Vec3d outlet_normal_main(-0.2678, -0.3191, 0.9084);
Vec3d outlet_cut_translation_main = Vec3d(-172.2628, 205.9036, -19.8868) * scaling + outlet_normal_main * (1.0 * dp_0 + 1.0 * (dp_0 - shell_resolution));
Vec3d outlet_buffer_translation_main = Vec3d(-172.2628, 205.9036, -19.8868) * scaling - outlet_normal_main * 2.0 * dp_0;
RotationResult outlet_rotation_result_main = RotationCalculator(outlet_normal_main, standard_direction);
Rotation3d outlet_disposer_rotation_main(outlet_rotation_result_main.angle, outlet_rotation_result_main.axis);
Rotation3d outlet_emitter_rotation_main(outlet_rotation_result_main.angle + Pi, outlet_rotation_result_main.axis);

// outlet x_pos 01: R=2.6964, (-207.4362, 136.7848, -252.6892), (0.636, 0.771, -0.022)
Real DW_out_left_01 = 2.6964 * 2 * scaling;
Vec3d outlet_half_left_01 = Vec3d(2.0 * dp_0, 2.8 * scaling, 2.8 * scaling);
Vec3d outlet_normal_left_01(-0.636, -0.771, 0.022);
Vec3d outlet_cut_translation_left_01 = Vec3d(-207.4362, 136.7848, -252.6892) * scaling + outlet_normal_left_01 * (1.0 * dp_0 + 1.0 * (dp_0 - shell_resolution));
Vec3d outlet_buffer_translation_left_01 = Vec3d(-207.4362, 136.7848, -252.6892) * scaling - outlet_normal_left_01 * 2.0 * dp_0;
RotationResult outlet_rotation_result_left_01 = RotationCalculator(outlet_normal_left_01, standard_direction);
Rotation3d outlet_disposer_rotation_left_01(outlet_rotation_result_left_01.angle, outlet_rotation_result_left_01.axis);
Rotation3d outlet_emitter_rotation_left_01(outlet_rotation_result_left_01.angle + Pi, outlet_rotation_result_left_01.axis);

// outlet x_pos 02: R=2.8306, (-193.2735, 337.4625, -270.2884), (-0.6714, 0.3331, -0.6620)
Real DW_out_left_02 = 2.8306 * 2 * scaling;
Vec3d outlet_half_left_02 = Vec3d(2.0 * dp_0, 3.0 * scaling, 3.0 * scaling);
Vec3d outlet_normal_left_02(-0.6714, 0.3331, -0.6620);
Vec3d outlet_cut_translation_left_02 = Vec3d(-193.2735, 337.4625, -270.2884) * scaling + outlet_normal_left_02 * (1.0 * dp_0 + 1.0 * (dp_0 - shell_resolution));
Vec3d outlet_buffer_translation_left_02 = Vec3d(-193.2735, 337.4625, -270.2884) * scaling - outlet_normal_left_02 * 2.0 * dp_0;
RotationResult outlet_rotation_result_left_02 = RotationCalculator(outlet_normal_left_02, standard_direction);
Rotation3d outlet_disposer_rotation_left_02(outlet_rotation_result_left_02.angle, outlet_rotation_result_left_02.axis);
Rotation3d outlet_emitter_rotation_left_02(outlet_rotation_result_left_02.angle + Pi, outlet_rotation_result_left_02.axis);

// outlet x_pos 03: R=2.2804, (-165.5566, 326.1601, -139.9323), (0.6563, -0.6250, 0.4226)
Real DW_out_left_03 = 2.2804 * 2 * scaling;
Vec3d outlet_half_left_03 = Vec3d(2.0 * dp_0, 2.5 * scaling, 2.5 * scaling);
Vec3d outlet_normal_left_03(-0.6563, 0.6250, -0.4226);
Vec3d outlet_cut_translation_left_03 = Vec3d(-165.5566, 326.1601, -139.9323) * scaling + outlet_normal_left_03 * (1.0 * dp_0 + 1.0 * (dp_0 - shell_resolution));
Vec3d outlet_buffer_translation_left_03 = Vec3d(-165.5566, 326.1601, -139.9323) * scaling - outlet_normal_left_03 * 2.0 * dp_0;
RotationResult outlet_rotation_result_left_03 = RotationCalculator(outlet_normal_left_03, standard_direction);
Rotation3d outlet_disposer_rotation_left_03(outlet_rotation_result_left_03.angle, outlet_rotation_result_left_03.axis);
Rotation3d outlet_emitter_rotation_left_03(outlet_rotation_result_left_03.angle + Pi, outlet_rotation_result_left_03.axis);

// outlet x_neg_front 01: R=2.6437, (-307.8, 312.1402, -333.2), (-0.185, -0.967, -0.176)
Real DW_out_rightF_01 = 2.6437 * 2 * scaling;
Vec3d outlet_half_rightF_01 = Vec3d(2.0 * dp_0, 2.8 * scaling, 2.8 * scaling);
Vec3d outlet_normal_rightF_01(-0.185, -0.967, -0.176);
Vec3d outlet_cut_translation_rightF_01 = Vec3d(-307.8, 312.1402, -333.2) * scaling + outlet_normal_rightF_01 * (1.0 * dp_0 + 1.0 * (dp_0 - shell_resolution));
Vec3d outlet_buffer_translation_rightF_01 = Vec3d(-307.8, 312.1402, -333.2) * scaling - outlet_normal_rightF_01 * 2.0 * dp_0;
RotationResult outlet_rotation_result_rightF_01 = RotationCalculator(outlet_normal_rightF_01, standard_direction);
Rotation3d outlet_disposer_rotation_rightF_01(outlet_rotation_result_rightF_01.angle, outlet_rotation_result_rightF_01.axis);
Rotation3d outlet_emitter_rotation_rightF_01(outlet_rotation_result_rightF_01.angle + Pi, outlet_rotation_result_rightF_01.axis);

// outlet x_neg_front 02: R=1.5424, (-369.1252, 235.2617, -193.7022), (-0.501, 0.059, -0.863)
Real DW_out_rightF_02 = 1.5424 * 2 * scaling;
Vec3d outlet_half_rightF_02 = Vec3d(2.0 * dp_0, 1.8 * scaling, 1.8 * scaling);
Vec3d outlet_normal_rightF_02(-0.501, 0.059, -0.863);
Vec3d outlet_cut_translation_rightF_02 = Vec3d(-369.1252, 235.2617, -193.7022) * scaling + outlet_normal_rightF_02 * (1.0 * dp_0 + 1.0 * (dp_0 - shell_resolution));
Vec3d outlet_buffer_translation_rightF_02 = Vec3d(-369.1252, 235.2617, -193.7022) * scaling - outlet_normal_rightF_02 * 2.0 * dp_0;
RotationResult outlet_rotation_result_rightF_02 = RotationCalculator(outlet_normal_rightF_02, standard_direction);
Rotation3d outlet_disposer_rotation_rightF_02(outlet_rotation_result_rightF_02.angle, outlet_rotation_result_rightF_02.axis);
Rotation3d outlet_emitter_rotation_rightF_02(outlet_rotation_result_rightF_02.angle + Pi, outlet_rotation_result_rightF_02.axis);

// outlet x_neg_behind 01: R=1.5743, (-268.3522, 116.0357, -182.4896), (0.325, -0.086, -0.942)
Real DW_out_rightB_01 = 1.5743 * 2 * scaling;
Vec3d outlet_half_rightB_01 = Vec3d(2.0 * dp_0, 1.8 * scaling, 1.8 * scaling);
Vec3d outlet_normal_rightB_01(0.325, -0.086, -0.942);
Vec3d outlet_cut_translation_rightB_01 = Vec3d(-268.3522, 116.0357, -182.4896) * scaling + outlet_normal_rightB_01 * (1.0 * dp_0 + 1.0 * (dp_0 - shell_resolution));
Vec3d outlet_buffer_translation_rightB_01 = Vec3d(-268.3522, 116.0357, -182.4896) * scaling - outlet_normal_rightB_01 * 2.0 * dp_0;
RotationResult outlet_rotation_result_rightB_01 = RotationCalculator(outlet_normal_rightB_01, standard_direction);
Rotation3d outlet_disposer_rotation_rightB_01(outlet_rotation_result_rightB_01.angle, outlet_rotation_result_rightB_01.axis);
Rotation3d outlet_emitter_rotation_rightB_01(outlet_rotation_result_rightB_01.angle + Pi, outlet_rotation_result_rightB_01.axis);

// outlet x_neg_behind 02: R=1.8204, (-329.0846, 180.5258, -274.3232), (-0.1095, 0.9194, -0.3777)
Real DW_out_rightB_02 = 1.8204 * 2 * scaling;
Vec3d outlet_half_rightB_02 = Vec3d(2.0 * dp_0, 2.0 * scaling, 2.0 * scaling);
Vec3d outlet_normal_rightB_02(-0.1095, 0.9194, -0.3777);
Vec3d outlet_cut_translation_rightB_02 = Vec3d(-329.0846, 180.5258, -274.3232) * scaling + outlet_normal_rightB_02 * (1.0 * dp_0 + 1.0 * (dp_0 - shell_resolution));
Vec3d outlet_buffer_translation_rightB_02 = Vec3d(-329.0846, 180.5258, -274.3232) * scaling - outlet_normal_rightB_02 * 2.0 * dp_0;
RotationResult outlet_rotation_result_rightB_02 = RotationCalculator(outlet_normal_rightB_02, standard_direction);
Rotation3d outlet_disposer_rotation_rightB_02(outlet_rotation_result_rightB_02.angle, outlet_rotation_result_rightB_02.axis);
Rotation3d outlet_emitter_rotation_rightB_02(outlet_rotation_result_rightB_02.angle + Pi, outlet_rotation_result_rightB_02.axis);

// outlet x_neg_behind 03: R=1.5491, (-342.1711, 197.1107, -277.8681), (0.1992, 0.5114, -0.8361)
Real DW_out_rightB_03 = 1.5491 * 2 * scaling;
Vec3d outlet_half_rightB_03 = Vec3d(2.0 * dp_0, 1.8 * scaling, 1.8 * scaling);
Vec3d outlet_normal_rightB_03(0.1992, 0.5114, -0.8361);
Vec3d outlet_cut_translation_rightB_03 = Vec3d(-342.1711, 197.1107, -277.8681) * scaling + outlet_normal_rightB_03 * (1.0 * dp_0 + 1.0 * (dp_0 - shell_resolution));
Vec3d outlet_buffer_translation_rightB_03 = Vec3d(-342.1711, 197.1107, -277.8681) * scaling - outlet_normal_rightB_03 * 2.0 * dp_0;
RotationResult outlet_rotation_result_rightB_03 = RotationCalculator(outlet_normal_rightB_03, standard_direction);
Rotation3d outlet_disposer_rotation_rightB_03(outlet_rotation_result_rightB_03.angle, outlet_rotation_result_rightB_03.axis);
Rotation3d outlet_emitter_rotation_rightB_03(outlet_rotation_result_rightB_03.angle + Pi, outlet_rotation_result_rightB_03.axis);

// outlet x_neg_behind 04: R=2.1598, (-362.0112, 200.5693, -253.8417), (0.3694, 0.6067, -0.7044)
Real DW_out_rightB_04 = 2.1598 * 2 * scaling;
Vec3d outlet_half_rightB_04 = Vec3d(2.0 * dp_0, 2.5 * scaling, 2.5 * scaling);
Vec3d outlet_normal_rightB_04(0.3694, 0.6067, -0.7044);
Vec3d outlet_cut_translation_rightB_04 = Vec3d(-362.0112, 200.5693, -253.8417) * scaling + outlet_normal_rightB_04 * (1.0 * dp_0 + 1.0 * (dp_0 - shell_resolution));
Vec3d outlet_buffer_translation_rightB_04 = Vec3d(-362.0112, 200.5693, -253.8417) * scaling - outlet_normal_rightB_04 * 2.0 * dp_0;
RotationResult outlet_rotation_result_rightB_04 = RotationCalculator(outlet_normal_rightB_04, standard_direction);
Rotation3d outlet_disposer_rotation_rightB_04(outlet_rotation_result_rightB_04.angle, outlet_rotation_result_rightB_04.axis);
Rotation3d outlet_emitter_rotation_rightB_04(outlet_rotation_result_rightB_04.angle + Pi, outlet_rotation_result_rightB_04.axis);
//----------------------------------------------------------------------
//	Global parameters on the fluid properties
//----------------------------------------------------------------------
Real rho0_f = 1060; /**< Reference density of fluid. */
Real U_f = 0.5;    /**< Characteristic velocity. */
/** Reference sound speed needs to consider the flow speed in the narrow channels. */
Real c_f = 10.0 * U_f * SMAX(Real(1), DW_in / (DW_out_main + DW_out_left_01 + DW_out_left_02 + DW_out_rightF_01 + 
    DW_out_rightF_02 + DW_out_rightB_01 + DW_out_rightB_02 + DW_out_rightB_03 + DW_out_rightB_04));
Real mu_f = 0.00355; /**< Dynamics viscosity. */
Real Outlet_pressure = 0;

//Real rho0_s = 1120;                /** Normalized density. */
//Real Youngs_modulus = 1.08e6;    /** Normalized Youngs Modulus. */
//Real poisson = 0.49;               /** Poisson ratio. */
//Real physical_viscosity = 0.25 * sqrt(rho0_s * Youngs_modulus) * 55.0 * scaling; /** physical damping */
//----------------------------------------------------------------------
//	Inflow velocity
//----------------------------------------------------------------------
struct InflowVelocity
{
    Real u_ref_, t_ref_, interval_;

    template <class BoundaryConditionType>
    InflowVelocity(BoundaryConditionType &boundary_condition)
        : u_ref_(0.1), t_ref_(0.218), interval_(0.5) {}

    Vecd operator()(Vecd &position, Vecd &velocity)
    {
        Vecd target_velocity = velocity;
        Real run_time = GlobalStaticVariables::physical_time_;
        int n = static_cast<int>(run_time / interval_);
        Real t_in_cycle = run_time - n * interval_;

        target_velocity[0] = t_in_cycle < t_ref_ ? 0.5 * sin(4 * Pi * (run_time + 0.0160236)) : u_ref_;
        return target_velocity;
    }
};

class TimeDependentAcceleration : public Gravity
{
    Real t_ref_, du_ave_dt_, interval_;

  public:
    explicit TimeDependentAcceleration(Vecd gravity_vector)
        : Gravity(gravity_vector), t_ref_(0.218), du_ave_dt_(0), interval_(0.5) {}

    virtual Vecd InducedAcceleration(const Vecd &position) override
    {
        Real run_time = GlobalStaticVariables::physical_time_;
        int n = static_cast<int>(run_time / interval_);
        Real t_in_cycle = run_time - n * interval_;

        du_ave_dt_ = 0.5 * 4 * Pi * cos(4 * Pi * run_time);

        return t_in_cycle < t_ref_ ? Vecd(0.0, 0.0, du_ave_dt_) : global_acceleration_;
    }
};

//----------------------------------------------------------------------
//	Pressure boundary definition.
//----------------------------------------------------------------------
struct InletInflowPressure
{
    template <class BoundaryConditionType>
    InletInflowPressure(BoundaryConditionType &boundary_condition) {}

    Real operator()(Real &p_)
    {
        return p_;
    }
};

struct OutletInflowPressure
{
    template <class BoundaryConditionType>
    OutletInflowPressure(BoundaryConditionType &boundary_condition) {}

    Real operator()(Real &p_)
    {
        /*constant pressure*/
        Real pressure = 0;
        return pressure;
    }
};

//class BoundaryGeometry : public BodyPartByParticle
//{
//  public:
//    BoundaryGeometry(SPHBody &body, const std::string &body_part_name)
//        : BodyPartByParticle(body, body_part_name)
//    {
//        TaggingParticleMethod tagging_particle_method = std::bind(&BoundaryGeometry::tagManually, this, _1);
//        tagParticles(tagging_particle_method);
//    };
//    virtual ~BoundaryGeometry(){};
//
//  private:
//    void tagManually(size_t index_i)
//    {
//        if (base_particles_.ParticlePositions()[index_i][2] < -30.8885 * scaling + 0.9935 * 4 * dp_0
//            || (base_particles_.ParticlePositions()[index_i][2] > 21.7855 * scaling - 0.9488 * 4 * dp_0
//                && base_particles_.ParticlePositions()[index_i][0] <= 0)
//            || (base_particles_.ParticlePositions()[index_i][2] > 18.6389 * scaling - 0.9972 * 4 * dp_0
//                && base_particles_.ParticlePositions()[index_i][0] > 0))
//        {
//            body_part_particles_.push_back(index_i);
//        }
//    };
//};

//-----------------------------------------------------------------------------------------------------------
//	Main program starts here.
//-----------------------------------------------------------------------------------------------------------
int main(int ac, char *av[])
{
    //----------------------------------------------------------------------
    //	Build up the environment of a SPHSystem with global controls.
    //----------------------------------------------------------------------
    SPHSystem sph_system(system_domain_bounds, dp_0);
    sph_system.setRunParticleRelaxation(false); // Tag for run particle relaxation for body-fitted distribution
    sph_system.setReloadParticles(true);       // Tag for computation with save particles distribution
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment();
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.cd
    //----------------------------------------------------------------------
    SolidBody shell_body(sph_system, makeShared<SolidBodyFromMesh>("ShellBody"));
    shell_body.defineAdaptation<SPHAdaptation>(1.15, dp_0/shell_resolution);
    shell_body.defineMaterial<Solid>();
    if (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
    {
        shell_body.generateParticles<SurfaceParticles, Reload>(shell_body.getName());
    }
    else
    {
        shell_body.defineBodyLevelSetShape(2.0)->correctLevelSetSign()->writeLevelSet(sph_system);
        shell_body.generateParticles<SurfaceParticles, FromVTPFile>(full_vtp_file_path);
    }

    FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBody"));
    water_block.defineMaterial<WeaklyCompressibleFluid>(rho0_f, c_f, mu_f);
    ParticleBuffer<ReserveSizeFactor> in_outlet_particle_buffer(0.5);
    //water_block.generateParticlesWithReserve<BaseParticles, Lattice>(in_outlet_particle_buffer);
    if (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
    {
        water_block.generateParticlesWithReserve<BaseParticles, Reload>(in_outlet_particle_buffer, water_block.getName());
    }
    else
    {
        water_block.defineBodyLevelSetShape(2.0)->correctLevelSetSign()->cleanLevelSet();
        water_block.generateParticles<BaseParticles, Lattice>();
    }
    //----------------------------------------------------------------------
    //	SPH Particle relaxation section
    //----------------------------------------------------------------------
    /** check whether run particle relaxation for body fitted particle distribution. */
    if (sph_system.RunParticleRelaxation())
    {
        // for shell
        BodyAlignedCylinderByCell inlet_detection_cylinder(shell_body, 
            makeShared<AlignedCylinderShape>(xAxis, SimTK::UnitVec3(-0.2987, -0.1312, -0.9445), inlet_half[1], inlet_half[0], simTK_resolution, inlet_cut_translation));
        BodyAlignedBoxByCell outlet_main_detection_box(shell_body, 
            makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_main), Vec3d(outlet_cut_translation_main)), outlet_half_main));
        BodyAlignedBoxByCell outlet_left01_detection_box(shell_body, 
            makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_left_01), Vec3d(outlet_cut_translation_left_01)), outlet_half_left_01));
        BodyAlignedBoxByCell outlet_left02_detection_box(shell_body, 
            makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_left_02), Vec3d(outlet_cut_translation_left_02)), outlet_half_left_02));
        BodyAlignedBoxByCell outlet_left03_detection_box(shell_body, 
            makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_left_03), Vec3d(outlet_cut_translation_left_03)), outlet_half_left_03));
        BodyAlignedBoxByCell outlet_rightF01_detection_box(shell_body, 
            makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_rightF_01), Vec3d(outlet_cut_translation_rightF_01)), outlet_half_rightF_01));
        BodyAlignedBoxByCell outlet_rightF02_detection_box(shell_body, 
            makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_rightF_02), Vec3d(outlet_cut_translation_rightF_02)), outlet_half_rightF_02));
        BodyAlignedBoxByCell outlet_rightB01_detection_box(shell_body, 
            makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_rightB_01), Vec3d(outlet_cut_translation_rightB_01)), outlet_half_rightB_01));
        BodyAlignedBoxByCell outlet_rightB02_detection_box(shell_body, 
            makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_rightB_02), Vec3d(outlet_cut_translation_rightB_02)), outlet_half_rightB_02));
        BodyAlignedBoxByCell outlet_rightB03_detection_box(shell_body, 
            makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_rightB_03), Vec3d(outlet_cut_translation_rightB_03)), outlet_half_rightB_03));
        BodyAlignedBoxByCell outlet_rightB04_detection_box(shell_body, 
            makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_rightB_04), Vec3d(outlet_cut_translation_rightB_04)), outlet_half_rightB_04));

        InnerRelation shell_inner(shell_body);
        InnerRelation blood_inner(water_block);
        //----------------------------------------------------------------------
        //	Methods used for particle relaxation.
        //----------------------------------------------------------------------
        using namespace relax_dynamics;
        //SimpleDynamics<RandomizeParticlePosition> random_imported_model_particles(imported_model);
        /** A  Physics relaxation step. */
        SurfaceRelaxationStep relaxation_step_inner_shell(shell_inner);
        RelaxationStepInner relaxation_step_inner_blood(blood_inner);
        ShellNormalDirectionPrediction shell_normal_prediction(shell_inner, thickness);

        SimpleDynamics<DeleteParticlesInCylinder> inlet_particles_detection(inlet_detection_cylinder);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet_main_particles_detection(outlet_main_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet_left01_particles_detection(outlet_left01_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet_left02_particles_detection(outlet_left02_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet_left03_particles_detection(outlet_left03_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet_rightF01_particles_detection(outlet_rightF01_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet_rightF02_particles_detection(outlet_rightF02_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet_rightB01_particles_detection(outlet_rightB01_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet_rightB02_particles_detection(outlet_rightB02_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet_rightB03_particles_detection(outlet_rightB03_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet_rightB04_particles_detection(outlet_rightB04_detection_box);

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
        relaxation_step_inner_shell.getOnSurfaceBounding().exec();
        relaxation_step_inner_blood.SurfaceBounding().exec();
        write_shell_to_vtp.writeToFile(0.0);
        write_blood_to_vtp.writeToFile(0.0);
        shell_body.updateCellLinkedList();
        //----------------------------------------------------------------------
        //	Particle relaxation time stepping start here.
        //----------------------------------------------------------------------
        int ite_p = 0;
        while (ite_p < 2000)
        {
            relaxation_step_inner_shell.exec();
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
        shell_body.updateCellLinkedListWithParticleSort(100);
        outlet_main_particles_detection.exec();
        shell_body.updateCellLinkedListWithParticleSort(100);
        outlet_left01_particles_detection.exec();
        shell_body.updateCellLinkedListWithParticleSort(100);
        outlet_left02_particles_detection.exec();
        shell_body.updateCellLinkedListWithParticleSort(100);
        outlet_left03_particles_detection.exec();
        shell_body.updateCellLinkedListWithParticleSort(100);
        outlet_rightF01_particles_detection.exec();
        shell_body.updateCellLinkedListWithParticleSort(100);
        outlet_rightF02_particles_detection.exec();
        shell_body.updateCellLinkedListWithParticleSort(100);
        outlet_rightB01_particles_detection.exec();
        shell_body.updateCellLinkedListWithParticleSort(100);
        outlet_rightB02_particles_detection.exec();
        shell_body.updateCellLinkedListWithParticleSort(100);
        outlet_rightB03_particles_detection.exec();
        shell_body.updateCellLinkedListWithParticleSort(100);
        outlet_rightB04_particles_detection.exec();
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
    TimeDependentAcceleration time_dependent_acceleration(Vecd::Zero());
    SimpleDynamics<GravityForce> apply_initial_force(water_block, time_dependent_acceleration);

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
    BodyAlignedCylinderByCell inlet_disposer_cylinder(water_block,
        makeShared<AlignedCylinderShape>(xAxis, SimTK::UnitVec3(-0.2987, -0.1312, -0.9445), inlet_half[1], inlet_half[0], simTK_resolution, inlet_buffer_translation));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletionArb<AlignedCylinderShape>> inlet_disposer_deletion(inlet_disposer_cylinder);
    BodyAlignedBoxByCell outlet_main_disposer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_disposer_rotation_main), Vec3d(outlet_buffer_translation_main)), outlet_half_main));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> main_disposer_injection(outlet_main_disposer);
    BodyAlignedBoxByCell outlet_left_01_disposer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_disposer_rotation_left_01), Vec3d(outlet_buffer_translation_left_01)), outlet_half_left_01));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> left_01_disposer_injection(outlet_left_01_disposer);
    BodyAlignedBoxByCell outlet_left_02_disposer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_disposer_rotation_left_02), Vec3d(outlet_buffer_translation_left_02)), outlet_half_left_02));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> left_02_disposer_injection(outlet_left_02_disposer);
    BodyAlignedBoxByCell outlet_left_03_disposer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_disposer_rotation_left_03), Vec3d(outlet_buffer_translation_left_03)), outlet_half_left_03));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> left_03_disposer_injection(outlet_left_03_disposer);
    BodyAlignedBoxByCell outlet_rightF_01_disposer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_disposer_rotation_rightF_01), Vec3d(outlet_buffer_translation_rightF_01)), outlet_half_rightF_01));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> rightF_01_disposer_injection(outlet_rightF_01_disposer);
    BodyAlignedBoxByCell outlet_rightF_02_disposer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_disposer_rotation_rightF_02), Vec3d(outlet_buffer_translation_rightF_02)), outlet_half_rightF_02));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> rightF_02_disposer_injection(outlet_rightF_02_disposer);
    BodyAlignedBoxByCell outlet_rightB_01_disposer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_disposer_rotation_rightB_01), Vec3d(outlet_buffer_translation_rightB_01)), outlet_half_rightB_01));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> rightB_01_disposer_injection(outlet_rightB_01_disposer);
    BodyAlignedBoxByCell outlet_rightB_02_disposer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_disposer_rotation_rightB_02), Vec3d(outlet_buffer_translation_rightB_02)), outlet_half_rightB_02));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> rightB_02_disposer_injection(outlet_rightB_02_disposer);
    BodyAlignedBoxByCell outlet_rightB_03_disposer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_disposer_rotation_rightB_03), Vec3d(outlet_buffer_translation_rightB_03)), outlet_half_rightB_03));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> rightB_03_disposer_injection(outlet_rightB_03_disposer);
    BodyAlignedBoxByCell outlet_rightB_04_disposer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_disposer_rotation_rightB_04), Vec3d(outlet_buffer_translation_rightB_04)), outlet_half_rightB_04));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> rightB_04_disposer_injection(outlet_rightB_04_disposer);

    // emitter
    BodyAlignedCylinderByCell inlet_emitter_cylinder(water_block,
        makeShared<AlignedCylinderShape>(xAxis, SimTK::UnitVec3(0.2987, 0.1312, 0.9445), inlet_half[1], inlet_half[0], simTK_resolution, inlet_buffer_translation));
    fluid_dynamics::NonPrescribedPressureBidirectionalBufferArb<AlignedCylinderShape> inlet_emitter_injection(inlet_emitter_cylinder, in_outlet_particle_buffer);
    BodyAlignedBoxByCell outlet_main_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_main), Vec3d(outlet_buffer_translation_main)), outlet_half_main));
    fluid_dynamics::BidirectionalBuffer<OutletInflowPressure> main_emitter_injection(outlet_main_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell outlet_left_01_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_left_01), Vec3d(outlet_buffer_translation_left_01)), outlet_half_left_01));
    fluid_dynamics::BidirectionalBuffer<OutletInflowPressure> left_01_emitter_injection(outlet_left_01_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell outlet_left_02_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_left_02), Vec3d(outlet_buffer_translation_left_02)), outlet_half_left_02));
    fluid_dynamics::BidirectionalBuffer<OutletInflowPressure> left_02_emitter_injection(outlet_left_02_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell outlet_left_03_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_left_03), Vec3d(outlet_buffer_translation_left_03)), outlet_half_left_03));
    fluid_dynamics::BidirectionalBuffer<OutletInflowPressure> left_03_emitter_injection(outlet_left_03_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell outlet_rightF_01_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_rightF_01), Vec3d(outlet_buffer_translation_rightF_01)), outlet_half_rightF_01));
    fluid_dynamics::BidirectionalBuffer<OutletInflowPressure> rightF_01_emitter_injection(outlet_rightF_01_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell outlet_rightF_02_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_rightF_02), Vec3d(outlet_buffer_translation_rightF_02)), outlet_half_rightF_02));
    fluid_dynamics::BidirectionalBuffer<OutletInflowPressure> rightF_02_emitter_injection(outlet_rightF_02_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell outlet_rightB_01_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_rightB_01), Vec3d(outlet_buffer_translation_rightB_01)), outlet_half_rightB_01));
    fluid_dynamics::BidirectionalBuffer<OutletInflowPressure> rightB_01_emitter_injection(outlet_rightB_01_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell outlet_rightB_02_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_rightB_02), Vec3d(outlet_buffer_translation_rightB_02)), outlet_half_rightB_02));
    fluid_dynamics::BidirectionalBuffer<OutletInflowPressure> rightB_02_emitter_injection(outlet_rightB_02_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell outlet_rightB_03_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_rightB_03), Vec3d(outlet_buffer_translation_rightB_03)), outlet_half_rightB_03));
    fluid_dynamics::BidirectionalBuffer<OutletInflowPressure> rightB_03_emitter_injection(outlet_rightB_03_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell outlet_rightB_04_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_rightB_04), Vec3d(outlet_buffer_translation_rightB_04)), outlet_half_rightB_04));
    fluid_dynamics::BidirectionalBuffer<OutletInflowPressure> rightB_04_emitter_injection(outlet_rightB_04_emitter, in_outlet_particle_buffer);

    InteractionWithUpdate<fluid_dynamics::DensitySummationPressureComplex> update_fluid_density(water_block_inner, water_shell_contact);
    SimpleDynamics<fluid_dynamics::PressureConditionCylinder<InletInflowPressure>> inlet_pressure_condition(inlet_emitter_cylinder);
    SimpleDynamics<fluid_dynamics::PressureCondition<OutletInflowPressure>> main_pressure_condition(outlet_main_emitter);
    SimpleDynamics<fluid_dynamics::PressureCondition<OutletInflowPressure>> left_01_pressure_condition(outlet_left_01_emitter);
    SimpleDynamics<fluid_dynamics::PressureCondition<OutletInflowPressure>> left_02_pressure_condition(outlet_left_02_emitter);
    SimpleDynamics<fluid_dynamics::PressureCondition<OutletInflowPressure>> left_03_pressure_condition(outlet_left_03_emitter);
    SimpleDynamics<fluid_dynamics::PressureCondition<OutletInflowPressure>> rightF_01_pressure_condition(outlet_rightF_01_emitter);
    SimpleDynamics<fluid_dynamics::PressureCondition<OutletInflowPressure>> rightF_02_pressure_condition(outlet_rightF_02_emitter);
    SimpleDynamics<fluid_dynamics::PressureCondition<OutletInflowPressure>> rightB_01_pressure_condition(outlet_rightB_01_emitter);
    SimpleDynamics<fluid_dynamics::PressureCondition<OutletInflowPressure>> rightB_02_pressure_condition(outlet_rightB_02_emitter);
    SimpleDynamics<fluid_dynamics::PressureCondition<OutletInflowPressure>> rightB_03_pressure_condition(outlet_rightB_03_emitter);
    SimpleDynamics<fluid_dynamics::PressureCondition<OutletInflowPressure>> rightB_04_pressure_condition(outlet_rightB_04_emitter);
    SimpleDynamics<fluid_dynamics::InflowVelocityConditionCylinder<InflowVelocity>> inflow_velocity_condition(inlet_emitter_cylinder);

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
    inlet_emitter_injection.tag_buffer_particles.exec();
    main_emitter_injection.tag_buffer_particles.exec();
    left_01_emitter_injection.tag_buffer_particles.exec();
    left_02_emitter_injection.tag_buffer_particles.exec();
    left_03_emitter_injection.tag_buffer_particles.exec();
    rightF_01_emitter_injection.tag_buffer_particles.exec();
    rightF_02_emitter_injection.tag_buffer_particles.exec();
    rightB_01_emitter_injection.tag_buffer_particles.exec();
    rightB_02_emitter_injection.tag_buffer_particles.exec();
    rightB_03_emitter_injection.tag_buffer_particles.exec();
    rightB_04_emitter_injection.tag_buffer_particles.exec();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    size_t number_of_iterations = sph_system.RestartStep();
    int screen_output_interval = 100;
    int observation_sample_interval = screen_output_interval * 2;
    Real end_time = 2.5;   /**< End time. */
    Real Output_Time = end_time/250; /**< Time stamps for output of body states. */
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
    //----------------------------------------------------------------------
    //	First output before the main loop.
    //----------------------------------------------------------------------
    body_states_recording.writeToFile();
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
            apply_initial_force.exec();

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

                inlet_pressure_condition.exec(dt);
                main_pressure_condition.exec(dt);
                left_01_pressure_condition.exec(dt);
                left_02_pressure_condition.exec(dt);
                left_03_pressure_condition.exec(dt);
                rightF_01_pressure_condition.exec(dt);
                rightF_02_pressure_condition.exec(dt);
                rightB_01_pressure_condition.exec(dt);
                rightB_02_pressure_condition.exec(dt);
                rightB_03_pressure_condition.exec(dt);
                rightB_04_pressure_condition.exec(dt);

                inflow_velocity_condition.exec();

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
            }
            number_of_iterations++;

            time_instance = TickCount::now();

            inlet_emitter_injection.injection.exec();
            main_emitter_injection.injection.exec();
            left_01_emitter_injection.injection.exec();
            left_02_emitter_injection.injection.exec();
            left_03_emitter_injection.injection.exec();
            rightF_01_emitter_injection.injection.exec();
            rightF_02_emitter_injection.injection.exec();
            rightB_01_emitter_injection.injection.exec();
            rightB_02_emitter_injection.injection.exec();
            rightB_03_emitter_injection.injection.exec();
            rightB_04_emitter_injection.injection.exec();

            inlet_disposer_deletion.exec();
            main_disposer_injection.exec();
            left_01_disposer_injection.exec();
            left_02_disposer_injection.exec();
            left_03_disposer_injection.exec();
            rightF_01_disposer_injection.exec();
            rightF_02_disposer_injection.exec();
            rightB_01_disposer_injection.exec();
            rightB_02_disposer_injection.exec();
            rightB_03_disposer_injection.exec();
            rightB_04_disposer_injection.exec();

            water_block.updateCellLinkedListWithParticleSort(100);
            //shell_update_normal.exec();
            //shell_body.updateCellLinkedList();
            //shell_curvature_inner.updateConfiguration();
            //shell_average_curvature.exec();
            //shell_water_contact.updateConfiguration();
            water_block_complex.updateConfiguration();

            interval_updating_configuration += TickCount::now() - time_instance;
            boundary_indicator.exec();

            inlet_emitter_injection.tag_buffer_particles.exec();
            main_emitter_injection.tag_buffer_particles.exec();
            left_01_emitter_injection.tag_buffer_particles.exec();
            left_02_emitter_injection.tag_buffer_particles.exec();
            left_03_emitter_injection.tag_buffer_particles.exec();
            rightF_01_emitter_injection.tag_buffer_particles.exec();
            rightF_02_emitter_injection.tag_buffer_particles.exec();
            rightB_01_emitter_injection.tag_buffer_particles.exec();
            rightB_02_emitter_injection.tag_buffer_particles.exec();
            rightB_03_emitter_injection.tag_buffer_particles.exec();
            rightB_04_emitter_injection.tag_buffer_particles.exec();
        }
        TickCount t2 = TickCount::now();
        body_states_recording.writeToFile();
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
