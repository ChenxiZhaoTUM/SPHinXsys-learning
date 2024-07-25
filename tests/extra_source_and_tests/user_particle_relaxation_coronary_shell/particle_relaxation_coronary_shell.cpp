/**
 * @file 	particle_relaxation_single_resolution.cpp
 * @brief 	This is the test of using levelset to generate particles with single resolution and relax particles.
 * @details We use this case to test the particle generation and relaxation for a complex geometry.
 *			Before particle generation, we clean the sharp corners of the model.
 * @author 	Yongchuan Yu and Xiangyu Hu
 */

#include "sphinxsys.h"
#include "base64_tobiaslocker.hpp"

using namespace SPH;
//----------------------------------------------------------------------
//	Setting for the geometry.
//	To use this, please commenting the setting for the first geometry.
//----------------------------------------------------------------------
std::string full_path_to_file = "./input/normal_fluid_repair.stl";
std::string full_path_to_vtp = "./input/normal1_mesh.vtp";
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Vec3d translation(0.0, 0.0, 0.0);
//Real scaling = pow(10, -3);
Real scaling = 1.0;
Vec3d domain_lower_bound(-375.0 * scaling, 100.0 * scaling, -340 * scaling);
Vec3d domain_upper_bound(-100.0 * scaling, 360.0 * scaling, 0.0 * scaling);
//----------------------------------------------------------------------
//	Below are common parts for the two test geometries.
//----------------------------------------------------------------------
BoundingBox system_domain_bounds(domain_lower_bound, domain_upper_bound);
//Real dp_0 = (domain_upper_bound[0] - domain_lower_bound[0]) / 200.0;  // 1.375 * pow(10, -3)
Real dp_0 = 0.6 * scaling;
Real thickness = 1.0 * dp_0;
//Real level_set_refinement_ratio = dp_0 / (0.1 * thickness);
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

class FromVTPFile;
template <>
class ParticleGenerator<SurfaceParticles, FromVTPFile> : public ParticleGenerator<SurfaceParticles>
{
    std::string base64_data_;
    size_t points_offset_;
    size_t verts_connectivity_offset_;
    size_t polys_connectivity_offset_;
    size_t polys_offset_;
    size_t number_of_points_;
    size_t number_of_polys_;

    Real total_volume_;
    std::vector<Real> face_areas_;
    Real particle_spacing_;
    const Real thickness_;
    Real avg_particle_volume_;
    size_t planned_number_of_particles_;
    Shape &initial_shape_;

  public:
    explicit ParticleGenerator(SPHBody &sph_body, SurfaceParticles &surface_particles, const std::string &vtp_file_path)
        : ParticleGenerator<SurfaceParticles>(sph_body, surface_particles),
          total_volume_(0),
          particle_spacing_(sph_body.sph_adaptation_->ReferenceSpacing()),
          thickness_(particle_spacing_),
          avg_particle_volume_(pow(particle_spacing_, 3) * thickness_),
          planned_number_of_particles_(0),
          initial_shape_(sph_body.getInitialShape())
    {

        if (!initial_shape_.isValid())
        {
            throw std::invalid_argument("Invalid initial shape provided.");
        }

        loadVTPFile(vtp_file_path);

        face_areas_.resize(number_of_polys_);
    }

    void prepareGeometricData() override
    {
        validateBase64Data();
        auto [decoded_points, decoded_polys_connectivity, decoded_polys] = decodeGeometricData();

        auto points = parsePoints(decoded_points);
        auto faces = parseFaces(decoded_polys_connectivity);

        calculateTotalVolume(points, faces);
        distributeParticles(points, faces);
    }

  private:
    void loadVTPFile(const std::string &filePath)
    {
        std::ifstream vtp_file(filePath);
        if (!vtp_file.is_open())
        {
            std::cerr << "Error: Failed to open the VTP file: " << filePath << std::endl;
            throw std::runtime_error("VTP file opening failed");
        }

        std::string line;

        while (std::getline(vtp_file, line))
        {
            if (line.find("NumberOfPoints=") != std::string::npos)
            {
                size_t pos = line.find("NumberOfPoints=") + 16;
                number_of_points_ = std::stoul(line.substr(pos));
            }
            if (line.find("NumberOfPolys=") != std::string::npos)
            {
                size_t pos = line.find("NumberOfPolys=") + 15;
                number_of_polys_ = std::stoul(line.substr(pos));
            }
            if (line.find("<Points>") != std::string::npos)
            {
                while (std::getline(vtp_file, line) && line.find("</Points>") == std::string::npos)
                {
                    if (line.find("offset=") != std::string::npos)
                    {
                        size_t pos = line.find("offset=") + 8;
                        points_offset_ = std::stoul(line.substr(pos));
                    }
                }
            }
            if (line.find("<Verts>") != std::string::npos)
            {
                while (std::getline(vtp_file, line) && line.find("</Verts>") == std::string::npos)
                {
                    if (line.find("connectivity") != std::string::npos && line.find("offset=") != std::string::npos)
                    {
                        size_t pos = line.find("offset=") + 8;
                        verts_connectivity_offset_ = std::stoul(line.substr(pos));
                    }
                }
            }
            if (line.find("<Polys>") != std::string::npos)
            {
                while (std::getline(vtp_file, line) && line.find("</Polys>") == std::string::npos)
                {
                    if (line.find("connectivity") != std::string::npos && line.find("offset=") != std::string::npos)
                    {
                        size_t pos = line.find("offset=") + 8;
                        polys_connectivity_offset_ = std::stoul(line.substr(pos));
                    }
                    else if (line.find("offsets") != std::string::npos && line.find("offset=") != std::string::npos)
                    {
                        size_t pos = line.find("offset=") + 8;
                        polys_offset_ = std::stoul(line.substr(pos));
                    }
                }
            }
            if (line.find("<AppendedData") != std::string::npos)
            {
                std::getline(vtp_file, line);  // Skip the first line of appended data
                base64_data_ = line.substr(4); // Remove the leading underscore
                break;
            }
        }

        vtp_file.close();
    }

    void validateBase64Data() const
    {
        if (base64_data_.empty())
        {
            throw std::runtime_error("Base64 data is empty");
        }

        std::cout << "Base64 data length: " << base64_data_.size() << std::endl;
        // std::cout << base64_data_ << std::endl;
        std::cout << "points_offset: " << points_offset_ << std::endl;
        std::cout << "verts_connectivity_offset: " << verts_connectivity_offset_ << std::endl;
        std::cout << "polys_connectivity_offset: " << polys_connectivity_offset_ << std::endl;
        std::cout << "polys_offset: " << polys_offset_ << std::endl;
    }

    std::tuple<std::string, std::string, std::string> decodeGeometricData()
    {
        std::string points_data = base64_data_.substr(points_offset_, verts_connectivity_offset_ - points_offset_);
        std::string polys_connectivity_data = base64_data_.substr(polys_connectivity_offset_, polys_offset_ - polys_connectivity_offset_);
        std::string polys_data = base64_data_.substr(polys_offset_);

        auto decoded_points = base64::from_base64(points_data);
        auto decoded_polys_connectivity = base64::from_base64(polys_connectivity_data);
        auto decoded_polys = base64::from_base64(polys_data);

        std::cout << "Decoded points length: " << decoded_points.size() << std::endl;
        std::cout << "Decoded polys_connectivity length: " << decoded_polys_connectivity.size() << std::endl;
        std::cout << "Decoded polys length: " << decoded_polys.size() << std::endl;

        return std::make_tuple(decoded_points, decoded_polys_connectivity, decoded_polys);
    }

    std::vector<std::array<float, 3>> parsePoints(const std::string &decoded_points)
    {
        std::vector<std::array<float, 3>> points;
        points.reserve(number_of_points_);
        size_t offset = 0;

        for (size_t i = 0; i < number_of_points_; ++i)
        {
            float x, y, z;
            std::memcpy(&x, &decoded_points[offset], sizeof(float));
            offset += sizeof(float);
            std::memcpy(&y, &decoded_points[offset], sizeof(float));
            offset += sizeof(float);
            std::memcpy(&z, &decoded_points[offset], sizeof(float));
            offset += sizeof(float);
            points.emplace_back(std::array<float, 3>{x, y, z});

            // std::cout << "x = " << x << ", y = " << y << ", z = " << z << std::endl;
        }

        std::cout << "offset for points at the end: " << offset << std::endl;
        return points;
    }

    std::vector<std::array<int, 3>> parseFaces(const std::string &decoded_polys_connectivity)
    {
        std::vector<std::array<int, 3>> faces;
        faces.reserve(number_of_polys_);
        size_t offset = 0;

        for (size_t i = 0; i < number_of_polys_; ++i)
        {
            int v1, v2, v3;
            std::memcpy(&v1, &decoded_polys_connectivity[offset], sizeof(int));
            offset += sizeof(int);
            std::memcpy(&v2, &decoded_polys_connectivity[offset], sizeof(int));
            offset += sizeof(int);
            std::memcpy(&v3, &decoded_polys_connectivity[offset], sizeof(int));
            offset += sizeof(int);
            faces.emplace_back(std::array<int, 3>{v1, v2, v3});
        }

        std::cout << "offset for faces at the end: " << offset << std::endl;
        return faces;
    }

    void calculateTotalVolume(const std::vector<std::array<float, 3>> &points, const std::vector<std::array<int, 3>> &faces)
    {
#pragma omp parallel for reduction(+ \
                                   : total_volume_)

        for (size_t i = 0; i < faces.size(); ++i)
        {
            const auto &face = faces[i];
            Vec3d vertices[3] = {
                Vec3d(points[face[0]][0], points[face[0]][1], points[face[0]][2]),
                Vec3d(points[face[1]][0], points[face[1]][1], points[face[1]][2]),
                Vec3d(points[face[2]][0], points[face[2]][1], points[face[2]][2])};
            Real area = calculateEachFaceArea(vertices);
            face_areas_[i] = area;
            total_volume_ += area;
        }
    }

    Real calculateEachFaceArea(const Vec3d vertices[3])
    {
        Vec3d edge1 = vertices[1] - vertices[0];
        Vec3d edge2 = vertices[2] - vertices[0];
        return 0.5 * edge1.cross(edge2).norm();
    }

    void distributeParticles(const std::vector<std::array<float, 3>> &points, const std::vector<std::array<int, 3>> &faces)
    {
        std::mt19937_64 rng(std::random_device{}());
        std::uniform_real_distribution<Real> unif(0.0, 1.0);

        planned_number_of_particles_ = static_cast<size_t>(total_volume_ / avg_particle_volume_ + 0.5);
        std::cout << "planned_number_of_particles calculation = " << planned_number_of_particles_ << std::endl;

        Real interval = static_cast<Real>(planned_number_of_particles_) / faces.size();

        for (size_t i = 0; i < faces.size(); ++i)
        {
            const auto &face = faces[i];
            Vec3d vertices[3] = {
                Vec3d(points[face[0]][0], points[face[0]][1], points[face[0]][2]),
                Vec3d(points[face[1]][0], points[face[1]][1], points[face[1]][2]),
                Vec3d(points[face[2]][0], points[face[2]][1], points[face[2]][2])};

            Real random_value = unif(rng);
            Real interval = planned_number_of_particles_ * (face_areas_[i] / total_volume_);
            if (random_value <= interval && base_particles_.TotalRealParticles() < planned_number_of_particles_)
            {
                int particles_per_face = std::max(1, static_cast<int>(interval));
                generateParticlesOnFace(vertices, particles_per_face);
            }
        }
    }

    void generateParticleAtFaceCenter(const Vec3d vertices[3])
    {
        Vec3d face_center = (vertices[0] + vertices[1] + vertices[2]) / 3.0;

        addPositionAndVolumetricMeasure(face_center, avg_particle_volume_ / thickness_);
        addSurfaceProperties(initial_shape_.findNormalDirection(face_center), thickness_);
    }

    void generateParticlesOnFace(const Vec3d vertices[3], int particles_per_face)
    {
        for (int k = 0; k < particles_per_face; ++k)
        {
            Real u = static_cast<Real>(rand()) / static_cast<Real>(RAND_MAX);
            Real v = static_cast<Real>(rand()) / static_cast<Real>(RAND_MAX);

            if (u + v > 1.0)
            {
                u = 1.0 - u;
                v = 1.0 - v;
            }
            Vec3d particle_position = (1 - u - v) * vertices[0] + u * vertices[1] + v * vertices[2];

            addPositionAndVolumetricMeasure(particle_position, avg_particle_volume_ / thickness_);
            addSurfaceProperties(initial_shape_.findNormalDirection(particle_position), thickness_);
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

// inlet: R=41.7567, (-203.6015, 204.1509, -135.3577), (0.2987, 0.1312, 0.9445)
Vec3d inlet_half = Vec3d(1.5 * dp_0, 42.36 * scaling, 42.36 * scaling);
Vec3d inlet_normal(-0.2987, -0.1312, -0.9445);
Vec3d inlet_translation = Vec3d(-203.6015, 204.1509, -135.3577) * scaling + inlet_normal * 1.0 * dp_0;
Vec3d inlet_standard_direction(1, 0, 0);
RotationResult inlet_rotation_result = RotationCalculator(inlet_normal, inlet_standard_direction);
Rotation3d inlet_rotation(inlet_rotation_result.angle, inlet_rotation_result.axis);

// outlet main: R=36.1590, (-172.2628, 205.9036, -19.8868), (0.2678, 0.3191, -0.9084)
// intersection occurs!!!
Vec3d outlet_half_main = Vec3d(1.5 * dp_0, 45.0 * scaling, 45.0 * scaling);
Vec3d outlet_normal_main(-0.2678, -0.3191, 0.9084);
Vec3d outlet_translation_main = Vec3d(-172.2628, 205.9036, -19.8868) * scaling + outlet_normal_main * 1.0 * dp_0;
Vec3d outlet_standard_direction_main(1, 0, 0);
RotationResult outlet_rotation_result_main = RotationCalculator(outlet_normal_main, outlet_standard_direction_main);
Rotation3d outlet_rotation_main(outlet_rotation_result_main.angle, outlet_rotation_result_main.axis);

// outlet x_pos 01: R=2.6964, (-207.4362, 136.7848, -252.6892), (0.636, 0.771, -0.022)
Vec3d outlet_half_left_01 = Vec3d(1.5 * dp_0, 9.0 * scaling, 9.0 * scaling);
Vec3d outlet_normal_left_01(-0.636, -0.771, 0.022);
Vec3d outlet_translation_left_01 = Vec3d(-207.4362, 136.7848, -252.6892) * scaling + outlet_normal_left_01 * 1.0 * dp_0;
Vec3d outlet_standard_direction_left_01(1, 0, 0);
RotationResult outlet_rotation_result_left_01 = RotationCalculator(outlet_normal_left_01, outlet_standard_direction_left_01);
Rotation3d outlet_rotation_left_01(outlet_rotation_result_left_01.angle, outlet_rotation_result_left_01.axis);

// outlet x_pos 02: R=2.8306, (-193.2735, 337.4625, -270.2884), (-0.6714, 0.3331, -0.6620)
Vec3d outlet_half_left_02 = Vec3d(1.5 * dp_0, 10.0 * scaling, 10.0 * scaling);
Vec3d outlet_normal_left_02(-0.6714, 0.3331, -0.6620);
Vec3d outlet_translation_left_02 = Vec3d(-193.2735, 337.4625, -270.2884) * scaling + outlet_normal_left_02 * 1.0 * dp_0;
Vec3d outlet_standard_direction_left_02(1, 0, 0);
RotationResult outlet_rotation_result_left_02 = RotationCalculator(outlet_normal_left_02, outlet_standard_direction_left_02);
Rotation3d outlet_rotation_left_02(outlet_rotation_result_left_02.angle, outlet_rotation_result_left_02.axis);

// outlet x_pos 03: R=2.2804, (-165.5566, 326.1601, -139.9323), (0.6563, -0.6250, 0.4226)
Vec3d outlet_half_left_03 = Vec3d(1.5 * dp_0, 9.0 * scaling, 9.0 * scaling);
Vec3d outlet_normal_left_03(-0.6563, 0.6250, -0.4226);
Vec3d outlet_translation_left_03 = Vec3d(-165.5566, 326.1601, -139.9323) * scaling + outlet_normal_left_03 * 1.0 * dp_0;
Vec3d outlet_standard_direction_left_03(1, 0, 0);
RotationResult outlet_rotation_result_left_03 = RotationCalculator(outlet_normal_left_03, outlet_standard_direction_left_03);
Rotation3d outlet_rotation_left_03(outlet_rotation_result_left_03.angle, outlet_rotation_result_left_03.axis);

// outlet x_neg_front 01: R=2.6437, (-307.8, 312.1402, -333.2), (-0.185, -0.967, -0.176)
Vec3d outlet_half_rightF_01 = Vec3d(1.5 * dp_0, 10.0 * scaling, 10.0 * scaling);
Vec3d outlet_normal_rightF_01(-0.185, -0.967, -0.176);
Vec3d outlet_translation_rightF_01 = Vec3d(-307.8, 312.1402, -333.2) * scaling + outlet_normal_rightF_01 * 1.0 * dp_0;
Vec3d outlet_standard_direction_rightF_01(1, 0, 0);
RotationResult outlet_rotation_result_rightF_01 = RotationCalculator(outlet_normal_rightF_01, outlet_standard_direction_rightF_01);
Rotation3d outlet_rotation_rightF_01(outlet_rotation_result_rightF_01.angle, outlet_rotation_result_rightF_01.axis);

// outlet x_neg_front 02: R=1.5424, (-369.1252, 235.2617, -193.7022), (-0.501, 0.059, -0.863)
Vec3d outlet_half_rightF_02 = Vec3d(1.5 * dp_0, 8.0 * scaling, 8.0 * scaling);
Vec3d outlet_normal_rightF_02(-0.501, 0.059, -0.863);
Vec3d outlet_translation_rightF_02 = Vec3d(-369.1252, 235.2617, -193.7022) * scaling + outlet_normal_rightF_02 * 1.0 * dp_0;
Vec3d outlet_standard_direction_rightF_02(1, 0, 0);
RotationResult outlet_rotation_result_rightF_02 = RotationCalculator(outlet_normal_rightF_02, outlet_standard_direction_rightF_02);
Rotation3d outlet_rotation_rightF_02(outlet_rotation_result_rightF_02.angle, outlet_rotation_result_rightF_02.axis);

// outlet x_neg_behind 01: R=1.5743, (-268.3522, 116.0357, -182.4896), (0.325, -0.086, -0.942)
Vec3d outlet_half_rightB_01 = Vec3d(1.5 * dp_0, 8.0 * scaling, 8.0 * scaling);
Vec3d outlet_normal_rightB_01(0.325, -0.086, -0.942);
Vec3d outlet_translation_rightB_01 = Vec3d(-268.3522, 116.0357, -182.4896) * scaling + outlet_normal_rightB_01 * 1.0 * dp_0;
Vec3d outlet_standard_direction_rightB_01(1, 0, 0);
RotationResult outlet_rotation_result_rightB_01 = RotationCalculator(outlet_normal_rightB_01, outlet_standard_direction_rightB_01);
Rotation3d outlet_rotation_rightB_01(outlet_rotation_result_rightB_01.angle, outlet_rotation_result_rightB_01.axis);

// outlet x_neg_behind 02: R=1.8204, (-329.0846, 180.5258, -274.3232), (-0.1095, 0.9194, -0.3777)
Vec3d outlet_half_rightB_02 = Vec3d(1.5 * dp_0, 9.0 * scaling, 9.0 * scaling);
Vec3d outlet_normal_rightB_02(-0.1095, 0.9194, -0.3777);
Vec3d outlet_translation_rightB_02 = Vec3d(-329.0846, 180.5258, -274.3232) * scaling + outlet_normal_rightB_02 * 1.0 * dp_0;
Vec3d outlet_standard_direction_rightB_02(1, 0, 0);
RotationResult outlet_rotation_result_rightB_02 = RotationCalculator(outlet_normal_rightB_02, outlet_standard_direction_rightB_02);
Rotation3d outlet_rotation_rightB_02(outlet_rotation_result_rightB_02.angle, outlet_rotation_result_rightB_02.axis);

// outlet x_neg_behind 03: R=1.5491, (-342.1711, 197.1107, -277.8681), (0.1992, 0.5114, -0.8361)
Vec3d outlet_half_rightB_03 = Vec3d(1.5 * dp_0, 8.0 * scaling, 8.0 * scaling);
Vec3d outlet_normal_rightB_03(0.1992, 0.5114, -0.8361);
Vec3d outlet_translation_rightB_03 = Vec3d(-342.1711, 197.1107, -277.8681) * scaling + outlet_normal_rightB_03 * 1.0 * dp_0;
Vec3d outlet_standard_direction_rightB_03(1, 0, 0);
RotationResult outlet_rotation_result_rightB_03 = RotationCalculator(outlet_normal_rightB_03, outlet_standard_direction_rightB_03);
Rotation3d outlet_rotation_rightB_03(outlet_rotation_result_rightB_03.angle, outlet_rotation_result_rightB_03.axis);

// outlet x_neg_behind 04: R=2.1598, (-362.0112, 200.5693, -253.8417), (0.3694, 0.6067, -0.7044)
Vec3d outlet_half_rightB_04 = Vec3d(1.5 * dp_0, 9.0 * scaling, 9.0 * scaling);
Vec3d outlet_normal_rightB_04(0.3694, 0.6067, -0.7044);
Vec3d outlet_translation_rightB_04 = Vec3d(-362.0112, 200.5693, -253.8417) * scaling + outlet_normal_rightB_04 * 1.0 * dp_0;
Vec3d outlet_standard_direction_rightB_04(1, 0, 0);
RotationResult outlet_rotation_result_rightB_04 = RotationCalculator(outlet_normal_rightB_04, outlet_standard_direction_rightB_04);
Rotation3d outlet_rotation_rightB_04(outlet_rotation_result_rightB_04.angle, outlet_rotation_result_rightB_04.axis);

//-----------------------------------------------------------------------------------------------------------
//	Main program starts here.
//-----------------------------------------------------------------------------------------------------------
int main(int ac, char *av[])
{
    //----------------------------------------------------------------------
    //	Build up -- a SPHSystem
    //----------------------------------------------------------------------
    SPHSystem sph_system(system_domain_bounds, dp_0);
    sph_system.setRunParticleRelaxation(false);
    sph_system.setReloadParticles(false);
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment();
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.
    //----------------------------------------------------------------------
    //SolidBodyFromMesh solid_body_from_mesh("SolidBodyFromMesh");
    //TriangleMeshShapeSTL* mesh_shape = solid_body_from_mesh.getMeshShape();

    RealBody imported_model(sph_system, makeShared<SolidBodyFromMesh>("ShellShape"));
    imported_model.defineAdaptation<SPHAdaptation>(1.15, 1.0);
    //imported_model.defineBodyLevelSetShape()->correctLevelSetSign()->writeLevelSet(sph_system);
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? imported_model.generateParticles<SurfaceParticles, Reload>(imported_model.getName())
        : imported_model.generateParticles<SurfaceParticles, FromVTPFile>(full_path_to_vtp);
    
    /*RealBody test_body_in(
        sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(inlet_rotation), Vec3d(inlet_translation)), inlet_half, "TestBodyIn"));
    test_body_in.generateParticles<BaseParticles, Lattice>();*/
    BodyAlignedBoxByCell inlet_detection_box(imported_model, 
        makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(inlet_rotation), Vec3d(inlet_translation)), inlet_half));

    /*RealBody test_body_out_main(
        sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_rotation_main), Vec3d(outlet_translation_main)), outlet_half_main, "TestBodyOutMain"));
    test_body_out_main.generateParticles<BaseParticles, Lattice>();*/
    BodyAlignedBoxByCell outlet_main_detection_box(imported_model, 
        makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_rotation_main), Vec3d(outlet_translation_main)), outlet_half_main));

    /*RealBody test_body_out_left01(
        sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_rotation_left_01), Vec3d(outlet_translation_left_01)), outlet_half_left_01, "TestBodyOutLeft01"));
    test_body_out_left01.generateParticles<BaseParticles, Lattice>();*/
    BodyAlignedBoxByCell outlet_left01_detection_box(imported_model, 
        makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_rotation_left_01), Vec3d(outlet_translation_left_01)), outlet_half_left_01));

    /*RealBody test_body_out_left02(
        sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_rotation_left_02), Vec3d(outlet_translation_left_02)), outlet_half_left_02, "TestBodyOutLeft02"));
    test_body_out_left02.generateParticles<BaseParticles, Lattice>();*/
    BodyAlignedBoxByCell outlet_left02_detection_box(imported_model, 
        makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_rotation_left_02), Vec3d(outlet_translation_left_02)), outlet_half_left_02));

    /*RealBody test_body_out_left03(
        sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_rotation_left_03), Vec3d(outlet_translation_left_03)), outlet_half_left_03, "TestBodyOutLeft03"));
    test_body_out_left03.generateParticles<BaseParticles, Lattice>();*/
    BodyAlignedBoxByCell outlet_left03_detection_box(imported_model, 
        makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_rotation_left_03), Vec3d(outlet_translation_left_03)), outlet_half_left_03));

    /*RealBody test_body_out_rightF01(
        sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_rotation_rightF_01), Vec3d(outlet_translation_rightF_01)), outlet_half_rightF_01, "TestBodyOutRightF01"));
    test_body_out_rightF01.generateParticles<BaseParticles, Lattice>();*/
    BodyAlignedBoxByCell outlet_rightF01_detection_box(imported_model, 
        makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_rotation_rightF_01), Vec3d(outlet_translation_rightF_01)), outlet_half_rightF_01));

    /*RealBody test_body_out_rightF02(
        sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_rotation_rightF_02), Vec3d(outlet_translation_rightF_02)), outlet_half_rightF_02, "TestBodyOutRightF02"));
    test_body_out_rightF02.generateParticles<BaseParticles, Lattice>();*/
    BodyAlignedBoxByCell outlet_rightF02_detection_box(imported_model, 
        makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_rotation_rightF_02), Vec3d(outlet_translation_rightF_02)), outlet_half_rightF_02));
    
    /*RealBody test_body_out_rightB01(
        sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_rotation_rightB_01), Vec3d(outlet_translation_rightB_01)), outlet_half_rightB_01, "TestBodyOutRightB01"));
    test_body_out_rightB01.generateParticles<BaseParticles, Lattice>();*/
    BodyAlignedBoxByCell outlet_rightB01_detection_box(imported_model, 
        makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_rotation_rightB_01), Vec3d(outlet_translation_rightB_01)), outlet_half_rightB_01));

    /*RealBody test_body_out_rightB02(
        sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_rotation_rightB_02), Vec3d(outlet_translation_rightB_02)), outlet_half_rightB_02, "TestBodyOutRightB02"));
    test_body_out_rightB02.generateParticles<BaseParticles, Lattice>();*/
    BodyAlignedBoxByCell outlet_rightB02_detection_box(imported_model, 
        makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_rotation_rightB_02), Vec3d(outlet_translation_rightB_02)), outlet_half_rightB_02));

    /*RealBody test_body_out_rightB03(
        sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_rotation_rightB_03), Vec3d(outlet_translation_rightB_03)), outlet_half_rightB_03, "TestBodyOutRightB03"));
    test_body_out_rightB03.generateParticles<BaseParticles, Lattice>();*/
    BodyAlignedBoxByCell outlet_rightB03_detection_box(imported_model, 
        makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_rotation_rightB_03), Vec3d(outlet_translation_rightB_03)), outlet_half_rightB_03));

    /*RealBody test_body_out_rightB04(
        sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_rotation_rightB_04), Vec3d(outlet_translation_rightB_04)), outlet_half_rightB_04, "TestBodyOutRightB04"));
    test_body_out_rightB04.generateParticles<BaseParticles, Lattice>();*/
    BodyAlignedBoxByCell outlet_rightB04_detection_box(imported_model, 
        makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_rotation_rightB_04), Vec3d(outlet_translation_rightB_04)), outlet_half_rightB_04));

    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the the range of bodies to build neighbor particle lists.
    //  Generally, we first define all the inner relations, then the contact relations.
    //  At last, we define the complex relaxations by combining previous defined
    //  inner and contact relations.
    //----------------------------------------------------------------------
    if (sph_system.RunParticleRelaxation())
    {
        InnerRelation imported_model_inner(imported_model);
        //----------------------------------------------------------------------
        //	Methods used for particle relaxation.
        //----------------------------------------------------------------------
        using namespace relax_dynamics;
        SimpleDynamics<RandomizeParticlePosition> random_imported_model_particles(imported_model);
        /** A  Physics relaxation step. */
        ShellRelaxationStep relaxation_step_inner(imported_model_inner);
        ShellNormalDirectionPrediction shell_normal_prediction(imported_model_inner, thickness);

        //SimpleDynamics<AlignedBoxParticlesDetection> inlet_particles_detection(inlet_detection_box);
        //SimpleDynamics<AlignedBoxParticlesDetection> outlet_main_particles_detection(outlet_main_detection_box);
        //SimpleDynamics<AlignedBoxParticlesDetection> outlet_left01_particles_detection(outlet_left01_detection_box);
        //SimpleDynamics<AlignedBoxParticlesDetection> outlet_left02_particles_detection(outlet_left02_detection_box);
        //SimpleDynamics<AlignedBoxParticlesDetection> outlet_left03_particles_detection(outlet_left03_detection_box);
        //SimpleDynamics<AlignedBoxParticlesDetection> outlet_rightF01_particles_detection(outlet_rightF01_detection_box);
        //SimpleDynamics<AlignedBoxParticlesDetection> outlet_rightF02_particles_detection(outlet_rightF02_detection_box);
        //SimpleDynamics<AlignedBoxParticlesDetection> outlet_rightB01_particles_detection(outlet_rightB01_detection_box);
        //SimpleDynamics<AlignedBoxParticlesDetection> outlet_rightB02_particles_detection(outlet_rightB02_detection_box);
        //SimpleDynamics<AlignedBoxParticlesDetection> outlet_rightB03_particles_detection(outlet_rightB03_detection_box);
        //SimpleDynamics<AlignedBoxParticlesDetection> outlet_rightB04_particles_detection(outlet_rightB04_detection_box);

        /** Write the body state to Vtp file. */
        BodyStatesRecordingToVtp write_imported_model_to_vtp({imported_model});
        /** Write the particle reload files. */
        ReloadParticleIO write_particle_reload_files(imported_model);
        //----------------------------------------------------------------------
        //	Particle relaxation starts here.
        //----------------------------------------------------------------------
        random_imported_model_particles.exec(0.25);
        relaxation_step_inner.MidSurfaceBounding().exec();
        write_imported_model_to_vtp.writeToFile(0.0);
        imported_model.updateCellLinkedList();
        // write_cell_linked_list.writeToFile(0.0);
        //----------------------------------------------------------------------
        //	Particle relaxation time stepping start here.
        //----------------------------------------------------------------------
        int ite_p = 0;
        while (ite_p < 500)
        {
            relaxation_step_inner.exec();
            ite_p += 1;
            if (ite_p % 50 == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "Relaxation steps for the imported model N = " << ite_p << "\n";
                write_imported_model_to_vtp.writeToFile(ite_p);
            }
        }
        std::cout << "The physics relaxation process of imported model finish !" << std::endl;

        //inlet_particles_detection.exec();
        //outlet_main_particles_detection.exec();
        //outlet_left01_particles_detection.exec();
        //outlet_left02_particles_detection.exec();
        //outlet_left03_particles_detection.exec();
        //outlet_rightF01_particles_detection.exec();
        //outlet_rightF02_particles_detection.exec();
        //outlet_rightB01_particles_detection.exec();
        //outlet_rightB02_particles_detection.exec();
        //outlet_rightB03_particles_detection.exec();
        //outlet_rightB04_particles_detection.exec();

        shell_normal_prediction.exec();
        write_imported_model_to_vtp.writeToFile(ite_p);
        write_particle_reload_files.writeToFile(0);
        return 0;
    }

    /*imported_model.updateCellLinkedList();

    SimpleDynamics<relax_dynamics::ParticlesInAlignedBoxDetectionByCell> inlet_particles_detection(inlet_detection_box);
    SimpleDynamics<relax_dynamics::ParticlesInAlignedBoxDetectionByCell> outlet_main_particles_detection(outlet_main_detection_box);
    SimpleDynamics<relax_dynamics::ParticlesInAlignedBoxDetectionByCell> outlet_left01_particles_detection(outlet_left01_detection_box);
    SimpleDynamics<relax_dynamics::ParticlesInAlignedBoxDetectionByCell> outlet_left02_particles_detection(outlet_left02_detection_box);
    SimpleDynamics<relax_dynamics::ParticlesInAlignedBoxDetectionByCell> outlet_left03_particles_detection(outlet_left03_detection_box);
    SimpleDynamics<relax_dynamics::ParticlesInAlignedBoxDetectionByCell> outlet_rightF01_particles_detection(outlet_rightF01_detection_box);
    SimpleDynamics<relax_dynamics::ParticlesInAlignedBoxDetectionByCell> outlet_rightF02_particles_detection(outlet_rightF02_detection_box);
    SimpleDynamics<relax_dynamics::ParticlesInAlignedBoxDetectionByCell> outlet_rightB01_particles_detection(outlet_rightB01_detection_box);
    SimpleDynamics<relax_dynamics::ParticlesInAlignedBoxDetectionByCell> outlet_rightB02_particles_detection(outlet_rightB02_detection_box);
    SimpleDynamics<relax_dynamics::ParticlesInAlignedBoxDetectionByCell> outlet_rightB03_particles_detection(outlet_rightB03_detection_box);
    SimpleDynamics<relax_dynamics::ParticlesInAlignedBoxDetectionByCell> outlet_rightB04_particles_detection(outlet_rightB04_detection_box);*/


    BodyStatesRecordingToVtp write_body_states(sph_system);

    /*inlet_particles_detection.exec();
    outlet_main_particles_detection.exec();
    outlet_left01_particles_detection.exec();
    outlet_left02_particles_detection.exec();
    outlet_left03_particles_detection.exec();
    outlet_rightF01_particles_detection.exec();
    outlet_rightF02_particles_detection.exec();
    outlet_rightB01_particles_detection.exec();
    outlet_rightB02_particles_detection.exec();
    outlet_rightB03_particles_detection.exec();
    outlet_rightB04_particles_detection.exec();*/

    write_body_states.writeToFile();
    return 0;
}
