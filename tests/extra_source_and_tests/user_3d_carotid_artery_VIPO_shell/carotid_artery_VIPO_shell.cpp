/**
 * @file 	T_shaped_pipe.cpp
 * @brief 	This is the benchmark test of multi-inlet and multi-outlet.
 * @details We consider a flow with one inlet and two outlets in a T-shaped pipe in 2D.
 * @author 	Xiangyu Hu, Shuoguo Zhang
 */

#include "sphinxsys.h"
#include "bidirectional_buffer.h"
#include "density_correciton.h"
#include "density_correciton.hpp"
#include "kernel_summation.h"
#include "kernel_summation.hpp"
#include "pressure_boundary.h"

using namespace SPH;
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
std::string full_path_to_file = "./input/carotid_fluid_geo.stl";
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Vec3d translation(0.0, 0.0, 0.0);
Real scaling = 1.0;
Vec3d domain_lower_bound(-6.0 * scaling, -4.0 * scaling, -32.5 * scaling);
Vec3d domain_upper_bound(12.0 * scaling, 10.0 * scaling, 23.5 * scaling);
BoundingBox system_domain_bounds(domain_lower_bound, domain_upper_bound);
Real dp_0 = 0.15 * scaling;
Real thickness = 1.0 * dp_0;
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
        add<ExtrudeShape<TriangleMeshShapeSTL>>(-thickness, full_path_to_file, translation, scaling);
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

// inlet R=2.9293, (1.5611, 5.8559, -30.8885), (0.1034, -0.0458, 0.9935)
Real DW_in = 2.9293 * 2 * scaling;
Vec3d inlet_half = Vec3d(2.0 * dp_0, 3.5 * scaling, 3.5 * scaling);
Vec3d inlet_normal(-0.1034, 0.0458, -0.9935);
Vec3d inlet_translation = Vec3d(1.5611, 5.8559, -30.8885) * scaling + inlet_normal * 2.0 * dp_0;
Vec3d inlet_buffer_translation = Vec3d(1.5611, 5.8559, -30.8885) * scaling - inlet_normal * 2.0 * dp_0;
Vec3d inlet_standard_direction(1, 0, 0);
RotationResult inlet_rotation_result = RotationCalculator(inlet_normal, inlet_standard_direction);
Rotation3d inlet_rotation(inlet_rotation_result.angle, inlet_rotation_result.axis);
Rotation3d inlet_emitter_rotation(inlet_rotation_result.angle + Pi, inlet_rotation_result.axis);

// outlet1 R=1.9416, (-2.6975, -0.4330, 21.7855), (-0.3160, -0.0009, 0.9488)
Real DW_out01 = 1.9416 * 2 * scaling;
Vec3d outlet_01_half = Vec3d(2.0 * dp_0, 2.4 * scaling, 2.4 * scaling);
Vec3d outlet_01_normal(-0.3160, -0.0009, 0.9488);
Vec3d outlet_01_translation = Vec3d(-2.6975, -0.4330, 21.7855) * scaling + outlet_01_normal * 2.0 * dp_0;
Vec3d outlet_01_buffer_translation = Vec3d(-2.6975, -0.4330, 21.7855) * scaling - outlet_01_normal * 2.0 * dp_0;
Vec3d outlet_01_standard_direction(1, 0, 0);
RotationResult outlet_01_rotation_result = RotationCalculator(outlet_01_normal, outlet_01_standard_direction);
Rotation3d outlet_01_rotation(outlet_01_rotation_result.angle, outlet_01_rotation_result.axis);
Rotation3d outlet_emitter_01_rotation(outlet_01_rotation_result.angle + Pi, outlet_01_rotation_result.axis);

// outlet2 R=1.3261, (9.0220, 0.9750, 18.6389), (-0.0399, 0.0693, 0.9972)
Real DW_out02 = 1.3261 * 2 * scaling;
Vec3d outlet_02_half = Vec3d(2.0 * dp_0, 2.0 * scaling, 2.0 * scaling);
Vec3d outlet_02_normal(-0.0399, 0.0693, 0.9972);
Vec3d outlet_02_translation = Vec3d(9.0220, 0.9750, 18.6389) * scaling + outlet_02_normal * 2.0 * dp_0;
Vec3d outlet_02_buffer_translation = Vec3d(9.0220, 0.9750, 18.6389) * scaling - outlet_02_normal * 2.0 * dp_0;
Vec3d outlet_02_standard_direction(1, 0, 0);
RotationResult outlet_02_rotation_result = RotationCalculator(outlet_02_normal, outlet_02_standard_direction);
Rotation3d outlet_02_rotation(outlet_02_rotation_result.angle, outlet_02_rotation_result.axis);
Rotation3d outlet_emitter_02_rotation(outlet_02_rotation_result.angle + Pi, outlet_02_rotation_result.axis);
//----------------------------------------------------------------------
//	Global parameters on the fluid properties
//----------------------------------------------------------------------
Real rho0_f = 1060; /**< Reference density of fluid. */
Real U_f = 0.5;    /**< Characteristic velocity. */
/** Reference sound speed needs to consider the flow speed in the narrow channels. */
Real c_f = 10.0 * U_f * SMAX(Real(1), DW_in / (DW_out01 + DW_out02));
Real mu_f = 0.00355; /**< Dynamics viscosity. */
//Real Outlet_pressure = 1.33 * pow(10, 4);
Real Inlet_pressure = 0.2;
Real Outlet_pressure = 0.1;

Real rho0_s = 1120;                /** Normalized density. */
Real Youngs_modulus = 1.08e6;    /** Normalized Youngs Modulus. */
Real poisson = 0.49;               /** Poisson ratio. */
//Real physical_viscosity = 0.25 * sqrt(rho0_s * Youngs_modulus) * 55.0 * scaling; /** physical damping */
//----------------------------------------------------------------------
//	Inflow velocity
//----------------------------------------------------------------------
struct InflowVelocity
{
    Real u_ref_, t_ref_, interval_;
    AlignedBoxShape &aligned_box_;

    template <class BoundaryConditionType>
    InflowVelocity(BoundaryConditionType &boundary_condition)
        : u_ref_(0.1), t_ref_(0.218), interval_(0.5),
        aligned_box_(boundary_condition.getAlignedBox()){}

    Vecd operator()(Vecd &position, Vecd &velocity)
    {
        Vecd target_velocity = velocity;
        Real run_time = GlobalStaticVariables::physical_time_;
        int n = static_cast<int>(run_time / interval_);
        Real t_in_cycle = run_time - n * interval_;

        target_velocity[2] = t_in_cycle < t_ref_ ? 0.5 * sin(4 * Pi * (run_time + 0.0160236)) : u_ref_;
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
struct LeftInflowPressure
{
    template <class BoundaryConditionType>
    LeftInflowPressure(BoundaryConditionType &boundary_condition) {}

    Real operator()(Real &p_)
    {
        return p_;
    }
};

struct RightInflowPressure
{
    template <class BoundaryConditionType>
    RightInflowPressure(BoundaryConditionType &boundary_condition) {}

    Real operator()(Real &p_)
    {
        /*constant pressure*/
        Real pressure = Outlet_pressure;
        return pressure;
    }
};
//----------------------------------------------------------------------
//	Define constrain class for tank translation and rotation.
//----------------------------------------------------------------------
class QuantityMomentOfMomentum : public QuantitySummation<Vecd>
{
protected:
	StdLargeVec<Real>& mass_;
	StdLargeVec<Vecd>& pos_;
	Vecd mass_center_;

public:
	explicit QuantityMomentOfMomentum(SPHBody& sph_body, Vecd mass_center)
		: QuantitySummation<Vecd>(sph_body, "Velocity"),
		mass_center_(mass_center), 
        mass_(*particles_->getVariableDataByName<Real>("Mass")), 
        pos_(*particles_->getVariableDataByName<Vecd>("Position"))
	{
		this->quantity_name_ = "Moment of Momentum";
	};
	virtual ~QuantityMomentOfMomentum() {};

	Vecd reduce(size_t index_i, Real dt = 0.0)
	{
		return (pos_[index_i] - mass_center_).cross(this->variable_[index_i]) * mass_[index_i];
	};
};

class QuantityMomentOfInertia : public QuantitySummation<Real>
{
protected:
	StdLargeVec<Vecd>& pos_;
	Vecd mass_center_;
	Real p_1_;
	Real p_2_;

public:
	explicit QuantityMomentOfInertia(SPHBody& sph_body, Vecd mass_center, Real position_1, Real position_2)
		: QuantitySummation<Real>(sph_body, "Mass"),
		pos_(*particles_->getVariableDataByName<Vecd>("Position")), 
        mass_center_(mass_center), p_1_(position_1), p_2_(position_2)
	{
		this->quantity_name_ = "Moment of Inertia";
	};
	virtual ~QuantityMomentOfInertia() {};

	Real reduce(size_t index_i, Real dt = 0.0)
	{
		if (p_1_ == p_2_)
		{
			return  ((pos_[index_i] - mass_center_).norm() * (pos_[index_i] - mass_center_).norm()
				- (pos_[index_i][p_1_] - mass_center_[p_1_]) * (pos_[index_i][p_2_] - mass_center_[p_2_]))
                * this->variable_[index_i];
		}
		else
		{
			return -(pos_[index_i][p_1_] - mass_center_[p_1_]) * (pos_[index_i][p_2_] - mass_center_[p_2_])
                * this->variable_[index_i];
		}
	};
};

class QuantityMassPosition : public QuantitySummation<Vecd>
{
protected:
	StdLargeVec<Real>& mass_;


public:
	explicit QuantityMassPosition(SPHBody& sph_body)
		: QuantitySummation<Vecd>(sph_body, "Position"),
		mass_(*particles_->getVariableDataByName<Real>("Mass"))
	{
		this->quantity_name_ = "Mass*Position";
	};
	virtual ~QuantityMassPosition() {};

	Vecd reduce(size_t index_i, Real dt = 0.0)
	{
		return this->variable_[index_i] * mass_[index_i];
	};
};

class Constrain3DSolidBodyRotation : public LocalDynamics, public DataDelegateSimple
{
private:
	Vecd mass_center_;
	Matd moment_of_inertia_;
	Vecd angular_velocity_;
	Vecd linear_velocity_;
	ReduceDynamics<QuantityMomentOfMomentum> compute_total_moment_of_momentum_;
	StdLargeVec<Vecd>& vel_;
	StdLargeVec<Vecd>& pos_;

protected:
	virtual void setupDynamics(Real dt = 0.0) override
	{
		angular_velocity_ = moment_of_inertia_.inverse() * compute_total_moment_of_momentum_.exec(dt);
	}

public:
	explicit Constrain3DSolidBodyRotation(SPHBody& sph_body, Vecd mass_center, Matd inertia_tensor)
		: LocalDynamics(sph_body), DataDelegateSimple(sph_body),
		vel_(*particles_->getVariableDataByName<Vecd>("Velocity")), 
        pos_(*particles_->getVariableDataByName<Vecd>("Position")), 
        compute_total_moment_of_momentum_(sph_body, mass_center),
		mass_center_(mass_center), moment_of_inertia_(inertia_tensor) {}

	virtual ~Constrain3DSolidBodyRotation() {};

	void update(size_t index_i, Real dt = 0.0)
	{
		linear_velocity_ = angular_velocity_.cross((pos_[index_i] - mass_center_));
		vel_[index_i] -= linear_velocity_;
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
    shell_body.defineAdaptation<SPHAdaptation>(1.15, 1.0);
    shell_body.defineBodyLevelSetShape(2.0)->correctLevelSetSign()->writeLevelSet(sph_system);
    shell_body.defineMaterial<SaintVenantKirchhoffSolid>(rho0_s, Youngs_modulus, poisson);
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? shell_body.generateParticles<SurfaceParticles, Reload>(shell_body.getName())
        : shell_body.generateParticles<SurfaceParticles, FromSTLFile>(mesh_shape);
    //----------------------------------------------------------------------
    //	SPH Particle relaxation section
    //----------------------------------------------------------------------
    /** check whether run particle relaxation for body fitted particle distribution. */
    if (sph_system.RunParticleRelaxation())
    {
        InnerRelation imported_model_inner(shell_body);
        BodyAlignedBoxByCell inlet_detection_box(shell_body,
                                             makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(inlet_rotation), Vec3d(inlet_translation)), inlet_half));
        BodyAlignedBoxByCell outlet01_detection_box(shell_body,
                                                makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_01_rotation), Vec3d(outlet_01_translation)), outlet_01_half));
        BodyAlignedBoxByCell outlet02_detection_box(shell_body,
                                                makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_02_rotation), Vec3d(outlet_02_translation)), outlet_02_half));

        RealBody test_body_in(
        sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(inlet_rotation), Vec3d(inlet_translation)), inlet_half, "TestBodyIn"));
        test_body_in.generateParticles<BaseParticles, Lattice>();

        RealBody test_body_out01(
        sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_01_rotation), Vec3d(outlet_01_translation)), outlet_01_half, "TestBodyOut01"));
        test_body_out01.generateParticles<BaseParticles, Lattice>();

        RealBody test_body_out02(
        sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_02_rotation), Vec3d(outlet_02_translation)), outlet_02_half, "TestBodyOut02"));
        test_body_out02.generateParticles<BaseParticles, Lattice>();
        //----------------------------------------------------------------------
        //	Methods used for particle relaxation.
        //----------------------------------------------------------------------
        using namespace relax_dynamics;
        //SimpleDynamics<RandomizeParticlePosition> random_imported_model_particles(shell_body);
        /** A  Physics relaxation step. */
        SurfaceRelaxationStep relaxation_step_inner(imported_model_inner);
        ShellNormalDirectionPrediction shell_normal_prediction(imported_model_inner, thickness);

        // here, need a class to switch particles in aligned box to ghost particles (not real particles)
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> inlet_particles_detection(inlet_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet01_particles_detection(outlet01_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet02_particles_detection(outlet02_detection_box);

        /** Write the body state to Vtp file. */
        BodyStatesRecordingToVtp write_imported_model_to_vtp({shell_body});
        write_imported_model_to_vtp.addToWrite<Vecd>(shell_body, "NormalDirection");
        BodyStatesRecordingToVtp write_all_bodies_to_vtp({sph_system});
        /** Write the particle reload files. */
        ReloadParticleIO write_particle_reload_files(shell_body);
        //----------------------------------------------------------------------
        //	Particle relaxation starts here.
        //----------------------------------------------------------------------
        //random_imported_model_particles.exec(0.25);
        relaxation_step_inner.getOnSurfaceBounding().exec();
        write_imported_model_to_vtp.writeToFile(0.0);
        shell_body.updateCellLinkedList();
        // write_cell_linked_list.writeToFile(0.0);
        //----------------------------------------------------------------------
        //	Particle relaxation time stepping start here.
        //----------------------------------------------------------------------
        int ite_p = 0;
        while (ite_p < 100000)
        {
            relaxation_step_inner.exec();
            ite_p += 1;
            if (ite_p % 5000 == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "Relaxation steps for the imported model N = " << ite_p << "\n";
                write_imported_model_to_vtp.writeToFile(ite_p);
            }
        }
        std::cout << "The physics relaxation process of imported model finish !" << std::endl;

        //shell_normal_prediction.exec();

        inlet_particles_detection.exec();
        shell_body.updateCellLinkedListWithParticleSort(100);
        outlet01_particles_detection.exec();
        shell_body.updateCellLinkedListWithParticleSort(100);
        outlet02_particles_detection.exec();
        shell_body.updateCellLinkedListWithParticleSort(100);
        write_all_bodies_to_vtp.writeToFile(ite_p);
        write_particle_reload_files.writeToFile(0);

        shell_normal_prediction.exec();

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
    FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBody"));
    water_block.defineBodyLevelSetShape()->cleanLevelSet();
    water_block.defineMaterial<WeaklyCompressibleFluid>(rho0_f, c_f, mu_f);
    ParticleBuffer<ReserveSizeFactor> in_outlet_particle_buffer(0.5);
    water_block.generateParticlesWithReserve<BaseParticles, Lattice>(in_outlet_particle_buffer);
    
    // for buffer location test
    /*RealBody buffer_body_in(
        sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(inlet_rotation), Vec3d(inlet_buffer_translation)), inlet_half, "BufferBodyIn"));
    buffer_body_in.generateParticles<BaseParticles, Lattice>();
    RealBody buffer_body_out01(
        sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_01_rotation), Vec3d(outlet_01_buffer_translation)), outlet_01_half, "BufferBodyOut01"));
    buffer_body_out01.generateParticles<BaseParticles, Lattice>();
    RealBody buffer_body_out02(
        sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_02_rotation), Vec3d(outlet_02_buffer_translation)), outlet_02_half, "BufferBodyOut02"));
    buffer_body_out02.generateParticles<BaseParticles, Lattice>();*/

    InnerRelation water_block_inner(water_block);
    InnerRelation shell_inner(shell_body);
    ContactRelationFromShellToFluid water_shell_contact(water_block, {&shell_body}, {false});
    ContactRelationFromFluidToShell shell_water_contact(shell_body, {&water_block}, {false});
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
    InteractionDynamics<thin_structure_dynamics::ShellCorrectConfiguration> shell_corrected_configuration(shell_inner);
    Dynamics1Level<thin_structure_dynamics::ShellStressRelaxationFirstHalf> shell_stress_relaxation_first(shell_inner, 3, true);
    Dynamics1Level<thin_structure_dynamics::ShellStressRelaxationSecondHalf> shell_stress_relaxation_second(shell_inner);
    ReduceDynamics<thin_structure_dynamics::ShellAcousticTimeStepSize> shell_time_step_size(shell_body);
    SimpleDynamics<thin_structure_dynamics::AverageShellCurvature> shell_average_curvature(shell_curvature_inner);
    SimpleDynamics<thin_structure_dynamics::UpdateShellNormalDirection> shell_update_normal(shell_body);

    /** Exert constrain on shell. */
	SimpleDynamics<solid_dynamics::ConstrainSolidBodyMassCenter> constrain_mass_center(shell_body);
	ReduceDynamics<QuantitySummation<Real>> compute_total_mass_(shell_body, "Mass");
	ReduceDynamics<QuantityMassPosition> compute_mass_position_(shell_body);
	Vecd mass_center = compute_mass_position_.exec() / compute_total_mass_.exec();
	Matd moment_of_inertia = Matd::Zero();
	for (int i = 0; i != Dimensions; ++i)
	{
		for (int j = 0; j != Dimensions; ++j)
		{
			ReduceDynamics<QuantityMomentOfInertia> compute_moment_of_inertia(shell_body, mass_center, i, j);
			moment_of_inertia(i, j) = compute_moment_of_inertia.exec();
		}
	}
	SimpleDynamics<Constrain3DSolidBodyRotation> constrain_rotation(shell_body, mass_center, moment_of_inertia);


    // fluid dynamics
    TimeDependentAcceleration time_dependent_acceleration(Vecd::Zero());
    SimpleDynamics<GravityForce> apply_gravity_force(water_block, time_dependent_acceleration);

    InteractionDynamics<NablaWVComplex> kernel_summation(water_block_inner, water_shell_contact);
    InteractionWithUpdate<SpatialTemporalFreeSurfaceIndicationComplex> boundary_indicator(water_block_inner, water_shell_contact);
    Dynamics1Level<fluid_dynamics::Integration1stHalfWithWallRiemann> pressure_relaxation(water_block_inner, water_shell_contact);
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallRiemann> density_relaxation(water_block_inner, water_shell_contact);
    ReduceDynamics<fluid_dynamics::AdvectionTimeStepSize> get_fluid_advection_time_step_size(water_block, U_f);
    ReduceDynamics<fluid_dynamics::AcousticTimeStepSize> get_fluid_time_step_size(water_block);
    InteractionWithUpdate<ComplexInteraction<fluid_dynamics::ViscousForce<Inner<>, Contact<Wall>>,
                                             fluid_dynamics::FixedViscosity, NoKernelCorrection>>
        viscous_acceleration(water_block_inner, water_shell_contact);
    // InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> viscous_acceleration(water_block_inner, water_shell_contact);
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionComplex<BulkParticles>> transport_velocity_correction(water_block_inner, water_shell_contact);

    BodyAlignedBoxByCell left_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(inlet_emitter_rotation), Vec3d(inlet_translation)), inlet_half));
    fluid_dynamics::NonPrescribedPressureBidirectionalBuffer left_emitter_inflow_injection(left_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell right_emitter_01(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_01_rotation), Vec3d(outlet_01_translation)), outlet_01_half));
    fluid_dynamics::BidirectionalBuffer<RightInflowPressure> right_emitter_inflow_injection_01(right_emitter_01, in_outlet_particle_buffer);
    BodyAlignedBoxByCell right_emitter_02(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_02_rotation), Vec3d(outlet_02_translation)), outlet_02_half));
    fluid_dynamics::BidirectionalBuffer<RightInflowPressure> right_emitter_inflow_injection_02(right_emitter_02, in_outlet_particle_buffer);

    BodyAlignedBoxByCell left_disposer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(inlet_rotation),Vec3d(inlet_translation)), inlet_half));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> left_disposer_outflow_deletion(left_disposer);
    BodyAlignedBoxByCell right_disposer_01(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_01_rotation), Vec3d(outlet_01_translation)), outlet_01_half));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> right_disposer_outflow_deletion_01(right_disposer_01);
    BodyAlignedBoxByCell right_disposer_02(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_02_rotation), Vec3d(outlet_02_translation)), outlet_02_half));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> right_disposer_outflow_deletion_02(right_disposer_02);

    InteractionWithUpdate<fluid_dynamics::DensitySummationPressureComplex> update_fluid_density(water_block_inner, water_shell_contact);
    SimpleDynamics<fluid_dynamics::PressureCondition<LeftInflowPressure>> left_inflow_pressure_condition(left_emitter);
    SimpleDynamics<fluid_dynamics::PressureCondition<RightInflowPressure>> right_inflow_pressure_condition_01(right_emitter_01);
    SimpleDynamics<fluid_dynamics::PressureCondition<RightInflowPressure>> right_inflow_pressure_condition_02(right_emitter_02);
    SimpleDynamics<fluid_dynamics::InflowVelocityCondition<InflowVelocity>> inflow_velocity_condition(left_emitter);
    

    // FSI
    InteractionWithUpdate<solid_dynamics::ViscousForceFromFluid> viscous_force_on_shell(shell_water_contact);
    InteractionWithUpdate<solid_dynamics::PressureForceFromFluid<decltype(density_relaxation)>> pressure_force_on_shell(shell_water_contact);
    solid_dynamics::AverageVelocityAndAcceleration average_velocity_and_acceleration(shell_body);
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations, observations
    //	and regression tests of the simulation.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp body_states_recording(sph_system);
    body_states_recording.addToWrite<Real>(water_block, "Pressure");
    body_states_recording.addToWrite<int>(water_block, "Indicator");
    body_states_recording.addToWrite<Real>(water_block, "Density");
    body_states_recording.addToWrite<int>(water_block, "BufferParticleIndicator");

    //body_states_recording.addToWrite<Real>(shell_body, "VonMisesStress");
    //body_states_recording.addToWrite<Real>(shell_body, "VonMisesStrain");
    body_states_recording.addToWrite<Vecd>(shell_body, "PressureForceFromFluid");
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    sph_system.initializeSystemConfigurations();
    shell_corrected_configuration.exec();
    shell_average_curvature.exec();
    water_block_complex.updateConfiguration();
    shell_water_contact.updateConfiguration();
    boundary_indicator.exec();
    left_emitter_inflow_injection.tag_buffer_particles.exec();
    right_emitter_inflow_injection_01.tag_buffer_particles.exec();
    right_emitter_inflow_injection_02.tag_buffer_particles.exec();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    size_t number_of_iterations = sph_system.RestartStep();
    int screen_output_interval = 100;
    int observation_sample_interval = screen_output_interval * 2;
    Real end_time = 2.5;   /**< End time. */
    Real Output_Time = 0.1; /**< Time stamps for output of body states. */
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
            apply_gravity_force.exec();
            Real Dt = get_fluid_advection_time_step_size.exec();
            //std::cout << "Dt = " << Dt << std::endl;
            update_fluid_density.exec();
            viscous_acceleration.exec();
            transport_velocity_correction.exec();
            interval_computing_time_step += TickCount::now() - time_instance;

            time_instance = TickCount::now();
            Real relaxation_time = 0.0;
            while (relaxation_time < Dt)
            {
                dt = SMIN(get_fluid_time_step_size.exec(), Dt);
                //std::cout << "dt = " << dt << std::endl;

                pressure_relaxation.exec(dt);
                kernel_summation.exec();
                left_inflow_pressure_condition.exec(dt);
                right_inflow_pressure_condition_01.exec(dt);
                right_inflow_pressure_condition_02.exec(dt);
                inflow_velocity_condition.exec();
                density_relaxation.exec(dt);

                Real dt_s_sum = 0.0;
                average_velocity_and_acceleration.initialize_displacement_.exec();
                while (dt_s_sum < dt)
                {
                    dt_s = 0.5 * shell_time_step_size.exec(); // why 0.5?
                    if (dt - dt_s_sum < dt_s)
                        dt_s = dt - dt_s_sum;
                    shell_stress_relaxation_first.exec(dt_s);

                    constrain_rotation.exec(dt_s);
					constrain_mass_center.exec(dt_s);

                    shell_stress_relaxation_second.exec(dt_s);
                    dt_s_sum += dt_s;
                }
                average_velocity_and_acceleration.update_averages_.exec(dt);

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

                body_states_recording.writeToFile();
            }
            number_of_iterations++;

            time_instance = TickCount::now();

            left_emitter_inflow_injection.injection.exec();
            right_emitter_inflow_injection_01.injection.exec();
            right_emitter_inflow_injection_02.injection.exec();
            left_disposer_outflow_deletion.exec();
            right_disposer_outflow_deletion_01.exec();
            right_disposer_outflow_deletion_02.exec();

            water_block.updateCellLinkedListWithParticleSort(100);
            shell_update_normal.exec();
            shell_body.updateCellLinkedList();
            shell_curvature_inner.updateConfiguration();
            shell_average_curvature.exec();
            shell_water_contact.updateConfiguration();
            water_block_complex.updateConfiguration();
            interval_updating_configuration += TickCount::now() - time_instance;
            boundary_indicator.exec();

            left_emitter_inflow_injection.tag_buffer_particles.exec();
            right_emitter_inflow_injection_01.tag_buffer_particles.exec();
            right_emitter_inflow_injection_02.tag_buffer_particles.exec();
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
