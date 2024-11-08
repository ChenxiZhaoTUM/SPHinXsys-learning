/* @file 	  arbitrary_shape_buffer_3d.h

 */

#ifndef ARBITRARY_SHAPE_BUFFER_3D_H
#define ARBITRARY_SHAPE_BUFFER_3D_H

#include "particle_generation_and_detection.h"
#include "TriangleMeshDistance.h"
#include "all_simbody.h"
#include "base_geometry.h"
#include "geometric_shape.h"
#include "triangle_mesh_shape.h"
#include "base_body_part.h"
#include "all_particle_dynamics.h"
#include "fluid_boundary.h"

namespace SPH
{
class AlignedCylinderShape : public TransformShape<GeometricShapeCylinder>
{
  const int alignment_axis_;

  public:
    /** construct directly */
    template <typename... Args>
    explicit AlignedCylinderShape(int upper_bound_axis, const Transform &transform, Args &&... args)
        : TransformShape<GeometricShapeCylinder>(transform, std::forward<Args>(args)...), 
        alignment_axis_(upper_bound_axis) {};  //upper_bound_axis default x
    /** construct from a shape already has aligned boundaries */
    // not sure if it is correct in Cylinder shape
    template <typename... Args>
    explicit AlignedCylinderShape(int upper_bound_axis, const Shape &shape, Args &&... args)
        : TransformShape<GeometricShapeCylinder>(
              Transform(Vecd(0.5 * (shape.bounding_box_.second_ + shape.bounding_box_.first_))),
              0.5 * (shape.bounding_box_.second_ - shape.bounding_box_.first_), std::forward<Args>(args)...),
        alignment_axis_(upper_bound_axis) {};
    virtual ~AlignedCylinderShape(){};

    Real HalfLength() { return halflength_; }
    Real radius() { return radius_; }
    bool checkInBounds(const Vecd &probe_point, Real lower_bound_fringe=0.0, Real upper_bound_fringe=0.0);
    bool checkUpperBound(const Vecd &probe_point, Real upper_bound_fringe=0.0);
    bool checkLowerBound(const Vecd &probe_point, Real lower_bound_fringe=0.0);
    bool checkNearUpperBound(const Vecd &probe_point, Real threshold);
    bool checkNearLowerBound(const Vecd &probe_point, Real threshold);
    Vecd getUpperPeriodic(const Vecd &probe_point);
    Vecd getLowerPeriodic(const Vecd &probe_point);
    int AlignmentAxis() { return alignment_axis_; };
};

class AlignedCylinderShapeByTriangleMesh : public TriangleMeshShapeCylinder
{
    const int alignment_axis_;
    SimTK::UnitVec3 cylinder_length_axis_;
    Real radius_;
    Real halflength_;
    Vec3d translation_;

  public:
    /** construct directly */
    template <typename... Args>
    explicit AlignedCylinderShapeByTriangleMesh(int upper_bound_axis, SimTK::UnitVec3 cylinder_length_axis, Real radius, Real halflength, int resolution, Vec3d translation,
                                       const std::string &shape_name = "TriangleMeshShapeCylinder")
        : TriangleMeshShapeCylinder(cylinder_length_axis, radius, halflength, resolution, translation, shape_name),
          alignment_axis_(upper_bound_axis),
          cylinder_length_axis_(cylinder_length_axis), radius_(radius), halflength_(halflength), translation_(translation) {};
    virtual ~AlignedCylinderShapeByTriangleMesh(){};

    SimTK::UnitVec3 LengthAxis() { return cylinder_length_axis_; }
    Vec3d Translation() { return translation_;  }
    bool checkInBounds(const Vecd &probe_point, Real lower_bound_fringe=0.0, Real upper_bound_fringe=0.0);
    bool checkUpperBound(const Vecd &probe_point, Real upper_bound_fringe=0.0);
    bool checkLowerBound(const Vecd &probe_point, Real lower_bound_fringe=0.0);
    bool checkNearUpperBound(const Vecd &probe_point, Real threshold);
    bool checkNearLowerBound(const Vecd &probe_point, Real threshold);
    Vecd getUpperPeriodic(const Vecd &probe_point);
    Vecd getLowerPeriodic(const Vecd &probe_point);
    int AlignmentAxis() { return alignment_axis_; };
};

using BodyAlignedCylinderByCell = BaseAlignedRegion<BodyRegionByCell, AlignedCylinderShape>;
using BodyAlignedCylinderByCellByTriangleMesh = BaseAlignedRegion<BodyRegionByCell, AlignedCylinderShapeByTriangleMesh>;

namespace relax_dynamics
{
using DeleteParticlesInCylinder = ParticlesInAlignedRegionDetectionByCell<AlignedCylinderShape>;
using DeleteParticlesInCylinderByTriangleMesh = ParticlesInAlignedRegionDetectionByCell<AlignedCylinderShapeByTriangleMesh>;
} // namespace relax_dynamics


namespace fluid_dynamics
{
template <typename TargetVelocity, typename AlignedShapeType>
class InflowVelocityConditionArb : public BaseFlowBoundaryCondition
{
  public:
    /** default parameter indicates prescribe velocity */
    explicit InflowVelocityConditionArb(BaseAlignedRegion<BodyRegionByCell, AlignedShapeType>& aligned_region_part, Real relaxation_rate = 1.0)
        : BaseFlowBoundaryCondition(aligned_region_part),
          relaxation_rate_(relaxation_rate), aligned_shape_(aligned_region_part.getAlignedShape()),
          transform_(aligned_shape_.getTransform()),
          target_velocity(*this),
          physical_time_(sph_system_.getSystemVariableDataByName<Real>("PhysicalTime")) {};
    virtual ~InflowVelocityConditionArb(){};
    AlignedShapeType &getAlignedShape() { return aligned_shape_; };

    void update(size_t index_i, Real dt = 0.0)
    {
        if (aligned_shape_.checkInBounds(pos_[index_i]))
        {
            Vecd frame_position = transform_.shiftBaseStationToFrame(pos_[index_i]);
            Vecd frame_velocity = transform_.xformBaseVecToFrame(vel_[index_i]);
            Vecd relaxed_frame_velocity = target_velocity(frame_position, frame_velocity, *physical_time_) * relaxation_rate_ +
                                          frame_velocity * (1.0 - relaxation_rate_);
            vel_[index_i] = transform_.xformFrameVecToBase(relaxed_frame_velocity);
        }
    };

  protected:
    Real relaxation_rate_;
    AlignedShapeType &aligned_shape_;
    Transform &transform_;
    TargetVelocity target_velocity;
    Real *physical_time_;
};

template <typename TargetPressure, typename AlignedShapeType>
class PressureConditionArb : public BaseFlowBoundaryCondition
{
  public:
    /** default parameter indicates prescribe pressure */
    template <typename... Args>
    explicit PressureConditionArb(BaseAlignedRegion<BodyRegionByCell, AlignedShapeType>& aligned_region_part, Args &&...args)
        : BaseFlowBoundaryCondition(aligned_region_part),
          aligned_shape_(aligned_region_part.getAlignedShape()),
          alignment_axis_(aligned_shape_.AlignmentAxis()),
          transform_(aligned_shape_.getTransform()),
          target_pressure_(TargetPressure(aligned_region_part)),
          kernel_sum_(particles_->getVariableDataByName<Vecd>("KernelSummation")),
          physical_time_(sph_system_.getSystemVariableDataByName<Real>("PhysicalTime")) {};
    virtual ~PressureConditionArb(){};
    AlignedShapeType &getAlignedShape() { return aligned_shape_; };

    TargetPressure *getTargetPressure() { return &target_pressure_; }

    void update(size_t index_i, Real dt = 0.0)
    {
        vel_[index_i] += 2.0 * kernel_sum_[index_i] * target_pressure_(p_[index_i], *physical_time_) / rho_[index_i] * dt;

        Vecd frame_velocity = Vecd::Zero();
        frame_velocity[alignment_axis_] = transform_.xformBaseVecToFrame(vel_[index_i])[alignment_axis_];
        vel_[index_i] = transform_.xformFrameVecToBase(frame_velocity);
    };

  protected:
    AlignedShapeType &aligned_shape_;
    const int alignment_axis_;
    Transform &transform_;
    TargetPressure target_pressure_;
    Vecd *kernel_sum_;
    Real *physical_time_;
};

//-------------------------------------------------------------------------------
//	InflowVelocityCondition and PressureCondition use Cylinder by TriangleMesh.
//-------------------------------------------------------------------------------
template <typename TargetVelocity>
class InflowVelocityConditionCylinderByTriangleMesh : public BaseFlowBoundaryCondition
{
  public:
    /** default parameter indicates prescribe velocity */
    explicit InflowVelocityConditionCylinderByTriangleMesh(BodyAlignedCylinderByCellByTriangleMesh& aligned_region_part, Real relaxation_rate = 1.0)
        : BaseFlowBoundaryCondition(aligned_region_part),
          relaxation_rate_(relaxation_rate), aligned_cylinder_(aligned_region_part.getAlignedShape()),
          cylinder_length_axis_(aligned_cylinder_.LengthAxis()), translation_(aligned_cylinder_.Translation()),
          target_velocity(*this),
          physical_time_(sph_system_.getSystemVariableDataByName<Real>("PhysicalTime")) {};
    virtual ~InflowVelocityConditionCylinderByTriangleMesh(){};

    AlignedCylinderShapeByTriangleMesh &getAlignedShape() { return aligned_cylinder_; };

    void update(size_t index_i, Real dt = 0.0)
    {
        if (aligned_cylinder_.checkInBounds(pos_[index_i]))
        {
            Vecd frame_position = pos_[index_i] - translation_;
            Vecd cylinder_length_axis(cylinder_length_axis_[0], cylinder_length_axis_[1], cylinder_length_axis_[2]);
            Vecd frame_velocity = cylinder_length_axis * vel_[index_i].dot(cylinder_length_axis);
            Vecd relaxed_frame_velocity = target_velocity(frame_position, frame_velocity, *physical_time_) * relaxation_rate_ +
                                          frame_velocity * (1.0 - relaxation_rate_);
            vel_[index_i] = relaxed_frame_velocity.dot(cylinder_length_axis) * cylinder_length_axis;
        }
    };

  protected:
    Real relaxation_rate_;
    AlignedCylinderShapeByTriangleMesh &aligned_cylinder_;
    SimTK::UnitVec3 cylinder_length_axis_;
    Vec3d translation_;
    TargetVelocity target_velocity;
    Real *physical_time_;
};

template <typename TargetPressure>
class PressureConditionCylinderByTriangleMesh : public BaseFlowBoundaryCondition
{
  public:
    /** default parameter indicates prescribe pressure */
    template <typename... Args>
    explicit PressureConditionCylinderByTriangleMesh(BodyAlignedCylinderByCellByTriangleMesh& aligned_region_part, Args &&...args)
        : BaseFlowBoundaryCondition(aligned_region_part),
          aligned_cylinder_(aligned_region_part.getAlignedShape()),
          cylinder_length_axis_(aligned_cylinder_.LengthAxis()), translation_(aligned_cylinder_.Translation()),
          target_pressure_(TargetPressure(aligned_region_part, std::forward<Args>(args)...)),
          kernel_sum_(particles_->getVariableDataByName<Vecd>("KernelSummation")),
          physical_time_(sph_system_.getSystemVariableDataByName<Real>("PhysicalTime")) {};
    virtual ~PressureConditionCylinderByTriangleMesh(){};

    AlignedCylinderShapeByTriangleMesh &getAlignedShape() { return aligned_cylinder_; };

    TargetPressure *getTargetPressure() { return &target_pressure_; }

    void update(size_t index_i, Real dt = 0.0)
    {
        vel_[index_i] += 2.0 * kernel_sum_[index_i] * target_pressure_(p_[index_i], *physical_time_) / rho_[index_i] * dt;

        Vecd cylinder_length_axis(cylinder_length_axis_[0], cylinder_length_axis_[1], cylinder_length_axis_[2]);
        Vecd frame_velocity = cylinder_length_axis * vel_[index_i].dot(cylinder_length_axis);

        vel_[index_i] = frame_velocity.dot(cylinder_length_axis) * cylinder_length_axis;
    };

  protected:
    AlignedCylinderShapeByTriangleMesh &aligned_cylinder_;
    SimTK::UnitVec3 cylinder_length_axis_;
    Vec3d translation_;
    TargetPressure target_pressure_;
    Vecd *kernel_sum_;
    Real *physical_time_;
};
} // namespace fluid_dynamics
} // namespace SPH
#endif // ARBITRARY_SHAPE_BUFFER_3D_H
