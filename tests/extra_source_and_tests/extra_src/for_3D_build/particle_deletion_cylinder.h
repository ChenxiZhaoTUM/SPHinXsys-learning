/* @file 	  particle_deletion_cylinder.h

 */

#ifndef PARTICLE_DELETION_CYLINDER_H
#define PARTICLE_DELETION_CYLINDER_H

#include "TriangleMeshDistance.h"
#include "all_simbody.h"
#include "base_geometry.h"
#include "triangle_mesh_shape.h"
#include "base_body_part.h"
#include "all_particle_dynamics.h"

namespace SPH
{
class AlignedCylinderShape : public TriangleMeshShapeCylinder
{
    const int alignment_axis_;
    SimTK::UnitVec3 cylinder_length_axis_;
    Real radius_;
    Real halflength_;
    Vec3d translation_;

  public:
    /** construct directly */
    template <typename... Args>
    explicit AlignedCylinderShape(int upper_bound_axis, SimTK::UnitVec3 cylinder_length_axis, Real radius, Real halflength, int resolution, Vec3d translation,
                                       const std::string &shape_name = "TriangleMeshShapeCylinder")
        : TriangleMeshShapeCylinder(cylinder_length_axis, radius, halflength, resolution, translation, shape_name),
          alignment_axis_(upper_bound_axis), cylinder_length_axis_(cylinder_length_axis), radius_(radius), halflength_(halflength), translation_(translation) {};
    virtual ~AlignedCylinderShape(){};

    bool checkInBounds(const Vecd &probe_point);
    /*bool checkUpperBound(const Vecd &probe_point);
    bool checkLowerBound(const Vecd &probe_point);
    bool checkNearUpperBound(const Vecd &probe_point, Real threshold);
    bool checkNearLowerBound(const Vecd &probe_point, Real threshold);
    Vecd getUpperPeriodic(const Vecd &probe_point);
    Vecd getLowerPeriodic(const Vecd &probe_point);*/
    int AlignmentAxis() { return alignment_axis_; };
};

template <class BodyRegionType>
class AlignedCylinderRegion : public BodyRegionType
{
  public:
    AlignedCylinderRegion(RealBody &real_body, AlignedCylinderShape &aligned_cylinder)
        : BodyRegionType(real_body, aligned_cylinder), aligned_cylinder_(aligned_cylinder){};
    AlignedCylinderRegion(RealBody& real_body, SharedPtr<AlignedCylinderShape> aligned_cylinder_ptr)
        : BodyRegionType(real_body, aligned_cylinder_ptr), aligned_cylinder_(*aligned_cylinder_ptr.get()){};
    virtual ~AlignedCylinderRegion(){};
    AlignedCylinderShape &getAlignedCylinderShape() { return aligned_cylinder_; };

  protected:
    AlignedCylinderShape &aligned_cylinder_;
};

using BodyAlignedCylinderByCell = AlignedCylinderRegion<BodyRegionByCell>;

namespace relax_dynamics
{
class ParticlesInAlignedCylinderDetectionByCell : public BaseLocalDynamics<BodyPartByCell>, public DataDelegateSimple
{
  public:
    ParticlesInAlignedCylinderDetectionByCell(BodyAlignedCylinderByCell &aligned_cylinder_part);
    virtual ~ParticlesInAlignedCylinderDetectionByCell(){};

    void update(size_t index_i, Real dt = 0.0);

  protected:
    std::mutex mutex_switch_to_ghost_; /**< mutex exclusion for memory conflict */
    StdLargeVec<Vecd> &pos_;
    AlignedCylinderShape &aligned_cylinder_;
};
} // namespace relax_dynamics
} // namespace SPH
#endif // PARTICLE_DELETION_CYLINDER_H
