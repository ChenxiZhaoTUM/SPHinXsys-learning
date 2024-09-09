/* @file 	  arbitrary_shape_buffer_3d.h

 */

#ifndef ARBITRARY_SHAPE_BUFFER_3D_H
#define ARBITRARY_SHAPE_BUFFER_3D_H

#include "arbitrary_shape_buffer.h"
#include "TriangleMeshDistance.h"
#include "all_simbody.h"
#include "base_geometry.h"
#include "triangle_mesh_shape.h"
#include "base_body_part.h"
#include "all_particle_dynamics.h"

namespace SPH
{
class AlignedCylinderShape : public TriangleMeshShapeCylinder, public BaseAlignedShape
{
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
          BaseAlignedShape(upper_bound_axis), cylinder_length_axis_(cylinder_length_axis), radius_(radius), halflength_(halflength), translation_(translation) {};
    virtual ~AlignedCylinderShape(){};

    bool checkInBounds(const Vecd &probe_point) override;
    /*bool checkUpperBound(const Vecd &probe_point);
    bool checkLowerBound(const Vecd &probe_point);
    bool checkNearUpperBound(const Vecd &probe_point, Real threshold);
    bool checkNearLowerBound(const Vecd &probe_point, Real threshold);
    Vecd getUpperPeriodic(const Vecd &probe_point);
    Vecd getLowerPeriodic(const Vecd &probe_point);*/
};

using BodyAlignedCylinderByCell = BaseAlignedRegion<BodyRegionByCell, AlignedCylinderShape>;

namespace relax_dynamics
{
using DeleteParticlesInCylinder = ParticlesInAlignedRegionDetectionByCell<AlignedCylinderShape>;
} // namespace relax_dynamics
} // namespace SPH
#endif // ARBITRARY_SHAPE_BUFFER_3D_H
