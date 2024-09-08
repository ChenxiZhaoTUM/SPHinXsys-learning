#include "particle_deletion_cylinder.h"

namespace SPH
{
bool AlignedCylinderShape::checkInBounds(const Vecd &probe_point)
{
    SimTKVec3 probe_point_simtk(probe_point[0], probe_point[1], probe_point[2]);
    SimTKVec3 translation_simtk(translation_[0], translation_[1], translation_[2]);
    SimTKVec3 relative_position = probe_point_simtk - translation_simtk;
    
    Real distance_along_axis = dot(relative_position, cylinder_length_axis_);
    if (distance_along_axis < -halflength_ || distance_along_axis > halflength_)
        return false;

    SimTKVec3 perpendicular_position = relative_position - distance_along_axis * cylinder_length_axis_;
    Real distance_perpendicular = perpendicular_position.norm();

    return distance_perpendicular <= radius_ ? true : false;
}
namespace relax_dynamics
{
//=================================================================================================//
ParticlesInAlignedCylinderDetectionByCell::
    ParticlesInAlignedCylinderDetectionByCell(BodyAlignedCylinderByCell &aligned_cylinder_part)
    : BaseLocalDynamics<BodyPartByCell>(aligned_cylinder_part),
      DataDelegateSimple(aligned_cylinder_part.getSPHBody()),
      pos_(*particles_->getVariableDataByName<Vecd>("Position")),
      aligned_cylinder_(aligned_cylinder_part.getAlignedCylinderShape()) {}
//=================================================================================================//
void ParticlesInAlignedCylinderDetectionByCell::update(size_t index_i, Real dt)
{
    mutex_switch_to_ghost_.lock();
    while (aligned_cylinder_.checkInBounds(pos_[index_i]) && index_i < particles_->TotalRealParticles())
    {
        particles_->switchToBufferParticle(index_i);
    }
    mutex_switch_to_ghost_.unlock();
}
//=================================================================================================//
} // namespace relax_dynamics
} // namespace SPH