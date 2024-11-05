#include "particle_generation_and_detection.h"
namespace SPH
{
namespace relax_dynamics
{
//=================================================================================================//
ParticlesInAlignedBoxDetectionByCell::
    ParticlesInAlignedBoxDetectionByCell(BodyAlignedBoxByCell &aligned_box_part)
    : BaseLocalDynamics<BodyPartByCell>(aligned_box_part),
      pos_(particles_->getVariableDataByName<Vecd>("Position")),
      aligned_box_(aligned_box_part.getAlignedBoxShape()) {}
//=================================================================================================//
void ParticlesInAlignedBoxDetectionByCell::update(size_t index_i, Real dt)
{
    mutex_switch_to_buffer_.lock();
    while (aligned_box_.checkInBounds(pos_[index_i]) && index_i < particles_->TotalRealParticles())
    {
        particles_->switchToBufferParticle(index_i);
    }
    mutex_switch_to_buffer_.unlock();
}
//=================================================================================================//
OnSurfaceBounding::OnSurfaceBounding(RealBody &real_body_)
    : LocalDynamics(real_body_),
      pos_(particles_->getVariableDataByName<Vecd>("Position"))
{
    shape_ = &real_body_.getInitialShape();
}
//=================================================================================================//
void OnSurfaceBounding::update(size_t index_i, Real dt)
{
    pos_[index_i] = shape_->findClosestPoint(pos_[index_i]);
}
//=================================================================================================//
SurfaceRelaxationStep::SurfaceRelaxationStep(BaseInnerRelation &inner_relation)
    : BaseDynamics<void>(),
      real_body_(DynamicCast<RealBody>(this, inner_relation.getSPHBody())),
      inner_relation_(inner_relation),
      relaxation_residue_(inner_relation),
      relaxation_scaling_(real_body_), position_relaxation_(real_body_),
      on_surface_bounding_(real_body_) {}
//=================================================================================================//
void SurfaceRelaxationStep::exec(Real ite_p)
{
    real_body_.updateCellLinkedList();
    inner_relation_.updateConfiguration();
    relaxation_residue_.exec();
    Real scaling = relaxation_scaling_.exec();
    position_relaxation_.exec(scaling);
    on_surface_bounding_.exec();
}
//=================================================================================================//
} // namespace relax_dynamics
} // namespace SPH