#include "arbitrary_shape_buffer.h"

namespace SPH
{
namespace fluid_dynamics
{
//=================================================================================================//
template <typename AlignedShapeType>
EmitterInflowInjectionArb<AlignedShapeType>::
    EmitterInflowInjectionArb(BaseAlignedRegion<BodyRegionByParticle, AlignedShapeType> &aligned_region_part, ParticleBuffer<Base> &buffer)
    : BaseLocalDynamics<BodyPartByParticle>(aligned_region_part),
      DataDelegateSimple(aligned_region_part.getSPHBody()),
      fluid_(DynamicCast<Fluid>(this, particles_->getBaseMaterial())),
      original_id_(particles_->ParticleOriginalIds()),
      sorted_id_(particles_->ParticleSortedIds()),
      pos_(*particles_->getVariableDataByName<Vecd>("Position")),
      rho_(*particles_->getVariableDataByName<Real>("Density")),
      p_(*particles_->getVariableDataByName<Real>("Pressure")),
      buffer_(buffer), aligned_shape_(aligned_region_part.getAlignedShape())
{
    buffer_.checkParticlesReserved();
}
//=================================================================================================//
template <typename AlignedShapeType>
void EmitterInflowInjectionArb<AlignedShapeType>::update(size_t original_index_i, Real dt)
{
    size_t sorted_index_i = sorted_id_[original_index_i];
    if (aligned_shape_.checkUpperBound(pos_[sorted_index_i]))
    {
        mutex_switch_to_real_.lock();
        buffer_.checkEnoughBuffer(*particles_);
        particles_->createRealParticleFrom(sorted_index_i);
        mutex_switch_to_real_.unlock();

        /** Periodic bounding. */
        pos_[sorted_index_i] = aligned_shape_.getUpperPeriodic(pos_[sorted_index_i]);
        rho_[sorted_index_i] = fluid_.ReferenceDensity();
        p_[sorted_index_i] = fluid_.getPressure(rho_[sorted_index_i]);
    }
}
//=================================================================================================//
template <typename AlignedShapeType>
DisposerOutflowDeletionArb<AlignedShapeType>::
    DisposerOutflowDeletionArb(BaseAlignedRegion<BodyRegionByCell, AlignedShapeType>& aligned_region_part)
    : BaseLocalDynamics<BodyPartByCell>(aligned_region_part),
      DataDelegateSimple(aligned_region_part.getSPHBody()),
      pos_(*particles_->getVariableDataByName<Vecd>("Position")),
      aligned_shape_(aligned_region_part.getAlignedShape()) {}
//=================================================================================================//
template <typename AlignedShapeType>
void DisposerOutflowDeletionArb<AlignedShapeType>::update(size_t index_i, Real dt)
{
    mutex_switch_to_buffer_.lock();
    while (aligned_shape_.checkUpperBound(pos_[index_i]) && index_i < particles_->TotalRealParticles())
    {
        particles_->switchToBufferParticle(index_i);
    }
    mutex_switch_to_buffer_.unlock();
}
//=================================================================================================//
template <typename AlignedShapeType>
DisposerOutflowDeletionWithWindkesselArb<AlignedShapeType>::
    DisposerOutflowDeletionWithWindkesselArb(BaseAlignedRegion<BodyRegionByCell, AlignedShapeType>& aligned_region_part, const std::string& body_part_name)
    : DisposerOutflowDeletionArb<AlignedShapeType>(aligned_region_part), 
      flow_rate_(*this->particles_->registerSingleVariable<Real>(body_part_name+"FlowRate")), 
      Vol_(*particles_->getVariableDataByName<Real>("VolumetricMeasure")) {}
//=================================================================================================//
template <typename AlignedShapeType>
void DisposerOutflowDeletionWithWindkesselArb<AlignedShapeType>::update(size_t index_i, Real dt)
{
    mutex_switch_to_buffer_.lock();
    while (aligned_shape_.checkUpperBound(pos_[index_i]) && index_i < particles_->TotalRealParticles())
    {
        particles_->switchToBufferParticle(index_i);
        flow_rate_ += Vol_[index_i];
    }
    mutex_switch_to_buffer_.unlock();
}
} // namespace fluid_dynamics
} // namespace SPH