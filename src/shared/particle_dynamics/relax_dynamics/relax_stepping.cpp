#include "relax_stepping.hpp"

namespace SPH
{
namespace relax_dynamics
{
//=================================================================================================//
RelaxationResidue<Inner<>>::RelaxationResidue(BaseInnerRelation &inner_relation)
    : RelaxationResidue<Base, DataDelegateInner>(inner_relation),
      relax_shape_(sph_body_.getInitialShape()){};
//=================================================================================================//
RelaxationResidue<Inner<>>::
    RelaxationResidue(BaseInnerRelation &inner_relation, const std::string &sub_shape_name)
    : RelaxationResidue<Base, DataDelegateInner>(inner_relation),
      relax_shape_(*DynamicCast<ComplexShape>(this, sph_body_.getInitialShape())
                        .getSubShapeByName(sub_shape_name)) {}
//=================================================================================================//
void RelaxationResidue<Inner<>>::interaction(size_t index_i, Real dt)
{
    Vecd residue = Vecd::Zero();
    const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
    for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
    {
        size_t index_j = inner_neighborhood.j_[n];
        residue -= 2.0 * inner_neighborhood.dW_ij_[n] * Vol_[index_j] * inner_neighborhood.e_ij_[n];
    }
    residue_[index_i] = residue;
};
//=================================================================================================//
void RelaxationResidue<Inner<LevelSetCorrection>>::interaction(size_t index_i, Real dt)
{
    RelaxationResidue<Inner<>>::interaction(index_i, dt);
    residue_[index_i] -= 2.0 * level_set_shape_.computeKernelGradientIntegral(
                                   pos_[index_i], sph_adaptation_->SmoothingLengthRatio(index_i));
}
//=================================================================================================//
void RelaxationResidue<Contact<>>::interaction(size_t index_i, Real dt)
{
    Vecd residue = Vecd::Zero();
    for (size_t k = 0; k < contact_configuration_.size(); ++k)
    {
        StdLargeVec<Real> &Vol_k = *(contact_Vol_[k]);
        Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
        for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
        {
            size_t index_j = contact_neighborhood.j_[n];
            residue -= 2.0 * contact_neighborhood.dW_ij_[n] * Vol_k[index_j] * contact_neighborhood.e_ij_[n];
        }
    }
    residue_[index_i] += residue;
}
//=================================================================================================//
RelaxationScaling::RelaxationScaling(SPHBody &sph_body)
    : LocalDynamicsReduce<ReduceMax>(sph_body),
      DataDelegateSimple(sph_body),
      residue_(*particles_->getVariableDataByName<Vecd>("ZeroOrderResidue")),
      h_ref_(sph_body.sph_adaptation_->ReferenceSmoothingLength()) {}
//=================================================================================================//
Real RelaxationScaling::reduce(size_t index_i, Real dt)
{
    return residue_[index_i].norm();
}
//=================================================================================================//
Real RelaxationScaling::outputResult(Real reduced_value)
{
    return 0.0625 * h_ref_ / (reduced_value + TinyReal);
}
//=================================================================================================//
PositionRelaxation::PositionRelaxation(SPHBody &sph_body)
    : LocalDynamics(sph_body), DataDelegateSimple(sph_body),
      sph_adaptation_(sph_body.sph_adaptation_),
      pos_(*particles_->getVariableDataByName<Vecd>("Position")),
      residue_(*particles_->getVariableDataByName<Vecd>("ZeroOrderResidue")) {}
//=================================================================================================//
void PositionRelaxation::update(size_t index_i, Real dt_square)
{
    pos_[index_i] += residue_[index_i] * dt_square * 0.5 / sph_adaptation_->SmoothingLengthRatio(index_i);
}
//=================================================================================================//
UpdateSmoothingLengthRatioByShape::
    UpdateSmoothingLengthRatioByShape(SPHBody &sph_body, Shape &target_shape)
    : LocalDynamics(sph_body), DataDelegateSimple(sph_body),
      h_ratio_(*particles_->getVariableDataByName<Real>("SmoothingLengthRatio")),
      Vol_(*particles_->getVariableDataByName<Real>("VolumetricMeasure")),
      pos_(*particles_->getVariableDataByName<Vecd>("Position")),
      target_shape_(target_shape),
      particle_adaptation_(DynamicCast<ParticleRefinementByShape>(this, sph_body.sph_adaptation_)),
      reference_spacing_(particle_adaptation_->ReferenceSpacing()) {}
//=================================================================================================//
UpdateSmoothingLengthRatioByShape::UpdateSmoothingLengthRatioByShape(SPHBody &sph_body)
    : UpdateSmoothingLengthRatioByShape(sph_body, sph_body.getInitialShape()) {}
//=================================================================================================//
void UpdateSmoothingLengthRatioByShape::update(size_t index_i, Real dt_square)
{
    Real local_spacing = particle_adaptation_->getLocalSpacing(target_shape_, pos_[index_i]);
    h_ratio_[index_i] = reference_spacing_ / local_spacing;
    Vol_[index_i] = pow(local_spacing, Dimensions);
}
//=================================================================================================//
ParticlesInAlignedBoxDetectionByCell::
    ParticlesInAlignedBoxDetectionByCell(BodyAlignedBoxByCell &aligned_box_part)
    : BaseLocalDynamics<BodyPartByCell>(aligned_box_part),
      DataDelegateSimple(aligned_box_part.getSPHBody()),
      pos_(*particles_->getVariableDataByName<Vecd>("Position")),
      aligned_box_(aligned_box_part.getAlignedBoxShape()) 
{
    //std::cout << "Particle num is " << aligned_box_part.getSPHBody().getBaseParticles().total_real_particles_ << std::endl;
}
//=================================================================================================//
void ParticlesInAlignedBoxDetectionByCell::update(size_t index_i, Real dt)
{
    // for debug
    /*if (index_i == 1014)
            std::cout << "index_i = 1014, " << "unsorted_id = " << particles_->unsorted_id_[index_i] << std::endl;*/

    mutex_switch_to_ghost_.lock();
    while (aligned_box_.checkInBounds(pos_[index_i]) && index_i < particles_->TotalRealParticles())
    {
        if(index_i == 26819)
            std::cout << "Find id=26819 in bounds!" << std::endl;

        if(index_i == 53007)
            std::cout << "Find id=53007 in bounds!" << std::endl;

        particles_->switchToBufferParticle(index_i);
    }
    mutex_switch_to_ghost_.unlock();
}
//=================================================================================================//
ParticlesInAlignedBoxDetectionByParticle::
    ParticlesInAlignedBoxDetectionByParticle(BodyAlignedBoxByParticle &aligned_box_part)
    : BaseLocalDynamics<BodyPartByParticle>(aligned_box_part),
      DataDelegateSimple(aligned_box_part.getSPHBody()),
      pos_(*particles_->getVariableDataByName<Vecd>("Position")),
      aligned_box_(aligned_box_part.getAlignedBoxShape()) {}
//=================================================================================================//
void ParticlesInAlignedBoxDetectionByParticle::update(size_t index_i, Real dt)
{
    mutex_switch_to_ghost_.lock();
    while (aligned_box_.checkInBounds(pos_[index_i]) && index_i < particles_->TotalRealParticles())
    {
        particles_->switchToBufferParticle(index_i);
    }
    mutex_switch_to_ghost_.unlock();
}
//=================================================================================================//
} // namespace relax_dynamics
} // namespace SPH
