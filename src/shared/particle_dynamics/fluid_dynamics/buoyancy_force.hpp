#pragma once

#include "buoyancy_force.h"

namespace SPH
{
namespace fluid_dynamics
{
//=================================================================================================//
template <class DataDelegationType>
template <class BaseRelationType>
PhiGradient<DataDelegationType>::PhiGradient(BaseRelationType &base_relation)
    : LocalDynamics(base_relation.getSPHBody()), DataDelegationType(base_relation),
      Vol_(this->particles_->template getVariableDataByName<Real>("VolumetricMeasure")),
      phi_(this->particles_->template getVariableDataByName<Real>("Phi")),
      phi_grad_(this->particles_->template registerStateVariableData<Vecd>("PhiGradient")) {}
//=================================================================================================//
template <class KernelCorrectionType>
PhiGradient<Inner<KernelCorrectionType>>::PhiGradient(BaseInnerRelation &inner_relation)
    : PhiGradient<DataDelegateInner>(inner_relation),
      kernel_correction_(particles_) {}
//=================================================================================================//
template <class KernelCorrectionType>
void PhiGradient<Inner<KernelCorrectionType>>::interaction(size_t index_i, Real dt)
{
    Vecd phi_grad = Vecd::Zero();
    Neighborhood &inner_neighborhood = inner_configuration_[index_i];
    for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
    {
        size_t index_j = inner_neighborhood.j_[n];
        Vecd nablaW_ijV_j = inner_neighborhood.dW_ij_[n] * Vol_[index_j] * inner_neighborhood.e_ij_[n];
        phi_grad -= (phi_[index_i] - phi_[index_j]) * nablaW_ijV_j;
    }

    phi_grad_[index_i] = phi_grad;
}
//=================================================================================================//
template <class KernelCorrectionType>
void PhiGradient<Inner<KernelCorrectionType>>::update(size_t index_i, Real dt)
{
    phi_grad_[index_i] = kernel_correction_(index_i) * phi_grad_[index_i];
}
//=================================================================================================//
} // namespace fluid_dynamics
} // namespace SPH
