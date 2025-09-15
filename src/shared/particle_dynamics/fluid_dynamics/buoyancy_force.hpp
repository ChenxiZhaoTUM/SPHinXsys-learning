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
      phi_grad_(this->particles_->template registerStateVariable<Vecd>("PhiGradient")) {}
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

namespace solid_dynamics
{
//=================================================================================================//
template <class ContactKernelGradientType>
PhiGradientFromFluid<ContactKernelGradientType>::PhiGradientFromFluid(BaseContactRelation &contact_relation)
    : LocalDynamics(contact_relation.getSPHBody()), DataDelegateContact(contact_relation),
      phi_(particles_->getVariableDataByName<Real>("Phi")),
      phi_grad_(particles_->registerStateVariable<Vecd>("PhiGradient"))
{
    for (size_t k = 0; k != this->contact_particles_.size(); ++k)
    {
        BaseParticles *contact_particles_k = this->contact_particles_[k];
        contact_kernel_gradients_.push_back(ContactKernelGradientType(this->particles_, contact_particles_k));
        contact_phi_.push_back(this->contact_particles_[k]->template getVariableDataByName<Real>("Phi"));
        contact_Vol_.push_back(this->contact_particles_[k]->template getVariableDataByName<Real>("VolumetricMeasure"));
    }
}
//=================================================================================================//
template <class ContactKernelGradientType>
void PhiGradientFromFluid<ContactKernelGradientType>::interaction(size_t index_i, Real dt)
{
    Vecd phi_grad = Vecd::Zero();
    for (size_t k = 0; k < contact_configuration_.size(); ++k)
    {
        Real *phi_fluid = contact_phi_[k];
        Real *Vol_k = contact_Vol_[k];
        Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
        for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
        {
            size_t index_j = contact_neighborhood.j_[n];
            const Vecd &e_ij = contact_neighborhood.e_ij_[n];
            Real dW_ijV_j = contact_neighborhood.dW_ij_[n] * Vol_k[index_j];
            const Vecd &grad_ijV_j = this->contact_kernel_gradients_[k](index_i, index_j, dW_ijV_j, e_ij);
            phi_grad -= (phi_[index_i] - phi_fluid[index_j]) * grad_ijV_j;
        }
    }

    phi_grad_[index_i] = phi_grad;
}
//=================================================================================================//
} // namespace solid_dynamics
} // namespace SPH
