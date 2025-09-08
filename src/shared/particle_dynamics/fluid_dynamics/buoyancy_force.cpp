#include "buoyancy_force.hpp"

namespace SPH
{
namespace fluid_dynamics
{
//=================================================================================================//
BuoyancyForce::BuoyancyForce(SPHBody &sph_body, const Real thermal_expansion_coeff, const Real phi_ref)
    : ForcePrior(sph_body, "BuoyancyForce"), 
      mass_(this->particles_->template getVariableDataByName<Real>("Mass")),
      gravity_(Vecd::Zero()), thermal_expansion_coeff_(thermal_expansion_coeff), phi_ref_(phi_ref),
      phi_(this->particles_->template getVariableDataByName<Real>("Phi"))
{
    gravity_[1] = -9.81;
}
//=================================================================================================//
void BuoyancyForce::update(size_t index_i, Real dt)
{
    current_force_[index_i] = -gravity_ * thermal_expansion_coeff_ * (phi_[index_i] - phi_ref_) * mass_[index_i];
    ForcePrior::update(index_i, dt);
}
//=================================================================================================//
PhiGradient<Contact<Wall>>::PhiGradient(BaseContactRelation &wall_contact_relation)
    : InteractionWithWall<PhiGradient>(wall_contact_relation)
{
    for (size_t k = 0; k != this->contact_particles_.size(); ++k)
    {
        wall_phi_.push_back(this->contact_particles_[k]->template getVariableDataByName<Real>("Phi"));
    }
}
//=================================================================================================//
void PhiGradient<Contact<Wall>>::interaction(size_t index_i, Real dt)
{
    Vecd phi_grad = Vecd::Zero();
    for (size_t k = 0; k < contact_configuration_.size(); ++k)
    {
        Real *phi_ave_k = wall_phi_[k];
        Real *Vol_k = wall_Vol_[k];
        Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
        for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
        {
            size_t index_j = contact_neighborhood.j_[n];
            const Vecd &e_ij = contact_neighborhood.e_ij_[n];
            Vecd nablaW_ijV_j = contact_neighborhood.dW_ij_[n] * Vol_k[index_j] * e_ij;
            phi_grad -= (phi_[index_i] - phi_ave_k[index_j]) * nablaW_ijV_j;
        }
    }

    phi_grad_[index_i] += phi_grad;
}
//=================================================================================================//
LocalNusseltNum::LocalNusseltNum(SPHBody& sph_body, Real nu_coeff)
    : LocalDynamics(sph_body), nu_coeff_(nu_coeff),
    spacing_ref_(sph_body_.getSPHAdaptation().ReferenceSpacing()),
    distance_from_wall_(particles_->getVariableDataByName<Vecd>("DistanceFromWall")),
    phi_grad_(particles_->getVariableDataByName<Vecd>("PhiGradient")),
    nu_num_(particles_->registerStateVariableData<Real>("LocalNusseltNumber")) {}
//=================================================================================================//
void LocalNusseltNum::update(size_t index_i, Real dt)
{
    const Vecd &distance_from_wall = distance_from_wall_[index_i];
    Real dist = distance_from_wall.norm();
    if (dist < 1.3 * spacing_ref_)
    {
        Vecd n_out = -(distance_from_wall / (dist + TinyReal));
        Real dTdn = n_out.dot(phi_grad_[index_i]);
        nu_num_[index_i] = -nu_coeff_ * dTdn;
    }
    else
    {
        nu_num_[index_i] = 0.0;
    }
}
//=================================================================================================//
} // namespace fluid_dynamics
} // namespace SPH
