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
} // namespace fluid_dynamics
} // namespace SPH
