#include "fluid_structure_interaction.hpp"

#include "viscosity.h"

namespace SPH
{
//=====================================================================================================//
namespace solid_dynamics
{
//=================================================================================================//
BaseForceFromFluid::BaseForceFromFluid(BaseContactRelation &contact_relation, const std::string &force_name)
    : ForcePrior(contact_relation.getSPHBody(), force_name), DataDelegateContact(contact_relation),
      solid_(DynamicCast<Solid>(this, sph_body_.getBaseMaterial())),
      Vol_(particles_->getVariableDataByName<Real>("VolumetricMeasure")),
      force_from_fluid_(particles_->getVariableDataByName<Vecd>(force_name))
{
    for (size_t k = 0; k != contact_particles_.size(); ++k)
    {
        contact_fluids_.push_back(DynamicCast<Fluid>(this, &contact_particles_[k]->getBaseMaterial()));
    }
}
//=================================================================================================//
ViscousForceFromFluid::ViscousForceFromFluid(BaseContactRelation &contact_relation)
    : BaseForceFromFluid(contact_relation, "ViscousForceFromFluid"),
      vel_ave_(solid_.AverageVelocity(particles_))
{
    for (size_t k = 0; k != contact_particles_.size(); ++k)
    {
        contact_vel_.push_back(contact_particles_[k]->getVariableDataByName<Vecd>("Velocity"));
        contact_Vol_.push_back(contact_particles_[k]->getVariableDataByName<Real>("VolumetricMeasure"));
        Viscosity &viscosity_k = DynamicCast<Viscosity>(this, contact_particles_[k]->getBaseMaterial());
        mu_.push_back(viscosity_k.ReferenceViscosity());
        smoothing_length_.push_back(contact_bodies_[k]->sph_adaptation_->ReferenceSmoothingLength());
    }
}
//=================================================================================================//
void ViscousForceFromFluid::interaction(size_t index_i, Real dt)
{
    Vecd force = Vecd::Zero();
    /** Contact interaction. */
    for (size_t k = 0; k < contact_configuration_.size(); ++k)
    {
        Real mu_k = mu_[k];
        Real smoothing_length_k = smoothing_length_[k];
        Vecd *vel_n_k = contact_vel_[k];
        Real *Vol_k = contact_Vol_[k];
        Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
        for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
        {
            size_t index_j = contact_neighborhood.j_[n];

            Vecd vel_derivative = 2.0 * (vel_ave_[index_i] - vel_n_k[index_j]) /
                                  (contact_neighborhood.r_ij_[n] + 0.01 * smoothing_length_k);
            force += 2.0 * mu_k * vel_derivative * contact_neighborhood.dW_ij_[n] * Vol_k[index_j];
        }
    }

    force_from_fluid_[index_i] = force * Vol_[index_i];
}
//=================================================================================================//
InitializeDisplacement::
    InitializeDisplacement(SPHBody &sph_body)
    : LocalDynamics(sph_body),
      pos_(particles_->getVariableDataByName<Vecd>("Position")),
      pos_temp_(particles_->registerStateVariable<Vecd>("TemporaryPosition")) {}
//=================================================================================================//
void InitializeDisplacement::update(size_t index_i, Real dt)
{
    pos_temp_[index_i] = pos_[index_i];
}
//=================================================================================================//
UpdateAverageVelocityAndAcceleration::
    UpdateAverageVelocityAndAcceleration(SPHBody &sph_body)
    : LocalDynamics(sph_body),
      pos_(particles_->getVariableDataByName<Vecd>("Position")),
      pos_temp_(particles_->getVariableDataByName<Vecd>("TemporaryPosition")),
      vel_ave_(particles_->getVariableDataByName<Vecd>("AverageVelocity")),
      acc_ave_(particles_->getVariableDataByName<Vecd>("AverageAcceleration")) {}
//=================================================================================================//
void UpdateAverageVelocityAndAcceleration::update(size_t index_i, Real dt)
{
    Vecd updated_vel_ave = (pos_[index_i] - pos_temp_[index_i]) / (dt + Eps);
    acc_ave_[index_i] = (updated_vel_ave - vel_ave_[index_i]) / (dt + Eps);
    vel_ave_[index_i] = updated_vel_ave;
}
//=================================================================================================//
AverageVelocityAndAcceleration::
    AverageVelocityAndAcceleration(SolidBody &solid_body)
    : initialize_displacement_(solid_body),
      update_averages_(solid_body) {}
//=================================================================================================//
SolidWSSFromFluid::
    SolidWSSFromFluid(BaseInnerRelation &inner_relation, BaseContactRelation &contact_relation)
    : LocalDynamics(inner_relation.getSPHBody()),
      DataDelegateInner(inner_relation), DataDelegateContact(contact_relation),
      solid_contact_indicator_(particles_->getVariableDataByName<int>("SolidTwoLayersIndicator")),
      Vol_(particles_->getVariableDataByName<Real>("VolumetricMeasure")),
      wall_shear_stress_(particles_->registerStateVariable<Matd>("SolidWallShearStress")),
      total_wall_shear_stress_(particles_->registerStateVariable<Matd>("SolidTotalWallShearStress")),
      WSS_magnitude_(particles_->registerStateVariable<Real>("WSSMagnitude"))
{
    for (size_t k = 0; k != contact_particles_.size(); ++k)
    {
        contact_Vol_.push_back(contact_particles_[k]->getVariableDataByName<Real>("VolumetricMeasure"));
        fluid_wall_shear_stress_.push_back(contact_particles_[k]->getVariableDataByName<Matd>("FluidWallShearStress"));
    }
}
//=================================================================================================//
void SolidWSSFromFluid::interaction(size_t index_i, Real dt)
{
    total_wall_shear_stress_[index_i] = Matd::Zero();
    Real ttl_weight(0);

    if (solid_contact_indicator_[index_i] == 1)
    {
        // interaction with first two layers of solid particles
        const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
        for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
        {
            size_t index_j = inner_neighborhood.j_[n];
            if (solid_contact_indicator_[index_j] == 1)
            {
                Real W_ij = inner_neighborhood.W_ij_[n];
                Real weight_j = W_ij * Vol_[index_j];
                ttl_weight += weight_j;
                total_wall_shear_stress_[index_i] += wall_shear_stress_[index_j] * weight_j;
            }
        }

        // interaction with fluid particles
        for (size_t k = 0; k < contact_configuration_.size(); ++k)
        {
            Real *Vol_k = contact_Vol_[k];
            Matd *fluid_WSS_k = fluid_wall_shear_stress_[k];
            Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
            for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
            {
                size_t index_j = contact_neighborhood.j_[n];
                Real W_ij = contact_neighborhood.W_ij_[n];
                Real weight_j = W_ij * Vol_k[index_j];
                ttl_weight += weight_j;
                total_wall_shear_stress_[index_i] += fluid_WSS_k[index_j] * weight_j;
            }
        }
    }

    wall_shear_stress_[index_i] = total_wall_shear_stress_[index_i] / (ttl_weight + TinyReal);
    WSS_magnitude_[index_i] = getWSSMagnitudeFromMatrix(wall_shear_stress_[index_i]);
}
//=================================================================================================//
//void SolidWSSFromFluid::update(size_t index_i, Real dt)
//{
//    Matd corrected_wall_shear_stress = Matd::Zero();
//    corrected_wall_shear_stress = total_wall_shear_stress_[index_i] / (weight_summation_[index_i] + TinyReal);
//    wall_shear_stress_[index_i] = corrected_wall_shear_stress;
//}
//=================================================================================================//
CorrectKernelWeightsSolidWSSFromFluid::
    CorrectKernelWeightsSolidWSSFromFluid(BaseInnerRelation &inner_relation, BaseContactRelation &contact_relation)
    : LocalDynamics(inner_relation.getSPHBody()),
      DataDelegateInner(inner_relation), DataDelegateContact(contact_relation),
      solid_contact_indicator_(particles_->getVariableDataByName<int>("SolidTwoLayersIndicator")),
      Vol_(particles_->getVariableDataByName<Real>("VolumetricMeasure")),
      wall_shear_stress_(particles_->registerStateVariable<Matd>("SolidWallShearStress")),
      total_wall_shear_stress_(particles_->registerStateVariable<Matd>("SolidTotalWallShearStress")),
      WSS_magnitude_(particles_->registerStateVariable<Real>("WSSMagnitude"))
{
    for (size_t k = 0; k != contact_particles_.size(); ++k)
    {
        contact_Vol_.push_back(contact_particles_[k]->getVariableDataByName<Real>("VolumetricMeasure"));
        fluid_wall_shear_stress_.push_back(contact_particles_[k]->getVariableDataByName<Matd>("FluidWallShearStress"));
    }
}
//=================================================================================================//
} // namespace solid_dynamics
} // namespace SPH
