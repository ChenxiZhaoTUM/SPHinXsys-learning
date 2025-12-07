#include "hemodynamic_indices.h"
#include "viscosity.h"

namespace SPH
{
//=====================================================================================================//    
namespace fluid_dynamics
{
//=================================================================================================//
TwoLayersFromWall::TwoLayersFromWall(BaseContactRelation &wall_contact_relation)
    : DistanceFromWall(wall_contact_relation),
      two_layers_indicatior_(this->particles_->template registerStateVariable<int>("TwoLayersIndicator")) {}
//=================================================================================================//
void TwoLayersFromWall::update(size_t index_i, Real dt)
{
    two_layers_indicatior_[index_i] = 0;

    Real squared_threshold = pow(2.0 * spacing_ref_, 2);
    if (distance_from_wall_[index_i].squaredNorm() <= squared_threshold)
    {
        two_layers_indicatior_[index_i] = 1;
    }
}
//=================================================================================================//
FluidWSS::FluidWSS(SPHBody &sph_body)
    : LocalDynamics(sph_body),
      two_layers_indicatior_(particles_->getVariableDataByName<int>("TwoLayersIndicator")),
      vel_grad_(particles_->getVariableDataByName<Matd>("VelocityGradient")),
      wall_shear_stress_(particles_->registerStateVariable<Matd>("FluidWallShearStress")),
      n_(particles_->getVariableDataByName<Vecd>("NormalDirection")) {}
//=================================================================================================//
void FluidWSS::update(size_t index_i, Real dt)
{
    wall_shear_stress_[index_i] = Matd::Zero();

    if (two_layers_indicatior_[index_i] == 1)
    {
        Matd D = 0.5 * (vel_grad_[index_i] + vel_grad_[index_i].transpose());

        Viscosity &viscosity = DynamicCast<Viscosity>(this, particles_->getBaseMaterial());
        Matd wss_total = 2 * viscosity.ReferenceViscosity() * D;

        Vecd wss_normal_vector = wss_total * n_[index_i];
        Vecd wss_normal = wss_normal_vector.dot(n_[index_i]) * n_[index_i]; 

        wall_shear_stress_[index_i] = wss_total - wss_normal * n_[index_i].transpose();
    }
}
//=================================================================================================//
FluidWSSTraction::FluidWSSTraction(SPHBody &sph_body)
    : LocalDynamics(sph_body),
      two_layers_indicatior_(particles_->getVariableDataByName<int>("TwoLayersIndicator")),
      vel_grad_(particles_->getVariableDataByName<Matd>("VelocityGradient")),
      wall_shear_stress_(particles_->registerStateVariable<Vecd>("FluidWallShearStressTraction")),
      n_(particles_->getVariableDataByName<Vecd>("NormalDirection")),
      p_(particles_->getVariableDataByName<Real>("Pressure")) {}
//=================================================================================================//
void FluidWSSTraction::update(size_t index_i, Real dt)
{
    wall_shear_stress_[index_i] = Vecd::Zero();

    if (two_layers_indicatior_[index_i] == 1)
    {
        Viscosity &viscosity = DynamicCast<Viscosity>(this, particles_->getBaseMaterial());
        //Matd sigma = p_[index_i] * Matd::Identity() + viscosity.ReferenceViscosity() * (vel_grad_[index_i] + vel_grad_[index_i].transpose());
        Matd sigma = viscosity.ReferenceViscosity() * (vel_grad_[index_i] + vel_grad_[index_i].transpose());

        Vecd traction = sigma * n_[index_i];
        Vecd wss_normal = traction.dot(n_[index_i]) * n_[index_i]; 

        wall_shear_stress_[index_i] = traction - wss_normal;
    }
}
//=================================================================================================//
} // namespace fluid_dynamics


namespace solid_dynamics
{
//=================================================================================================//
WallShearStress::WallShearStress(BaseContactRelation &contact_relation)
    : BaseForceFromFluid(contact_relation, "ViscousForceFromFluid"),
      vel_ave_(solid_.AverageVelocity(particles_)),
      spacing_ref_(sph_body_.sph_adaptation_->ReferenceSpacing()),
      n_(particles_->getVariableDataByName<Vecd>("NormalDirection")),
      wall_shear_stress_(particles_->registerStateVariable<Vecd>("WallShearStress"))
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
void WallShearStress::interaction(size_t index_i, Real dt)
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

    Real particle_area = pow(spacing_ref_, Dimensions - 1);
    wall_shear_stress_[index_i] = force_from_fluid_[index_i] / particle_area;

    /*Vecd normal_force = (force_from_fluid_[index_i].dot(n_[index_i])) * n_[index_i];
    Vecd tangential_force = force_from_fluid_[index_i] - normal_force;
    wall_shear_stress_[index_i] = tangential_force / particle_area;*/
}
//=================================================================================================//
HemodynamicIndiceCalculation::HemodynamicIndiceCalculation(SPHBody &sph_body, Real cardiac_cycle_time)
    : LocalDynamics(sph_body),
      cardiac_cycle_time_(cardiac_cycle_time),
      acc_time_(particles_->registerStateVariable<Real>("AccTime")),
      wall_shear_stress_(particles_->getVariableDataByName<Vecd>("WallShearStress")),
      acc_wall_shear_stress_(particles_->registerStateVariable<Real>("AccWallShearStress")),
      time_averaged_wall_shear_stress_(particles_->registerStateVariable<Real>("TimeAveragedWallShearStress")),
      acc_vector_wall_shear_stress_(particles_->registerStateVariable<Vecd>("AccVectorWallShearStress")),
      oscillatory_shear_index_(particles_->registerStateVariable<Real>("OscillatoryShearIndex"))
{}
//=================================================================================================//
void HemodynamicIndiceCalculation::update(size_t index_i, Real dt)
{
    acc_time_[index_i] += dt;
    acc_wall_shear_stress_[index_i] += wall_shear_stress_[index_i].norm() * dt;
    acc_vector_wall_shear_stress_[index_i] += wall_shear_stress_[index_i] * dt;
    
    if (acc_time_[index_i] >= cardiac_cycle_time_)
    {
        time_averaged_wall_shear_stress_[index_i] = acc_wall_shear_stress_[index_i] / cardiac_cycle_time_;
        oscillatory_shear_index_[index_i] = 0.5 * (1 - acc_vector_wall_shear_stress_[index_i].norm() / (acc_wall_shear_stress_[index_i] + TinyReal));

        if (acc_vector_wall_shear_stress_[index_i].norm() < 1.0E-10 && acc_wall_shear_stress_[index_i] < 1.0E-10)
            oscillatory_shear_index_[index_i] = 0;

        acc_wall_shear_stress_[index_i] = 0.0;
        acc_vector_wall_shear_stress_[index_i] = Vecd::Zero();
        acc_time_[index_i] = 0.0;
    }
}
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
SolidWSSFromFluidTraction::
    SolidWSSFromFluidTraction(BaseInnerRelation &inner_relation, BaseContactRelation &contact_relation)
    : LocalDynamics(inner_relation.getSPHBody()),
      DataDelegateInner(inner_relation), DataDelegateContact(contact_relation),
      solid_contact_indicator_(particles_->getVariableDataByName<int>("SolidTwoLayersIndicator")),
      Vol_(particles_->getVariableDataByName<Real>("VolumetricMeasure")),
      wall_shear_stress_(particles_->registerStateVariable<Vecd>("SolidWallShearStressTraction")),
      total_wall_shear_stress_(particles_->registerStateVariable<Vecd>("SolidTotalWallShearStressTraction")),
      WSS_magnitude_(particles_->registerStateVariable<Real>("WSSMagnitude"))
{
    for (size_t k = 0; k != contact_particles_.size(); ++k)
    {
        contact_Vol_.push_back(contact_particles_[k]->getVariableDataByName<Real>("VolumetricMeasure"));
        fluid_wall_shear_stress_.push_back(contact_particles_[k]->getVariableDataByName<Vecd>("FluidWallShearStressTraction"));
    }
}
//=================================================================================================//
void SolidWSSFromFluidTraction::interaction(size_t index_i, Real dt)
{
    total_wall_shear_stress_[index_i] = Vecd::Zero();
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
            Vecd *fluid_WSS_k = fluid_wall_shear_stress_[k];
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
    WSS_magnitude_[index_i] = wall_shear_stress_[index_i].norm();
}
//=================================================================================================//
SolidWSSFromFluidTangent::
    SolidWSSFromFluidTangent(BaseInnerRelation &inner_relation, BaseContactRelation &contact_relation)
    : LocalDynamics(inner_relation.getSPHBody()),
      DataDelegateInner(inner_relation), DataDelegateContact(contact_relation),
      solid_contact_indicator_(particles_->getVariableDataByName<int>("SolidTwoLayersIndicator")),
      Vol_(particles_->getVariableDataByName<Real>("VolumetricMeasure")),
      n_(particles_->getVariableDataByName<Vecd>("NormalDirection")),
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
void SolidWSSFromFluidTangent::interaction(size_t index_i, Real dt)
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

    Vecd wss_normal_vector = total_wall_shear_stress_[index_i] * n_[index_i];
    Vecd wss_normal = wss_normal_vector.dot(n_[index_i]) * n_[index_i]; 
    wall_shear_stress_[index_i] = (total_wall_shear_stress_[index_i] - wss_normal * n_[index_i].transpose()) / (ttl_weight + TinyReal);
    WSS_magnitude_[index_i] = getWSSMagnitudeFromMatrix(wall_shear_stress_[index_i]);

    //wall_shear_stress_[index_i] = total_wall_shear_stress_[index_i]/ (ttl_weight + TinyReal);
    //Vecd reference_t(1.0, 0.0, 0.0); // Tangential direction along x-axis
    //Vecd tangential_direction = reference_t - n_[index_i] * (n_[index_i].dot(reference_t));
    //if (tangential_direction.norm() < TinyReal)
    //{
    //    tangential_direction = Vecd(0.0, 1.0, 0.0);
    //}
    //else
    //{
    //    tangential_direction = tangential_direction.normalized();
    //}
    //Vecd tangential_traction = total_wall_shear_stress_[index_i] * tangential_direction;
    //WSS_magnitude_[index_i] = tangential_traction.norm();
}
//=================================================================================================//
} // namespace solid_dynamics
} // namespace SPH