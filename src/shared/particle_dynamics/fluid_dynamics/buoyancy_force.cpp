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
    //current_force_[index_i] = Vecd::Zero(); // pure heat conduction
    ForcePrior::update(index_i, dt);
}
//=================================================================================================//
PhiGradient<Contact<>>::PhiGradient(BaseContactRelation &contact_relation)
    : PhiGradient<DataDelegateContact>(contact_relation)
{
    for (size_t k = 0; k != this->contact_particles_.size(); ++k)
    {
        contact_Vol_.push_back(this->contact_particles_[k]->template getVariableDataByName<Real>("VolumetricMeasure"));
        contact_phi_.push_back(this->contact_particles_[k]->template getVariableDataByName<Real>("Phi"));
    }
}
//=================================================================================================//
void PhiGradient<Contact<>>::interaction(size_t index_i, Real dt)
{
    Vecd phi_grad = Vecd::Zero();
    for (size_t k = 0; k < contact_configuration_.size(); ++k)
    {
        Real *phi_ave_k = contact_phi_[k];
        Real *Vol_k = contact_Vol_[k];
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
PhiGradient<Contact<Wall>>::PhiGradient(BaseContactRelation &wall_contact_relation)
    : InteractionWithWall<PhiGradient>(wall_contact_relation),
    distance_from_wall_(particles_->getVariableDataByName<Vecd>("DistanceFromWall"))
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
    const Vecd &distance_from_wall = distance_from_wall_[index_i];
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

            Vecd distance_diff = distance_from_wall - contact_neighborhood.r_ij_[n] * e_ij;
            Real coeff = 1.0;
            Real factor = 1.0 - coeff * distance_from_wall.dot(distance_diff) / distance_from_wall.squaredNorm();

            phi_grad -= factor * (phi_[index_i] - phi_ave_k[index_j]) * nablaW_ijV_j;
        }
    }

    phi_grad_[index_i] += phi_grad;
}
//=================================================================================================//
LocalNusseltNum::LocalNusseltNum(SPHBody& sph_body, Real nu_coeff)
    : LocalDynamics(sph_body), nu_coeff_(nu_coeff),
    spacing_ref_(sph_body_.sph_adaptation_->ReferenceSpacing()),
    distance_from_wall_(particles_->getVariableDataByName<Vecd>("DistanceFromWall")),
    phi_grad_(particles_->getVariableDataByName<Vecd>("PhiGradient")),
    nu_num_(particles_->registerStateVariable<Real>("LocalNusseltNumber")) {}
//=================================================================================================//
void LocalNusseltNum::update(size_t index_i, Real dt)
{
    const Vecd &distance_from_wall = distance_from_wall_[index_i];
    Real dist = distance_from_wall.norm();
    //if (dist <= 2.01 * spacing_ref_)
    //{
    //    Vecd n_out = -(distance_from_wall / (dist + TinyReal));
    //    Real dTdn = n_out.dot(phi_grad_[index_i]);
    //    nu_num_[index_i] = -nu_coeff_ * dTdn;
    //}
    //else
    //{
    //    nu_num_[index_i] = 0.0;
    //}
    Vecd n_out = -(distance_from_wall / (dist + TinyReal));
    Real dTdn = n_out.dot(phi_grad_[index_i]);
    nu_num_[index_i] = -nu_coeff_ * dTdn;
}
//=================================================================================================//
TargetFluidParticles::TargetFluidParticles(BaseContactRelation &wall_contact_relation)
    : DistanceFromWall(wall_contact_relation),
      first_layer_indicatior_(particles_->registerStateVariable<int>("FirstLayerIndicator")),
      second_layer_indicatior_(particles_->registerStateVariable<int>("SecondLayerIndicator")) {}
//=================================================================================================//
void TargetFluidParticles::update(size_t index_i, Real dt)
{
    first_layer_indicatior_[index_i] = 0;
    second_layer_indicatior_[index_i] = 0;

    Real first_squared_threshold = pow(spacing_ref_, 2);
    Real second_squared_threshold = pow(2.0 * spacing_ref_, 2);
    if (distance_from_wall_[index_i].squaredNorm() <= first_squared_threshold)
    {
        first_layer_indicatior_[index_i] = 1;
    }
    else if (distance_from_wall_[index_i].squaredNorm() <= second_squared_threshold)
    {
        second_layer_indicatior_[index_i] = 1;
    }
}
//=================================================================================================//
FluidLocalVerticalHeatFlux::FluidLocalVerticalHeatFlux(BodyPartByCell &body_part, Real coeff, Real kappa)
    : BaseLocalDynamics<BodyPartByCell>(body_part), coeff_(coeff), kappa_(kappa),
    vel_(particles_->getVariableDataByName<Vecd>("Velocity")),
    phi_grad_(particles_->getVariableDataByName<Vecd>("PhiGradient")),
    phi_(this->particles_->template getVariableDataByName<Real>("Phi")),
    fluid_local_heat_flux_y_(particles_->registerStateVariable<Real>("FluidLocalVerticalHeatFlux")) {}
//=================================================================================================//
void FluidLocalVerticalHeatFlux::update(size_t index_i, Real dt)
{
    Real vT = vel_[index_i][1] * phi_[index_i];
    Real dTdy = phi_grad_[index_i][1];
    Real Qy = vT - kappa_ * dTdy;
    fluid_local_heat_flux_y_[index_i] = coeff_ * Qy;
}
//=================================================================================================//
} // namespace fluid_dynamics

namespace solid_dynamics
{
//=================================================================================================//
LocalNusseltNumWall::LocalNusseltNumWall(SPHBody& sph_body, Real nu_coeff)
    : LocalDynamics(sph_body), nu_coeff_(nu_coeff),
    n_(particles_->getVariableDataByName<Vecd>("NormalDirection")),
    phi_grad_(particles_->getVariableDataByName<Vecd>("PhiGradient")),
    nu_num_(particles_->registerStateVariable<Real>("WallLocalNusseltNumber")) {}
//=================================================================================================//
void LocalNusseltNumWall::update(size_t index_i, Real dt)
{
    Real dTdn = n_[index_i].dot(phi_grad_[index_i]);
    nu_num_[index_i] = -nu_coeff_ * dTdn;
}
//=================================================================================================//
LocalNusseltNumFromFluid::LocalNusseltNumFromFluid(BaseInnerRelation &inner_relation, BaseContactRelation &contact_relation, Real nu_coeff)
    : LocalDynamics(inner_relation.getSPHBody()), DataDelegateInner(inner_relation), DataDelegateContact(contact_relation),
    nu_coeff_(nu_coeff),
    n_(particles_->getVariableDataByName<Vecd>("NormalDirection")),
    phi_grad_(particles_->registerStateVariable<Vecd>("PhiGradient")),
    Vol_(this->particles_->template getVariableDataByName<Real>("VolumetricMeasure")),
    nu_num_(particles_->registerStateVariable<Real>("WallLocalNusseltNumber")),
    solid_contact_indicator_(particles_->getVariableDataByName<int>("SolidFirstLayerIndicator"))
{
    for (size_t k = 0; k != this->contact_particles_.size(); ++k)
    {
        BaseParticles *contact_particles_k = this->contact_particles_[k];
        contact_Vol_.push_back(this->contact_particles_[k]->template getVariableDataByName<Real>("VolumetricMeasure"));
        contact_phi_grad_.push_back(this->contact_particles_[k]->template getVariableDataByName<Vecd>("PhiGradient"));
    }
}
//=================================================================================================//
void LocalNusseltNumFromFluid::interaction(size_t index_i, Real dt)
{
    if (solid_contact_indicator_[index_i] != 1) 
    {
        phi_grad_[index_i] = Vecd::Zero();
        return;
    }

    Real sum_dTdn = 0.0;
    Real ttl_weight(0);
    if (solid_contact_indicator_[index_i] == 1)
    {
        // interaction with first two layers of solid particles
        //const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
        //for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
        //{
        //    size_t index_j = inner_neighborhood.j_[n];
        //    if (solid_contact_indicator_[index_j] == 1)
        //    {
        //        Real W_ij = inner_neighborhood.W_ij_[n];
        //        Real weight_j = W_ij * Vol_[index_j];
        //        ttl_weight += weight_j;
        //        total_phi_grad += phi_grad_[index_j] * weight_j;
        //    }
        //}

        // interaction with fluid particles
        for (size_t k = 0; k < contact_configuration_.size(); ++k)
        {
            Real *Vol_k = contact_Vol_[k];
            Vecd *fluid_phi_gradient = contact_phi_grad_[k];
            Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
            for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
            {
                size_t index_j = contact_neighborhood.j_[n];
                Real W_ij = contact_neighborhood.W_ij_[n];
                Real weight_j = W_ij * Vol_k[index_j];
                Real dTdn_j = n_[index_i].dot(fluid_phi_gradient[index_j]);
                sum_dTdn += dTdn_j * weight_j;
                ttl_weight += weight_j;
            }
        }
    }
    Real dTdn = sum_dTdn / (ttl_weight + TinyReal);
    phi_grad_[index_i] = n_[index_i] * dTdn; 
}
//=================================================================================================//
void LocalNusseltNumFromFluid::update(size_t index_i, Real dt)
{
    Real dTdn = n_[index_i].dot(phi_grad_[index_i]);
    nu_num_[index_i] = -nu_coeff_ * dTdn;
}
//=================================================================================================//
WallLocalNusseltNumberByFlux::WallLocalNusseltNumberByFlux(BaseContactRelation &contact_relation, Real nu_coeff)
    : LocalDynamics(contact_relation.getSPHBody()), DataDelegateContact(contact_relation),
    nu_coeff_(nu_coeff),
    nu_num_(particles_->registerStateVariable<Real>("WallLocalNusseltNumber")),
    wall_phi_(particles_->getVariableDataByName<Real>("Phi")),
    n_(particles_->getVariableDataByName<Vecd>("NormalDirection")),
    Vol_(particles_->getVariableDataByName<Real>("VolumetricMeasure")),
    spacing_ref_(sph_body_.sph_adaptation_->ReferenceSpacing()),
    solid_contact_indicator_(particles_->getVariableDataByName<int>("SolidFirstLayerIndicator"))
{
    for (size_t k = 0; k != this->contact_particles_.size(); ++k)
    {
        BaseParticles *contact_particles_k = this->contact_particles_[k];
        contact_Vol_.push_back(this->contact_particles_[k]->template getVariableDataByName<Real>("VolumetricMeasure"));
        contact_phi_.push_back(this->contact_particles_[k]->template getVariableDataByName<Real>("Phi"));
    }
}
//=================================================================================================//
//void WallLocalNusseltNumberByFlux::interaction(size_t index_i, Real dt)
//{
//    if (solid_contact_indicator_[index_i] != 1) 
//    {
//        nu_num_[index_i] = 0.0;
//        return;
//    }
//
//    Vecd g = Vecd::Zero();
//    Matd B = Matd::Zero();
//
//    // interaction with fluid particles
//    for (size_t k = 0; k < contact_configuration_.size(); ++k)
//    {
//        Real *Vol_k = contact_Vol_[k];
//        Real *fluid_phi = contact_phi_[k];
//        Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
//        for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
//        {
//            size_t index_j = contact_neighborhood.j_[n];
//            Real dW_ijV_j = contact_neighborhood.dW_ij_[n] * Vol_k[index_j];
//            Vecd e_ij = contact_neighborhood.e_ij_[n];
//            Vecd gradW_ijV_j = dW_ijV_j * e_ij;
//            g += (fluid_phi[index_j] - wall_phi_[index_i]) * gradW_ijV_j;
//            B += (contact_neighborhood.r_ij_[n] * e_ij) * gradW_ijV_j.transpose();
//        }
//    }
//
//    Vecd grad_phi = B.inverse() * g;
//    nu_num_[index_i] = -nu_coeff_ * grad_phi.dot(n_[index_i]);
//}
//
//void WallLocalNusseltNumberByFlux::interaction(size_t index_i, Real dt)
//{
//    if (solid_contact_indicator_[index_i] != 1) 
//    {
//        nu_num_[index_i] = 0.0;
//        return;
//    }
//
//    Matd B = Matd::Zero();
//
//    // interaction with fluid particles
//    for (size_t k = 0; k < contact_configuration_.size(); ++k)
//    {
//        Real *Vol_k = contact_Vol_[k];
//        Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
//        for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
//        {
//            size_t index_j = contact_neighborhood.j_[n];
//            Real dW_ijV_j = contact_neighborhood.dW_ij_[n] * Vol_k[index_j];
//            Vecd e_ij = contact_neighborhood.e_ij_[n];
//            Vecd gradW_ijV_j = dW_ijV_j * e_ij;
//            B += (contact_neighborhood.r_ij_[n] * e_ij) * gradW_ijV_j.transpose();
//        }
//    }
//
//    Real qn = 0.0;  
//
//    // interaction with fluid particles
//    for (size_t k = 0; k < contact_configuration_.size(); ++k)
//    {
//        Real *Vol_k = contact_Vol_[k];
//        Real *fluid_phi = contact_phi_[k];
//        Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
//        for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
//        {
//            size_t index_j = contact_neighborhood.j_[n];
//            Vecd e_ij = contact_neighborhood.e_ij_[n];
//            Real r_ij = contact_neighborhood.r_ij_[n];
//
//            Real rdotn  = r_ij * e_ij.dot(n_[index_i]);
//
//            Real chi  = e_ij.dot(B.inverse() * e_ij);
//            Real rGradW = - r_ij * contact_neighborhood.dW_ij_[n] * chi;
//
//            qn += (fluid_phi[index_j] - wall_phi_[index_i]) *  (rdotn * rGradW) / (r_ij*r_ij + TinyReal) * Vol_k[index_j];
//        }
//    }
//
//    nu_num_[index_i] = nu_coeff_ * qn;
//}
//
//void WallLocalNusseltNumberByFlux::interaction(size_t index_i, Real dt)
//{
//    if (solid_contact_indicator_[index_i] != 1) 
//    {
//        nu_num_[index_i] = 0.0;
//        return;
//    }
//
//    Vecd n_w = n_[index_i] / (n_[index_i].norm() + TinyReal);
//    Real numer  = 0.0;  
//    Real denom = 0.0;
//    // interaction with fluid particles
//    for (size_t k = 0; k < contact_configuration_.size(); ++k)
//    {
//        Real *Vol_k = contact_Vol_[k];
//        Real *fluid_phi = contact_phi_[k];
//        Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
//        for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
//        {
//            size_t index_j = contact_neighborhood.j_[n];
//            Vecd e_ij = contact_neighborhood.e_ij_[n];
//            Real r_ij = contact_neighborhood.r_ij_[n];
//
//            Real rdotn  = - r_ij * e_ij.dot(n_w);
//            Real rGradW = - r_ij * contact_neighborhood.dW_ij_[n];
//
//            numer += (-contact_neighborhood.dW_ij_[n]) * Vol_k[index_j] * (fluid_phi[index_j] - wall_phi_[index_i]) * rdotn;
//            denom +=  (-contact_neighborhood.dW_ij_[n]) * Vol_k[index_j] * rdotn * rdotn;
//        }
//    }
//    Real grad_phi = numer / denom;
//
//    nu_num_[index_i] = nu_coeff_ * grad_phi;
//}
//
void WallLocalNusseltNumberByFlux::interaction(size_t index_i, Real dt)
{
    if (solid_contact_indicator_[index_i] != 1) 
    {
        nu_num_[index_i] = 0.0;
        return;
    }

    Matd B = Matd::Zero();

    //interaction with fluid particles
    for (size_t k = 0; k < contact_configuration_.size(); ++k)
    {
        Real *Vol_k = contact_Vol_[k];
        Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
        for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
        {
            size_t index_j = contact_neighborhood.j_[n];
            Real dW_ijV_j = contact_neighborhood.dW_ij_[n] * Vol_k[index_j];
            Vecd e_ij = contact_neighborhood.e_ij_[n];
            Vecd gradW_ijV_j = dW_ijV_j * e_ij;
            B += (contact_neighborhood.r_ij_[n] * e_ij) * gradW_ijV_j.transpose();
        }
    }

    Real sa = 0.0;
    //interaction with fluid particles
    for (size_t k = 0; k < contact_configuration_.size(); ++k)
    {
        Real *Vol_k = contact_Vol_[k];
        Real *fluid_phi = contact_phi_[k];
        Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
        for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
        {
            size_t index_j = contact_neighborhood.j_[n];
            Vecd e_ij = contact_neighborhood.e_ij_[n];
            Real r_ij = contact_neighborhood.r_ij_[n];

            Real rdotn  = r_ij * e_ij.dot(n_[index_i]);

            Real chi  = e_ij.dot(B.inverse() * e_ij);
            Real rGradW = r_ij * contact_neighborhood.dW_ij_[n] * chi;

            sa += (fluid_phi[index_j] - wall_phi_[index_i]) *  rGradW / (r_ij*r_ij + TinyReal) * Vol_k[index_j];
        }
    }
    Real grad_phi = Vol_[index_i] * sa / spacing_ref_;
    nu_num_[index_i] = nu_coeff_ * grad_phi;
}

//void WallLocalNusseltNumberByFlux::interaction(size_t index_i, Real dt)
//{
//    if (solid_contact_indicator_[index_i] != 1)
//    {
//        nu_num_[index_i] = 0.0;
//        return;
//    }
//
//    const Real T_w = wall_phi_[index_i];
//    const Vecd n_w = n_[index_i] / (n_[index_i].norm() + TinyReal); // 单位外法向
//
//    Matd M = Matd::Zero();
//    Vecd r = Vecd::Zero();
//
//    for (size_t k = 0; k < contact_configuration_.size(); ++k)
//    {
//        Real *Vol_k = contact_Vol_[k];
//        Real *T_f = contact_phi_[k];
//        Neighborhood &contact_nb = (*contact_configuration_[k])[index_i];
//
//        for (size_t n = 0; n < contact_nb.current_size_; ++n)
//        {
//            const size_t index_j = contact_nb.j_[n];
//            const Vecd dx = contact_nb.r_ij_[n] * contact_nb.e_ij_[n];
//
//            const Real wj = contact_nb.dW_ij_[n] * Vol_k[index_j]; // 权重
//
//            M += wj * dx * dx.transpose();                         // M项
//            r += wj * (T_f[index_j] - T_w) * dx;                   // r项
//        }
//    }
//
//    // 正则化防止 M 奇异
//    const Real epsilon = TinyReal * M.trace();
//    for (int d = 0; d != Dimensions; ++d) M(d,d) += epsilon;
//
//    const Vecd grad_T = M.inverse() * r;
//    const Real grad_Tn = grad_T.dot(n_w);                         // 法向梯度
//
//    nu_num_[index_i] = -nu_coeff_ * grad_Tn;
//}

//=================================================================================================//
FFDForNu::FFDForNu(BaseContactRelation &contact_relation, Real nu_coeff)
    : LocalDynamics(contact_relation.getSPHBody()), DataDelegateContact(contact_relation),
    nu_coeff_(nu_coeff),
    n_(particles_->getVariableDataByName<Vecd>("NormalDirection")),
    nu_num_(particles_->registerStateVariable<Real>("WallLocalNusseltNumber")),
    wall_phi_(particles_->getVariableDataByName<Real>("Phi")),
    spacing_ref_(sph_body_.sph_adaptation_->ReferenceSpacing()),
    solid_contact_indicator_(particles_->getVariableDataByName<int>("SolidFirstLayerIndicator"))
{
    for (size_t k = 0; k != this->contact_particles_.size(); ++k)
    {
        BaseParticles *contact_particles_k = this->contact_particles_[k];
        contact_phi_.push_back(this->contact_particles_[k]->template getVariableDataByName<Real>("Phi"));
        contact_first_layer_indicator_.push_back(this->contact_particles_[k]->template getVariableDataByName<int>("FirstLayerIndicator"));
        contact_second_layer_indicator_.push_back(this->contact_particles_[k]->template getVariableDataByName<int>("SecondLayerIndicator"));
    }
}
//=================================================================================================//
void FFDForNu::interaction(size_t index_i, Real dt)
{
    if (!solid_contact_indicator_[index_i])
    {
        nu_num_[index_i] = Real(0);
        return;
    }

    const Real Tw = wall_phi_[index_i];

    Real min_distance_1 = MaxReal;
    Real min_distance_2 = MaxReal;
    Real d1(0), d2(0);
    Real T1(0), T2(0);

    for (size_t k = 0; k < contact_configuration_.size(); ++k)
    {
        Real* fluid_phi = contact_phi_[k];
        int* fluid_first_indicator = contact_first_layer_indicator_[k];
        int* fluid_second_indicator = contact_second_layer_indicator_[k];
        Neighborhood& contact_neighborhood = (*contact_configuration_[k])[index_i];


        for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
        {
            size_t index_j = contact_neighborhood.j_[n];
            Real r_ij = contact_neighborhood.r_ij_[n];
            Vecd e_ij = contact_neighborhood.e_ij_[n];
            
            if (fluid_first_indicator[index_j])
            {
                if (r_ij < min_distance_1)
                {
                    min_distance_1 = r_ij;
                    d1 = -(r_ij * e_ij).dot(n_[index_i]);
                    T1 = fluid_phi[index_j];
                }
            }

            if (fluid_second_indicator[index_j])
            {
                if (r_ij < min_distance_2)
                {
                    min_distance_2 = r_ij;
                    d2 = -(r_ij * e_ij).dot(n_[index_i]);
                    T2 = fluid_phi[index_j];
                }
            }
        }
    }
    Real phi_grad = 0;

    if (d1 < std::numeric_limits<Real>::max() && min_distance_1 < 1.3 * spacing_ref_)
    {
        if (d2 < std::numeric_limits<Real>::max() && std::abs(d2 - d1) > TinyReal)
        {
            phi_grad = -(2 * d1 + d2) / (d1 * (d1 + d2) + TinyReal) * Tw + (d1 + d2) / (d1 * d2 + TinyReal) * T1 - d1 / (d2 * (d1 + d2) + TinyReal) * T2;

            //phi_grad = (- 3 * Tw + 4 * T1 - T2) / (2 * spacing_ref_);
        }
        else
        {
            phi_grad = (T1 - Tw) / (d1 + TinyReal);
        }

        nu_num_[index_i] = phi_grad * nu_coeff_;
    } 
    else
    {
        nu_num_[index_i] = Real(0);
    }
}
//=================================================================================================//
ProjectionForNu::ProjectionForNu(BaseContactRelation &contact_relation, Real nu_coeff)
    : LocalDynamics(contact_relation.getSPHBody()), DataDelegateContact(contact_relation),
    nu_coeff_(nu_coeff),
    nu_num_(particles_->registerStateVariable<Real>("WallLocalNusseltNumber")),
    solid_contact_indicator_(particles_->getVariableDataByName<int>("SolidFirstLayerIndicator"))
{
    for (size_t k = 0; k != this->contact_particles_.size(); ++k)
    {
        BaseParticles *contact_particles_k = this->contact_particles_[k];
        contact_nu_.push_back(this->contact_particles_[k]->template getVariableDataByName<Real>("LocalNusseltNumber"));
        contact_Vol_.push_back(this->contact_particles_[k]->template getVariableDataByName<Real>("VolumetricMeasure"));
        contact_first_layer_indicator_.push_back(this->contact_particles_[k]->template getVariableDataByName<int>("FirstLayerIndicator"));
        contact_second_layer_indicator_.push_back(this->contact_particles_[k]->template getVariableDataByName<int>("SecondLayerIndicator"));
    }
}
//=================================================================================================//
void ProjectionForNu::interaction(size_t index_i, Real dt)
{
    if (solid_contact_indicator_[index_i] != 1) 
    {
        nu_num_[index_i] = 0.0;
        return;
    }

    Real sum = 0.0;
    Real weight = 0.0;
    //interaction with fluid particles
    for (size_t k = 0; k < contact_configuration_.size(); ++k)
    {
        Real *Vol_k = contact_Vol_[k];
        Real *fluid_nu = contact_nu_[k];
        int *fluid_indicator_1st = contact_second_layer_indicator_[k];
        int *fluid_indicator_2nd = contact_second_layer_indicator_[k];
        Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
        for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
        {
            size_t index_j = contact_neighborhood.j_[n];
            int layer_indicator_1st = fluid_indicator_1st[index_j];
            int layer_indicator_2nd = fluid_indicator_2nd[index_j];
            if (layer_indicator_1st || layer_indicator_2nd)
            {
                sum += fluid_nu[index_j] * contact_neighborhood.W_ij_[n] * Vol_k[index_j];
                weight += contact_neighborhood.W_ij_[n] * Vol_k[index_j];
            }
            //sum += fluid_nu[index_j] * contact_neighborhood.W_ij_[n] * Vol_k[index_j];
            //weight += contact_neighborhood.W_ij_[n] * Vol_k[index_j];
        }
    }
    nu_num_[index_i] = (weight > TinyReal) ? sum / weight : 0.0;
}
//=================================================================================================//
} // namespace solid_dynamics
} // namespace SPH
