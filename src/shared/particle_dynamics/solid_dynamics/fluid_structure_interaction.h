/* ------------------------------------------------------------------------- *
 *                                SPHinXsys                                  *
 * ------------------------------------------------------------------------- *
 * SPHinXsys (pronunciation: s'finksis) is an acronym from Smoothed Particle *
 * Hydrodynamics for industrial compleX systems. It provides C++ APIs for    *
 * physical accurate simulation and aims to model coupled industrial dynamic *
 * systems including fluid, solid, multi-body dynamics and beyond with SPH   *
 * (smoothed particle hydrodynamics), a meshless computational method using  *
 * particle discretization.                                                  *
 *                                                                           *
 * SPHinXsys is partially funded by German Research Foundation               *
 * (Deutsche Forschungsgemeinschaft) DFG HU1527/6-1, HU1527/10-1,            *
 *  HU1527/12-1 and HU1527/12-4.                                             *
 *                                                                           *
 * Portions copyright (c) 2017-2023 Technical University of Munich and       *
 * the authors' affiliations.                                                *
 *                                                                           *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may   *
 * not use this file except in compliance with the License. You may obtain a *
 * copy of the License at http://www.apache.org/licenses/LICENSE-2.0.        *
 *                                                                           *
 * ------------------------------------------------------------------------- */
/**
 * @file 	fluid_structure_interaction.h
 * @brief 	Here, we define the algorithm classes for fluid structure interaction.
 * @author	Chi Zhang and Xiangyu Hu
 */

#ifndef FLUID_STRUCTURE_INTERACTION_H
#define FLUID_STRUCTURE_INTERACTION_H

#include "all_particle_dynamics.h"
#include "base_material.h"
#include "elastic_dynamics.h"
#include "force_prior.hpp"
#include "riemann_solver.h"

namespace SPH
{
namespace solid_dynamics
{
/**
 * @class BaseForceFromFluid
 * @brief Base class for computing the forces from the fluid
 */
class BaseForceFromFluid : public ForcePrior, public DataDelegateContact
{
  public:
    explicit BaseForceFromFluid(BaseContactRelation &contact_relation, const std::string &force_name);
    virtual ~BaseForceFromFluid(){};
    Vecd *getForceFromFluid() { return force_from_fluid_; };

  protected:
    Solid &solid_;
    Real *Vol_;
    StdVec<Fluid *> contact_fluids_;
    Vecd *force_from_fluid_;
};

/**
 * @class ViscousForceFromFluid
 * @brief Computing the viscous force from the fluid
 */
class ViscousForceFromFluid : public BaseForceFromFluid
{
  public:
    explicit ViscousForceFromFluid(BaseContactRelation &contact_relation);
    virtual ~ViscousForceFromFluid(){};
    void interaction(size_t index_i, Real dt = 0.0);

  protected:
    Vecd *vel_ave_;
    StdVec<Real *> contact_Vol_;
    StdVec<Vecd *> contact_vel_;
    StdVec<Real> mu_;
    StdVec<Real> smoothing_length_;
};

/**
 * @class PressureForceFromFluid
 * @brief Template class fro computing the pressure force from the fluid with different Riemann solvers.
 * The pressure force is added on the viscous force of the latter is computed.
 * This class is for FSI applications to achieve smaller solid dynamics
 * time step size compared to the fluid dynamics
 */
template <class FluidIntegration2ndHalfType>
class PressureForceFromFluid : public BaseForceFromFluid
{
    using RiemannSolverType = typename FluidIntegration2ndHalfType::RiemannSolver;

  public:
    explicit PressureForceFromFluid(BaseContactRelation &contact_relation);
    virtual ~PressureForceFromFluid(){};
    void interaction(size_t index_i, Real dt = 0.0);

  protected:
    Vecd *vel_ave_, *acc_ave_, *n_;
    StdVec<Real *> contact_rho_, contact_mass_, contact_p_, contact_Vol_;
    StdVec<Vecd *> contact_vel_, contact_force_prior_;
    StdVec<RiemannSolverType> riemann_solvers_;
};

/**
 * @class InitializeDisplacement
 * @brief initialize the displacement for computing average velocity.
 * This class is for FSI applications to achieve smaller solid dynamics
 * time step size compared to the fluid dynamics
 */
class InitializeDisplacement : public LocalDynamics
{
  protected:
    Vecd *pos_, *pos_temp_;

  public:
    explicit InitializeDisplacement(SPHBody &sph_body);
    virtual ~InitializeDisplacement(){};

    void update(size_t index_i, Real dt = 0.0);
};

/**
 * @class UpdateAverageVelocityAndAcceleration
 * @brief Computing average velocity.
 * This class is for FSI applications to achieve smaller solid dynamics
 * time step size compared to the fluid dynamics
 */
class UpdateAverageVelocityAndAcceleration : public LocalDynamics
{
  protected:
    Vecd *pos_, *pos_temp_, *vel_ave_, *acc_ave_;

  public:
    explicit UpdateAverageVelocityAndAcceleration(SPHBody &sph_body);
    virtual ~UpdateAverageVelocityAndAcceleration(){};

    void update(size_t index_i, Real dt = 0.0);
};

/**
 * @class AverageVelocityAndAcceleration
 * @brief Impose force matching between fluid and solid dynamics.
 * Note that the fluid time step should be larger than that of solid time step.
 * Otherwise numerical instability may occur.
 */
class AverageVelocityAndAcceleration
{
  public:
    SimpleDynamics<InitializeDisplacement> initialize_displacement_;
    SimpleDynamics<UpdateAverageVelocityAndAcceleration> update_averages_;

    explicit AverageVelocityAndAcceleration(SolidBody &solid_body);
    ~AverageVelocityAndAcceleration(){};
};

class TwoLayersFromFluid : public LocalDynamics
{
  public:
    TwoLayersFromFluid(SPHBody &solid_body, SPHBody &fluid_body)
        : LocalDynamics(solid_body),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          solid_contact_indicator_(this->particles_->template registerStateVariable<int>("SolidTwoLayersIndicator")),
          fluid_body_(fluid_body),
          spacing_ref_(sph_body_.sph_adaptation_->ReferenceSpacing()){};
    virtual ~TwoLayersFromFluid(){};

    void update(size_t index_i, Real dt = 0.0)
    {
        Real phi = fluid_body_.getInitialShape().findSignedDistance(pos_[index_i]);
        solid_contact_indicator_[index_i] = 1;
        if (phi > 2.0 * 1.15 * spacing_ref_)
            solid_contact_indicator_[index_i] = 0;
    }

  protected:
    Vecd *pos_;
    int *solid_contact_indicator_;
    SPHBody &fluid_body_;
    Real spacing_ref_;
};

class SolidWSSFromFluid : public LocalDynamics, public DataDelegateInner, public DataDelegateContact
{
  public:
    explicit SolidWSSFromFluid(BaseInnerRelation &inner_relation, BaseContactRelation &contact_relation);
    virtual ~SolidWSSFromFluid(){};

    void interaction(size_t index_i, Real dt = 0.0);
    
    Real getWSSMagnitudeFromMatrix(const Mat2d &sigma)
    {
        Real sigmaxx = sigma(0, 0);
        Real sigmayy = sigma(1, 1);
        Real sigmaxy = sigma(0, 1);

        return sqrt(sigmaxx * sigmaxx + sigmayy * sigmayy - sigmaxx * sigmayy + 3.0 * sigmaxy * sigmaxy);
    }
    
    Real getWSSMagnitudeFromMatrix(const Mat3d &sigma)
    {
        Real sigmaxx = sigma(0, 0);
        Real sigmayy = sigma(1, 1);
        Real sigmazz = sigma(2, 2);
        Real sigmaxy = sigma(0, 1);
        Real sigmaxz = sigma(0, 2);
        Real sigmayz = sigma(1, 2);

        return sqrt(sigmaxx * sigmaxx + sigmayy * sigmayy + sigmazz * sigmazz -
                    sigmaxx * sigmayy - sigmaxx * sigmazz - sigmayy * sigmazz +
                    3.0 * (sigmaxy * sigmaxy + sigmaxz * sigmaxz + sigmayz * sigmayz));
    }
    //void update(size_t index_i, Real dt = 0.0);

  protected:
    int *solid_contact_indicator_;
    Real *Vol_;
    StdVec<Real *> contact_Vol_;
    Matd *wall_shear_stress_;
    StdVec<Matd *> fluid_wall_shear_stress_;
    Matd *total_wall_shear_stress_;
    Real *WSS_magnitude_;
};

class CorrectKernelWeightsSolidWSSFromFluid : public LocalDynamics, public DataDelegateInner, public DataDelegateContact
{
  public:
    explicit CorrectKernelWeightsSolidWSSFromFluid(BaseInnerRelation &inner_relation, BaseContactRelation &contact_relation);
    virtual ~CorrectKernelWeightsSolidWSSFromFluid(){};

    inline void interaction(size_t index_i, Real dt = 0.0)
    {
        total_wall_shear_stress_[index_i] = Matd::Zero();
        Real ttl_weight(0);

        if (solid_contact_indicator_[index_i] == 1)
        {
            Vecd weight_correction = Vecd::Zero();
            Matd local_configuration = Eps * Matd::Identity();

            for (size_t k = 0; k < contact_configuration_.size(); ++k)
            {
                Real *Vol_k = contact_Vol_[k];
                Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
                for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
                {
                    size_t index_j = contact_neighborhood.j_[n];
                    Real weight_j = contact_neighborhood.W_ij_[n] * Vol_k[index_j];
                    Vecd r_ji = -contact_neighborhood.r_ij_[n] * contact_neighborhood.e_ij_[n];
                    Vecd gradW_ijV_j = contact_neighborhood.dW_ij_[n] * Vol_k[index_j] * contact_neighborhood.e_ij_[n];

                    weight_correction += weight_j * r_ji;
                    local_configuration += r_ji * gradW_ijV_j.transpose();
                }
            }

            // correction matrix for interacting configuration
            Matd B = local_configuration.inverse();
            Vecd normalized_weight_correction = B * weight_correction;
            // Add the kernel weight correction to W_ij_ of neighboring particles.
            for (size_t k = 0; k < contact_configuration_.size(); ++k)
            {
                Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
                for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
                {
                    contact_neighborhood.W_ij_[n] -= normalized_weight_correction.dot(contact_neighborhood.e_ij_[n]) *
                                                     contact_neighborhood.dW_ij_[n];
                }
            }
            
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
    };

  protected:
    int *solid_contact_indicator_;
    Real *Vol_;
    StdVec<Real *> contact_Vol_;
    Matd *wall_shear_stress_;
    StdVec<Matd *> fluid_wall_shear_stress_;
    Matd *total_wall_shear_stress_;
};
} // namespace solid_dynamics
} // namespace SPH
#endif // FLUID_STRUCTURE_INTERACTION_H