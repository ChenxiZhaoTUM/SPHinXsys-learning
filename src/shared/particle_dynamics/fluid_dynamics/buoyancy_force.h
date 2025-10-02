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
 * Portions copyright (c) 2017-2025 Technical University of Munich and       *
 * the authors' affiliations.                                                *
 *                                                                           *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may   *
 * not use this file except in compliance with the License. You may obtain a *
 * copy of the License at http://www.apache.org/licenses/LICENSE-2.0.        *
 *                                                                           *
 * ------------------------------------------------------------------------- */
/**
 * @file 	buoyancy_force.h
 * @brief   
 * @details  
 * @author	 
 */

#ifndef BUOYANCY_FORCE_H
#define BUOYANCY_FORCE_H

#include "base_fluid_dynamics.h"
#include "force_prior.h"
#include "diffusion_dynamics.h"
#include "near_wall_boundary.h"

namespace SPH
{
namespace fluid_dynamics
{
class BuoyancyForce : public ForcePrior
{
  protected:
    Real* mass_;
    Vecd gravity_;
    Real thermal_expansion_coeff_, phi_ref_;
    Real* phi_;
    
  public:
    BuoyancyForce(SPHBody &sph_body, const Real thermal_expansion_coeff, const Real phi_ref);
    virtual ~BuoyancyForce() {};
    void update(size_t index_i, Real dt = 0.0);
};

template <typename... InteractionTypes>
class PhiGradient;

template <class DataDelegationType>
class PhiGradient<DataDelegationType>
    : public LocalDynamics, public DataDelegationType
{
  public:
    template <class BaseRelationType>
    explicit PhiGradient(BaseRelationType &base_relation);
    virtual ~PhiGradient(){};

  protected:
    Real *Vol_;
    Real *phi_;
    Vecd *phi_grad_;
};

template <class KernelCorrectionType>
class PhiGradient<Inner<KernelCorrectionType>>
    : public PhiGradient<DataDelegateInner>
{
  public:
    explicit PhiGradient(BaseInnerRelation &inner_relation);
    virtual ~PhiGradient(){};
    void interaction(size_t index_i, Real dt = 0.0);
    void update(size_t index_i, Real dt = 0.0);

  protected:
    KernelCorrectionType kernel_correction_;
};
using PhiGradientInner = PhiGradient<Inner<NoKernelCorrection>>;

template <>
class PhiGradient<Contact<>> : public PhiGradient<DataDelegateContact>
{
  public:
    explicit PhiGradient(BaseContactRelation &contact_relation);
    virtual ~PhiGradient(){};
    void interaction(size_t index_i, Real dt = 0.0);

  protected:
    StdVec<Real *> contact_Vol_, contact_phi_;
};

template <class KernelCorrectionType>
using PhiGradientComplex = ComplexInteraction<PhiGradient<Inner<KernelCorrectionType>, Contact<>>>;

/**
 * @class LocalNusseltNum
 * @brief  compute Nusselt number in the fluid field
 */
class LocalNusseltNum : public LocalDynamics
{
  public:
    explicit LocalNusseltNum(SPHBody &sph_body, Real nu_coeff);
    virtual ~LocalNusseltNum() {};

    void update(size_t index_i, Real dt = 0.0);

  protected:
    Real nu_coeff_, spacing_ref_;
    Vecd *distance_from_wall_;
    Vecd *phi_grad_;
    Real *nu_num_;
};

class TargetFluidParticles : public DistanceFromWall
{
  public:
    explicit TargetFluidParticles(BaseContactRelation &wall_contact_relation);
    virtual ~TargetFluidParticles(){};
    void update(size_t index_i, Real dt = 0.0);

  protected:
    int *first_layer_indicatior_;
    int *second_layer_indicatior_;
};

} // namespace fluid_dynamics

namespace solid_dynamics
{
template <typename... KernelGradientType>
class PhiGradientFromFluid;

template <class ContactKernelGradientType>
class PhiGradientFromFluid<ContactKernelGradientType> : public LocalDynamics, public DataDelegateContact
{
  public:
    explicit PhiGradientFromFluid(BaseContactRelation &contact_relation);
    virtual ~PhiGradientFromFluid() {};
    void interaction(size_t index_i, Real dt = 0.0);

  protected:
    StdVec<ContactKernelGradientType> contact_kernel_gradients_;
    Real *phi_;
    Vecd *phi_grad_;
    StdVec<Real *> contact_Vol_, contact_phi_;
};

class LocalNusseltNumWall : public LocalDynamics
{
  public:
    explicit LocalNusseltNumWall(SPHBody &sph_body, Real nu_coeff);
    virtual ~LocalNusseltNumWall() {};

    void update(size_t index_i, Real dt = 0.0);

  protected:
    Real nu_coeff_;
    Vecd *n_, *phi_grad_;
    Real *nu_num_;
};

class FirstLayerFromFluid : public LocalDynamics
{
  public:
    FirstLayerFromFluid(SPHBody &solid_body, SPHBody &fluid_body)
        : LocalDynamics(solid_body),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          solid_contact_indicator_(particles_->registerStateVariableData<int>("SolidFirstLayerIndicator")),
          fluid_body_(fluid_body),
          spacing_ref_(sph_body_.getSPHAdaptation().ReferenceSpacing()){};
    virtual ~FirstLayerFromFluid(){};

    void update(size_t index_i, Real dt = 0.0)
    {
        Real phi = fluid_body_.getInitialShape().findSignedDistance(pos_[index_i]);
        solid_contact_indicator_[index_i] = 1;
        if (phi > 1.01 * spacing_ref_)
            solid_contact_indicator_[index_i] = 0;
    }

  protected:
    Vecd *pos_;
    int *solid_contact_indicator_;
    SPHBody &fluid_body_;
    Real spacing_ref_;
};

class LocalNusseltNumFromFluid : public LocalDynamics, public DataDelegateInner, public DataDelegateContact
{
  public:
    explicit LocalNusseltNumFromFluid(BaseInnerRelation &inner_relation, BaseContactRelation &contact_relation, Real nu_coeff);
    virtual ~LocalNusseltNumFromFluid() {};

    void interaction(size_t index_i, Real dt = 0.0);
    void update(size_t index_i, Real dt = 0.0);

  protected:
    Real nu_coeff_;
    Vecd *n_, *phi_grad_;
    Real *Vol_, *nu_num_;
    int *solid_contact_indicator_;
    StdVec<Real *> contact_Vol_;
    StdVec<Vecd *> contact_phi_grad_;
};

class FFDForNu : public LocalDynamics, public DataDelegateContact
{
  public:
    explicit FFDForNu(BaseContactRelation &contact_relation, Real nu_coeff);
    virtual ~FFDForNu() {};

    void interaction(size_t index_i, Real dt = 0.0);

  protected:
    Real nu_coeff_;
    Vecd *n_;
    Real *nu_num_, *wall_phi_;
    Real spacing_ref_;
    int *solid_contact_indicator_;
    StdVec<Real *> contact_phi_;
    StdVec<int *> contact_first_layer_indicator_, contact_second_layer_indicator_;
};

} // namespace solid_dynamics
} // namespace SPH
#endif // BUOYANCY_FORCE_H