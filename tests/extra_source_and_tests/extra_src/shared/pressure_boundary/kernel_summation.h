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
 * @file 	kernel_summation.h
 * @brief   Here, according to the zeroth order consistency, we calculate the
            kernel summation for imposing the pressure boundary condition.
 * @author	Shuoguo Zhang and Xiangyu Hu
 */

#ifndef KERNEL_SUMMATION_H
#define KERNEL_SUMMATION_H

#include "base_general_dynamics.h"

namespace SPH
{

class LevelSetShape;
class LevelSetCorrection;

template <typename... InteractionTypes>
class NablaWV;

template <class DataDelegationType>
class NablaWV<DataDelegationType>
    : public LocalDynamics, public DataDelegationType
{
  public:
    template <class BaseRelationType>
    explicit NablaWV(BaseRelationType &base_relation);
    virtual ~NablaWV(){};

  protected:
    StdLargeVec<Vecd> &kernel_sum_;
};

template <>
class NablaWV<Inner<>>
    : public NablaWV<DataDelegateInner>
{
  public:
    explicit NablaWV(BaseInnerRelation &inner_relation);
    virtual ~NablaWV(){};
    void interaction(size_t index_i, Real dt = 0.0);

  protected:
    StdLargeVec<Real> &Vol_;
};

template <>
class NablaWV<Inner<LevelSetCorrection>> : public NablaWV<Inner<>>
{
  public:
    template <typename... Args>
    NablaWV(Args &&...args)
        : NablaWV<Inner<>>(std::forward<Args>(args)...),
        inner_shape_(sph_body_.getInitialShape()),
      pos_(*particles_->getVariableByName<Vecd>("Position")),
        sph_adaptation_(this->sph_body_.sph_adaptation_),
      level_set_shape_(DynamicCast<LevelSetShape>(this, inner_shape_)) {};

    template <typename BodyRelationType, typename FirstArg>
    explicit NablaWV(ConstructorArgs<BodyRelationType, FirstArg> parameters)
        : NablaWV<Inner<>>(parameters.body_relation_),
          inner_shape_(*DynamicCast<ComplexShape>(this, sph_body_.getInitialShape())
                        .getSubShapeByName(std::get<0>(parameters.others_))),
          pos_(*particles_->getVariableByName<Vecd>("Position")),
          sph_adaptation_(this->sph_body_.sph_adaptation_),
          level_set_shape_(DynamicCast<LevelSetShape>(this, inner_shape_)) {};

    virtual ~NablaWV(){};

    void interaction(size_t index_i, Real dt = 0.0)
    {
        NablaWV<Inner<>>::interaction(index_i, dt);
        kernel_sum_[index_i] += level_set_shape_.computeKernelGradientIntegral(
                                       pos_[index_i], sph_adaptation_->SmoothingLengthRatio(index_i));
    }

  protected:
    Shape &inner_shape_;
    StdLargeVec<Vecd> &pos_;
    LevelSetShape &level_set_shape_;
    SPHAdaptation *sph_adaptation_;
};

template <>
class NablaWV<Contact<>>
    : public NablaWV<DataDelegateContact>
{
  public:
    explicit NablaWV(BaseContactRelation &contact_relation)
        : NablaWV<DataDelegateContact>(contact_relation)
    {
        for (size_t k = 0; k < contact_configuration_.size(); ++k)
        {
            contact_Vol_.push_back(contact_particles_[k]->getVariableByName<Real>("VolumetricMeasure"));
        }
    };
    virtual ~NablaWV(){};
    void interaction(size_t index_i, Real dt = 0.0);

    StdVec<StdLargeVec<Real> *> contact_Vol_;
};

using NablaWVComplex = ComplexInteraction<NablaWV<Inner<>, Contact<>>>;

using NablaWVLevelSetCorrectionInner = NablaWV<Inner<LevelSetCorrection>>;
using NablaWVLevelSetCorrectionComplex = ComplexInteraction<NablaWV<Inner<LevelSetCorrection>, Contact<>>>;

} // namespace SPH
#endif // KERNEL_SUMMATION_H
