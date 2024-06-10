#pragma once

#include "kernel_summation.h"

namespace SPH
{
//=================================================================================================//
template <class DataDelegationType>
template <class BaseRelationType>
NablaWV<DataDelegationType>::NablaWV(BaseRelationType &base_relation)
    : LocalDynamics(base_relation.getSPHBody()), DataDelegationType(base_relation),
      kernel_sum_(*this->particles_->template registerSharedVariable<Vecd>("KernelSummation"))
 {}
//=================================================================================================//
template <typename... Args>
NablaWV<Inner<LevelSetCorrection>>::NablaWV(Args &&...args)
    : NablaWV<Inner<>>(std::forward<Args>(args)...),
        inner_shape_(sph_body_.getInitialShape()),
      pos_(*particles_->getVariableByName<Vecd>("Position")),
        sph_adaptation_(this->sph_body_.sph_adaptation_),
      level_set_shape_(DynamicCast<LevelSetShape>(this, inner_shape_)) {}
//=================================================================================================//
template <typename BodyRelationType, typename FirstArg>
NablaWV<Inner<LevelSetCorrection>>::NablaWV(ConstructorArgs<BodyRelationType, FirstArg> parameters)
    : NablaWV<Inner<>>(parameters.body_relation_),
        inner_shape_(*DynamicCast<ComplexShape>(this, sph_body_.getInitialShape()).getSubShapeByName(std::get<0>(parameters.others_))),
        pos_(*particles_->getVariableByName<Vecd>("Position")),
        sph_adaptation_(this->sph_body_.sph_adaptation_),
        level_set_shape_(DynamicCast<LevelSetShape>(this, inner_shape_)) {}
//=================================================================================================//
} // namespace SPH