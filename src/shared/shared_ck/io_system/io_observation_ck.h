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
 * @file 	io_observation_ck.h
 * @brief 	TBD.
 * @author	Xiangyu Hu
 */

#ifndef IO_OBSERVATION_CK_H
#define IO_OBSERVATION_CK_H

#include "io_observation.h"

#include "execution_policy.h"
#include "interpolation_dynamics.hpp"

namespace SPH
{
template <class ExecutionPolicy, typename DataType>
class ObservedQuantityRecording<ExecutionPolicy, DataType>
    : public BaseQuantityRecording,
      public ObservingAQuantityCK<ExecutionPolicy, DataType>
{
  protected:
    SPHBody &observer_;
    BaseParticles &base_particles_;

  public:
    DataType type_indicator_; /*< this is an indicator to identify the variable type. */

  public:
    ObservedQuantityRecording(const std::string &quantity_name, Relation<Contact<>> &contact_relation)
        : BaseQuantityRecording(contact_relation.getSPHBody().getSPHSystem(),
                                contact_relation.getSPHBody().getName(), quantity_name),
          ObservingAQuantityCK<ExecutionPolicy, DataType>(contact_relation, quantity_name),
          observer_(contact_relation.getSPHBody()),
          base_particles_(observer_.getBaseParticles())
    {
        std::ofstream out_file(filefullpath_output_.c_str(), std::ios::app);
        out_file << "run_time" << "   ";
        DataType *interpolated_quantities = this->dv_interpolated_quantities_->Data();
        for (size_t i = 0; i != base_particles_.TotalRealParticles(); ++i)
        {
            std::string quantity_name_i = quantity_name + "[" + std::to_string(i) + "]";
            plt_engine_.writeAQuantityHeader(out_file, interpolated_quantities[i], quantity_name_i);
        }
        out_file << "\n";
        out_file.close();
    };
    virtual ~ObservedQuantityRecording() {};

    virtual void writeToFile(size_t iteration_step = 0) override
    {
        std::ofstream out_file(filefullpath_output_.c_str(), std::ios::app);
        out_file << sv_physical_time_->getValue() << "   ";
        this->exec();
        this->dv_interpolated_quantities_->prepareForOutput(ExecutionPolicy{});
        DataType *interpolated_quantities = this->dv_interpolated_quantities_->Data();
        for (size_t i = 0; i != base_particles_.TotalRealParticles(); ++i)
        {
            plt_engine_.writeAQuantity(out_file, interpolated_quantities[i]);
        }
        out_file << "\n";
        out_file.close();
    };

    DataType *getObservedQuantity()
    {
        return this->interpolated_quantities_;
    }
};

template <class ExecutionPolicy, class LocalReduceMethodType>
class ReducedQuantityRecording<ExecutionPolicy, LocalReduceMethodType> : public BaseQuantityRecording
{
  protected:
    ReduceDynamicsCK<ExecutionPolicy, LocalReduceMethodType> reduce_method_;

  public:
    /*< deduce variable type from reduce method. */
    using VariableType = typename LocalReduceMethodType::FinishDynamics::OutputType;
    VariableType type_indicator_; /*< this is an indicator to identify the variable type. */

  public:
    template <class DynamicsIdentifier, typename... Args>
    ReducedQuantityRecording(DynamicsIdentifier &identifier, Args &&...args)
        : BaseQuantityRecording(identifier.getSPHBody().getSPHSystem(),
                                identifier.getName(), ""),
          reduce_method_(identifier, std::forward<Args>(args)...)
    {
        quantity_name_ = reduce_method_.QuantityName();
        std::ofstream out_file(filefullpath_output_.c_str(), std::ios::app);
        out_file << "\"run_time\"" << "   ";
        plt_engine_.writeAQuantityHeader(out_file, ZeroData<VariableType>::value, quantity_name_);
        out_file << "\n";
        out_file.close();
    };
    virtual ~ReducedQuantityRecording() {};

    virtual void writeToFile(size_t iteration_step = 0) override
    {
        std::ofstream out_file(filefullpath_output_.c_str(), std::ios::app);
        out_file << sv_physical_time_->getValue() << "   ";
        plt_engine_.writeAQuantity(out_file, reduce_method_.exec());
        out_file << "\n";
        out_file.close();
    };
};
} // namespace SPH
#endif // IO_OBSERVATION_CK_H
