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
 * @file 	pressure_boundary.h
 * @brief 	Here, we define the pressure boundary condition class for fluid dynamics.
 * @details The boundary conditions very often based on different types of buffers.
 * @author  Shuoguo Zhang and Xiangyu Hu
 */

#ifndef PRESSURE_BOUNDARY_H
#define PRESSURE_BOUNDARY_H

#include "fluid_boundary.h"

namespace SPH
{
namespace fluid_dynamics
{
template <typename TargetPressure>
class PressureCondition : public BaseFlowBoundaryCondition
{
  public:
    /** default parameter indicates prescribe pressure */
    explicit PressureCondition(BodyAlignedBoxByCell &aligned_box_part)
        : BaseFlowBoundaryCondition(aligned_box_part),
          aligned_box_(aligned_box_part.getAlignedBoxShape()),
          alignment_axis_(aligned_box_.AlignmentAxis()),
          transform_(aligned_box_.getTransform()),
          target_pressure_(*this),
          kernel_sum_(*particles_->getVariableDataByName<Vecd>("KernelSummation")){};
    virtual ~PressureCondition(){};
    AlignedBoxShape &getAlignedBox() { return aligned_box_; };

    void update(size_t index_i, Real dt = 0.0)
    {
        vel_[index_i] += 2.0 * kernel_sum_[index_i] * target_pressure_(p_[index_i]) / rho_[index_i] * dt;

        Vecd frame_velocity = Vecd::Zero();
        frame_velocity[alignment_axis_] = transform_.xformBaseVecToFrame(vel_[index_i])[alignment_axis_];
        vel_[index_i] = transform_.xformFrameVecToBase(frame_velocity);
    };

  protected:
    AlignedBoxShape &aligned_box_;
    const int alignment_axis_;
    Transform &transform_;
    TargetPressure target_pressure_;
    StdLargeVec<Vecd> &kernel_sum_;
};

class WindkesselCondition : public BaseFlowBoundaryCondition
{
  public:
    /** default parameter indicates prescribe pressure */
    explicit WindkesselCondition(BodyAlignedBoxByCell &aligned_box_part, Real R1, Real R2, Real C, Real radius)
        : BaseFlowBoundaryCondition(aligned_box_part),
          aligned_box_(aligned_box_part.getAlignedBoxShape()),
          alignment_axis_(aligned_box_.AlignmentAxis()),
          transform_(aligned_box_.getTransform()),
          kernel_sum_(*particles_->getVariableDataByName<Vecd>("KernelSummation")),
          R1_(R1), R2_(R2), C_(C), outlet_radius_(radius),
          p_target_(*particles_->registerSingleVariable<Real>("TargetPressure")),
          Q_(*particles_->registerSingleVariable<Real>("FlowRate")),
          Q_pre_(*particles_->registerSingleVariable<Real>("PreViousFlowRate")) {};
    virtual ~WindkesselCondition(){};
    AlignedBoxShape &getAlignedBox() { return aligned_box_; };

    void computeFlowRate()
    {
        Real vel_normal_sum = 0.0;
        for (size_t n = 0; n != vel_.size(); ++n)
        {
            Vecd frame_velocity = Vecd::Zero();
            frame_velocity[alignment_axis_] = transform_.xformBaseVecToFrame(vel_[n])[alignment_axis_];
            vel_normal_sum += frame_velocity[alignment_axis_];
            //not sure
        }
        Real vel_normal_val = vel_normal_sum / vel_.size();
        Q_ = vel_normal_val * Pi * pow(outlet_radius_, 2);
    }

    Real operator()(Real p_current)
    {
        // update p_target_
        Real dp_dt = - p_target_ / (C_ * R2_) + (R1 + R2) * Q_ / (C_ * R2_) + R1_ * (Q_ - Q_pre_) / dt;
        Real p_star = p_target_ + dp_dt * dt;
        Real dp_dt_star = - p_star / (C_ * R2_) + (R1_ + R2_) * Q_ / (C_ * R2_) + R1_ * (Q_ - Q_pre_) / dt;
        p_target_ += 0.5 * dt * (dp_dt + dp_dt_star);
        return p_target_;
    }

    void update(size_t index_i, Real dt = 0.0)
    {
        vel_[index_i] += 2.0 * kernel_sum_[index_i] * this->(p_[index_i]) / rho_[index_i] * dt;
        Vecd frame_velocity = Vecd::Zero();
        frame_velocity[alignment_axis_] = transform_.xformBaseVecToFrame(vel_[index_i])[alignment_axis_];
        vel_[index_i] = transform_.xformFrameVecToBase(frame_velocity);
    };

    void updatePreviousTargetP()
    {
        Q_pre_ = Q_;
    }

  protected:
    AlignedBoxShape &aligned_box_;
    const int alignment_axis_;
    Transform &transform_;
    StdLargeVec<Vecd> &kernel_sum_;

    // parameters about Wendkessel model
    Real R1_, R2_, C_;
    Real outlet_radius_;
    Real p_target_;
    Real Q_, Q_pre_;
};
} // namespace fluid_dynamics
} // namespace SPH
#endif // PRESSURE_BOUNDARY_H