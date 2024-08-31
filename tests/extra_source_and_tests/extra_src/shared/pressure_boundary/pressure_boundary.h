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

class TotalVelocityNormVal
    : public BaseLocalDynamicsReduce<ReduceSum<Real>, BodyPartByCell>,
      public DataDelegateSimple
{
  protected:
    StdLargeVec<Vecd> &vel_;
    AlignedBoxShape &aligned_box_;
    const int alignment_axis_;
    Transform &transform_;

  public:
      explicit TotalVelocityNormVal(BodyAlignedBoxByCell& aligned_box_part)
          : BaseLocalDynamicsReduce<ReduceSum<Real>, BodyPartByCell>(aligned_box_part),
          DataDelegateSimple(aligned_box_part.getSPHBody()),
          vel_(*particles_->getVariableDataByName<Vecd>("Velocity")),
          aligned_box_(aligned_box_part.getAlignedBoxShape()),
          alignment_axis_(aligned_box_.AlignmentAxis()),
          transform_(aligned_box_.getTransform())
      {
          // std::cout << "TotalVelocityNormVal constructor!" << std::endl;
      };

    virtual ~TotalVelocityNormVal(){};

    Real reduce(size_t index_i, Real dt = 0.0)
    {
        Vecd frame_velocity = Vecd::Zero();
        frame_velocity[alignment_axis_] = transform_.xformBaseVecToFrame(vel_[index_i])[alignment_axis_];
        //std::cout << "frame_velocity =  " << frame_velocity << std::endl;
        //std::cout << "frame_velocity[alignment_axis_] =  " << frame_velocity[alignment_axis_] << std::endl;
        return -frame_velocity[alignment_axis_];
    }
};

template <class ReduceSumType>
class AverageFlowRate : public ReduceSumType
{
  public:
    explicit AverageFlowRate(BodyAlignedBoxByCell& aligned_box_part, Real outlet_area)
          : ReduceSumType(aligned_box_part),
            Q_(*this->particles_->registerSingleVariable<Real>("FlowRate")),
            outlet_area_(outlet_area) {};
    virtual ~AverageFlowRate(){};

    virtual Real outputResult(Real reduced_value) override
    {
        //std::cout << "ReduceSumType::outputResult(reduced_value) =  " << ReduceSumType::outputResult(reduced_value) << std::endl;

        Real average_velocity_norm = ReduceSumType::outputResult(reduced_value) / Real(this->getDynamicsIdentifier().SizeOfLoopRange());
        Q_ = average_velocity_norm * outlet_area_;
        std::cout << "Q_ = " << Q_ << std::endl;
        return Q_;
    }

  private:
    Real &Q_;
    Real outlet_area_;
};

class RCRPressure : public BaseLocalDynamics<BodyPartByCell>, public DataDelegateSimple
{
  public:
    explicit RCRPressure(BodyAlignedBoxByCell& aligned_box_part, Real R1, Real R2, Real C)
        : BaseLocalDynamics<BodyPartByCell>(aligned_box_part),
          DataDelegateSimple(aligned_box_part.getSPHBody()),
          R1_(R1), R2_(R2), C_(C), dt_(0.0),
          Q_(*particles_->getSingleVariableByName<Real>("FlowRate")),
          Q_pre_(*particles_->registerSingleVariable<Real>("PreViousFlowRate")),
          p_outlet_(*particles_->registerSingleVariable<Real>("OutletPressure")),
          p_outlet_next_(*particles_->registerSingleVariable<Real>("NextOutletPressure")),
          initial_q_pre_set_(false) {};
    virtual ~RCRPressure(){};

    void setInitialQPre()
    {
        if (!initial_q_pre_set_) // Check if it has been executed before
        {
            Q_pre_ = Q_;
            initial_q_pre_set_ = true; // Set to true after the first execution
        }
    }

    void setTimeStep(Real dt) 
    {  
        dt_ = dt;
        //std::cout << "Now time is setting: dt_ = " << dt_ << std::endl;
    }

    void updateNextPressure()
    {
        std::cout << "Q_ for p_next calculation is Q_ = " << Q_ << std::endl;
        std::cout << "Q_pre_ for p_next calculation is Q_pre_ = " << Q_pre_ << std::endl;
        Real dp_dt = - p_outlet_ / (C_ * R2_) + (R1_ + R2_) * Q_ / (C_ * R2_) + R1_ * (Q_ - Q_pre_) / (dt_ + TinyReal);
        Real p_star = p_outlet_ + dp_dt * dt_;
        Real dp_dt_star = - p_star / (C_ * R2_) + (R1_ + R2_) * Q_ / (C_ * R2_) + R1_ * (Q_ - Q_pre_) / (dt_ + TinyReal);
        p_outlet_next_ = p_outlet_ + 0.5 * dt_ * (dp_dt + dp_dt_star);

        Q_pre_ = Q_;
        p_outlet_ = p_outlet_next_;
        //std::cout << "p_outlet_next_ = " << p_outlet_next_ << std::endl;
    }

    Real operator()(Real &p_current)
    {
        return p_outlet_next_;
    }

  protected:
    // parameters about Windkessel model
    Real R1_, R2_, C_;
    Real dt_;
    Real &Q_, &Q_pre_;
    Real &p_outlet_, &p_outlet_next_;
    bool initial_q_pre_set_;;
};

template <typename TargetPressure>
class WindkesselCondition : public BaseFlowBoundaryCondition
{
  public:
    /** default parameter indicates prescribe pressure */
    explicit WindkesselCondition(BodyAlignedBoxByCell &aligned_box_part, Real R1, Real R2, Real C)
        : BaseFlowBoundaryCondition(aligned_box_part),
          aligned_box_(aligned_box_part.getAlignedBoxShape()),
          alignment_axis_(aligned_box_.AlignmentAxis()),
          transform_(aligned_box_.getTransform()),
          target_pressure_(TargetPressure(aligned_box_part, R1, R2, C)),
          kernel_sum_(*particles_->getVariableDataByName<Vecd>("KernelSummation")){};
    virtual ~WindkesselCondition(){};
    AlignedBoxShape &getAlignedBox() { return aligned_box_; };

    TargetPressure *getTargetPressure() { return &target_pressure_; }

    void update(size_t index_i, Real dt = 0.0)
    {
        vel_[index_i] += 2.0 * kernel_sum_[index_i] * target_pressure_(p_[index_i]) / rho_[index_i] * dt;
        //std::cout << "target_pressure_(p_[index_i]) = " << target_pressure_(p_[index_i]) << std::endl;
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
} // namespace fluid_dynamics
} // namespace SPH
#endif // PRESSURE_BOUNDARY_H