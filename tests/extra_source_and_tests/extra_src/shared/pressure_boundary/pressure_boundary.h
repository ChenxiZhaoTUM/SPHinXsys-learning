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

//-------------------------------------------------------------------------------
//	Calculate flow rate by velocity times area.
//-------------------------------------------------------------------------------
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
        Real average_velocity_norm = ReduceSumType::outputResult(reduced_value) / Real(this->getDynamicsIdentifier().SizeOfLoopRange());
        Q_ = average_velocity_norm * outlet_area_;
        //std::cout << "Q_ = " << Q_ << std::endl;  /* instantaneous flow rate */
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
          R1_(R1), R2_(R2), C_(C),
          Q_transient_(*particles_->getSingleVariableByName<Real>("FlowRate")),
          Q_ave_(8.273e-5), //temporary
          p_outlet_(*particles_->registerSingleVariable<Real>("OutletPressure")),
          p_outlet_next_(*particles_->registerSingleVariable<Real>("NextOutletPressure")),
          accumulated_flow_(0.0), accumulated_time_(0.0), pre_accumulated_flow_(0.0) {};
    virtual ~RCRPressure(){};

    void accumulateFlow(Real dt)
    {
        //std::cout << "Q_transient_ = " << Q_transient_ << std::endl;
        accumulated_flow_ += Q_transient_ * dt;
        accumulated_time_ += dt;
    }

    void updatePreAndResetAcc()
    {
        pre_accumulated_flow_ = accumulated_flow_;
        p_outlet_ = p_outlet_next_;
        std::cout << "p_outlet_next_ = " << p_outlet_next_ << std::endl;

        accumulated_flow_ = 0.0;
        accumulated_time_ = 0.0;
    }

    void writeOutletP()
    {
        std::string output_folder = "./output";
        std::string filefullpath = output_folder + "/" + "outlet_pressure.dat";
        std::ofstream out_file(filefullpath.c_str(), std::ios::app);
        out_file << GlobalStaticVariables::physical_time_ << "   " << p_outlet_next_ <<  "\n";
        out_file.close();
    }

    void updateNextPressure()
    {
        Real Q_current = accumulated_flow_ / accumulated_time_ - Q_ave_;
        Real dp_dt = - p_outlet_ / (C_ * R2_) + (R1_ + R2_) * Q_current / (C_ * R2_) + R1_ * (accumulated_flow_ - pre_accumulated_flow_) / (accumulated_time_ + TinyReal);
        Real p_star = p_outlet_ + dp_dt * accumulated_time_;
        Real dp_dt_star = - p_star / (C_ * R2_) + (R1_ + R2_) * Q_current / (C_ * R2_) + R1_ * (accumulated_flow_ - pre_accumulated_flow_) / (accumulated_time_ + TinyReal);
        p_outlet_next_ = p_outlet_ + 0.5 * accumulated_time_ * (dp_dt + dp_dt_star);


        //p_outlet_next_ = ((Q_current * (1.0 + R1_ / R2_) + C_ * R1_ * (accumulated_flow_ - pre_accumulated_flow_) / accumulated_time_) * accumulated_time_ / C_ + p_outlet_) / (1.0 + accumulated_time_ / (C_ * R2_));

        updatePreAndResetAcc();
        writeOutletP();
    }

    /*
    void updateNextPressureRK4()
    {
        // Compute dp/dt at current state (t_n, p_outlet_)
        auto dp_dt = [&](Real p) {
            return - p / (C_ * R2_) + (R1_ + R2_) * accumulated_flow_ / (C_ * R2_) + R1_ * (accumulated_flow_ - Q_pre_) / (accumulated_time_ + TinyReal);
        };

        // RK4 coefficients
        Real k1 = dp_dt(p_outlet_);
        Real k2 = dp_dt(p_outlet_ + 0.5 * accumulated_time_ * k1);
        Real k3 = dp_dt(p_outlet_ + 0.5 * accumulated_time_ * k2);
        Real k4 = dp_dt(p_outlet_ + accumulated_time_ * k3);

        // Update p_outlet_ using RK4
        Real dp = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        p_outlet_next_ = p_outlet_ + dp * accumulated_time_;

        updatePreAndResetAcc();
        writeOutletP();
    }

    void updateNextPressureGeneralizedAlpha()
    {
        // Define parameters for the generalized-alpha method
        Real rho_inf = 0.8; // High-frequency damping control parameter
        Real alpha_m = (2.0 * rho_inf - 1) / (rho_inf + 1);
        Real alpha_f = rho_inf / (rho_inf + 1);
        Real gamma = 0.5 + alpha_m - alpha_f;
        Real beta = 0.25 * (1.0 + alpha_m - alpha_f) * (1.0 + alpha_m - alpha_f);

        // Predictor step for p and dp/dt
        Real dp_dt_n = -p_outlet_ / (C_ * R2_) + (R1_ + R2_) * accumulated_flow_ / (C_ * R2_) + R1_ * (accumulated_flow_ - Q_pre_) / (accumulated_time_ + TinyReal);

        // Predict velocity and acceleration at time n+1/2
        Real p_pred = p_outlet_ + accumulated_time_ * dp_dt_n;
        Real dp_dt_pred = dp_dt_n; // Start with current derivative as initial guess for predictor

        // Implicit step to solve for dp/dt and p at n+1
        Real dp_dt_n1 = dp_dt_pred;
        Real p_n1 = p_pred;

        // Use Newton-Raphson method for implicit update
        for (int iter = 0; iter < 10; ++iter)
        {
            Real dp_dt_star = -p_n1 / (C_ * R2_) + (R1_ + R2_) * accumulated_flow_ / (C_ * R2_) + R1_ * (accumulated_flow_ - Q_pre_) / (accumulated_time_ + TinyReal);

            // Residual and Jacobian
            Real residual = p_n1 - p_outlet_ - accumulated_time_ * ((1 - gamma) * dp_dt_n + gamma * dp_dt_star);
            Real jacobian = 1.0 + accumulated_time_ * gamma / (C_ * R2_);

            // Newton-Raphson update
            Real delta = -residual / jacobian;
            p_n1 += delta;

            // Convergence check
            if (fabs(delta) < 1e-6) break;
        }

        p_outlet_next_ = p_n1;

        updatePreAndResetAcc();
        writeOutletP();
    }
    */

    Real operator()(Real &p_current)
    {
        return p_outlet_next_;
    }

  protected:
    // parameters about Windkessel model
    Real R1_, R2_, C_;
    Real& Q_transient_;
    Real Q_ave_;
    Real &p_outlet_, &p_outlet_next_;
    Real accumulated_flow_, accumulated_time_;
    Real pre_accumulated_flow_;
};

//-------------------------------------------------------------------------------
//	Calculate flow rate by total particle volume of injection(-) and deletion(+).
//-------------------------------------------------------------------------------
class RCRPressureByDeletion : public BaseLocalDynamics<BodyPartByCell>, public DataDelegateSimple
{
  public:
    explicit RCRPressureByDeletion(BodyAlignedBoxByCell& aligned_box_part, Real R1, Real R2, Real C)
        : BaseLocalDynamics<BodyPartByCell>(aligned_box_part),
          DataDelegateSimple(aligned_box_part.getSPHBody()),
          R1_(R1), R2_(R2), C_(C),
          Q_(*particles_->getSingleVariableByName<Real>("TotalVolDeletion")),
          Q_pre_(*particles_->registerSingleVariable<Real>("PreViousFlowRate")),
          p_outlet_(*particles_->registerSingleVariable<Real>("OutletPressure")),
          p_outlet_next_(*particles_->registerSingleVariable<Real>("NextOutletPressure")),
          initial_q_pre_set_(false),
          accumulated_time_(0.0) {};
    virtual ~RCRPressureByDeletion(){};

    void setInitialQPre()
    {
        if (!initial_q_pre_set_) // Check if it has been executed before
        {
            Q_pre_ = Q_;
            initial_q_pre_set_ = true; // Set to true after the first execution
        }
    }

    void setAccumulationTime(Real dt)
    {
        accumulated_time_ = dt;
    }

    void resetAccumulation()
    {
        Q_ = 0.0;
        accumulated_time_ = 0.0;
    }

    void updateNextPressure()
    {
        std::cout << "Q_ for p_next calculation is Q_ = " << Q_ << std::endl;
        //std::cout << "Q_pre_ for p_next calculation is Q_pre_ = " << Q_pre_ << std::endl;
        Real dp_dt = - p_outlet_ / (C_ * R2_) + (R1_ + R2_) * Q_ / (C_ * R2_) + R1_ * (Q_ - Q_pre_) / (accumulated_time_ + TinyReal);
        Real p_star = p_outlet_ + dp_dt * accumulated_time_;
        Real dp_dt_star = - p_star / (C_ * R2_) + (R1_ + R2_) * Q_ / (C_ * R2_) + R1_ * (Q_ - Q_pre_) / (accumulated_time_ + TinyReal);
        p_outlet_next_ = p_outlet_ + 0.5 * accumulated_time_ * (dp_dt + dp_dt_star);

        Q_pre_ = Q_;
        p_outlet_ = p_outlet_next_;
        std::cout << "p_outlet_next_ = " << p_outlet_next_ << std::endl;
        resetAccumulation();
        
        std::string output_folder = "./output";
        std::string filefullpath = output_folder + "/" + "outlet_pressure.dat";
        std::ofstream out_file(filefullpath.c_str(), std::ios::app);
        out_file << GlobalStaticVariables::physical_time_ << "   " << p_outlet_next_ <<  "\n";
        out_file.close();
    }

    Real operator()(Real &p_current)
    {
        return p_outlet_next_;
    }

  protected:
    // parameters about Windkessel model
    Real R1_, R2_, C_;
    Real &Q_, &Q_pre_;
    Real &p_outlet_, &p_outlet_next_;
    bool initial_q_pre_set_;
    Real accumulated_time_;
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