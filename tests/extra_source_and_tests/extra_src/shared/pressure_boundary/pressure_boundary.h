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
    template <typename... Args>
    explicit PressureCondition(BodyAlignedBoxByCell &aligned_box_part, Args &&...args)
        : BaseFlowBoundaryCondition(aligned_box_part),
          aligned_box_(aligned_box_part.getAlignedBoxShape()),
          alignment_axis_(aligned_box_.AlignmentAxis()),
          transform_(aligned_box_.getTransform()),
          target_pressure_(TargetPressure(aligned_box_part, std::forward<Args>(args)...)),
          kernel_sum_(*particles_->getVariableDataByName<Vecd>("KernelSummation")){};
    virtual ~PressureCondition(){};
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

//-------------------------------------------------------------------------------
//	Calculate flow rate and update pressure with WindkesselModel class.
//-------------------------------------------------------------------------------
class WindkesselModel : public BaseLocalDynamics<BodyPartByCell>, public DataDelegateSimple
{
  public:
    explicit WindkesselModel(BodyAlignedBoxByCell &aligned_box_part, const std::string &body_part_name, Real R1, Real R2, Real C, Real dt, Real Q_ave)
        : BaseLocalDynamics<BodyPartByCell>(aligned_box_part),
          DataDelegateSimple(aligned_box_part.getSPHBody()),
          aligned_box_(aligned_box_part.getAlignedBoxShape()),
          pos_(*particles_->getVariableDataByName<Vecd>("Position")),
          Vol_(*particles_->getVariableDataByName<Real>("VolumetricMeasure")),
          buffer_particle_indicator_(*particles_->registerSharedVariable<int>("BufferParticleIndicator")),
          body_part_name_(body_part_name),
          R1_(R1), R2_(R2), C_(C), delta_t_(dt), Q_ave_(Q_ave),
          Q_n_(0.0), Q_0_(0.0),
          p_n_(0.0), p_0_(0.0),
          flow_rate_(0.0), current_flow_rate_(0.0), previous_flow_rate_(0.0),
          updateP_n(0), ifUpdateP(false)
    {
        particles_->addVariableToSort<int>("BufferParticleIndicator");
    };
    virtual ~WindkesselModel(){};

    void update(size_t index_i, Real dt = 0.0)
    {
        calculateFlowRate(index_i);

        updateP_lock_.lock();
        Real run_time = GlobalStaticVariables::physical_time_;
        ifUpdateP = (run_time >= updateP_n * delta_t_) ? true : false;
        if (ifUpdateP)
        {
            updateNextPressure();
            ++updateP_n;
            ifUpdateP = false;
        }
        updateP_lock_.unlock();
    }

    Real &getOutletPressure()
    {
        return p_n_;
    }

  protected:
    AlignedBoxShape &aligned_box_;
    StdLargeVec<Vecd> &pos_;
    StdLargeVec<Real> &Vol_;
    StdLargeVec<int> &buffer_particle_indicator_;
    
    std::string body_part_name_;
    Real R1_, R2_, C_, delta_t_, Q_ave_;
    Real Q_n_, Q_0_;
    Real p_n_, p_0_;
    Real flow_rate_, current_flow_rate_, previous_flow_rate_;

    int updateP_n;
    bool ifUpdateP;
    std::mutex updateP_lock_;

    void calculateFlowRate(size_t index_i)
    {
        if (aligned_box_.checkUpperBound(pos_[index_i]) && index_i < particles_->TotalRealParticles())
        {
            flow_rate_ += Vol_[index_i];
        }

        if (aligned_box_.checkLowerBound(pos_[index_i]) && buffer_particle_indicator_[index_i] == 1)
        {
            flow_rate_ -= Vol_[index_i];
        }
    }

    void updateNextPressure()
    {
        Q_0_ = Q_n_;
        p_0_ = p_n_;
        current_flow_rate_ = flow_rate_ - previous_flow_rate_;
        previous_flow_rate_ = flow_rate_;

        Q_n_ = current_flow_rate_ / delta_t_ - Q_ave_;
        /*Real dp_dt = - p_0_ / (C_ * R2_) + (R1_ + R2_) * Q_n_ / (C_ * R2_) + R1_ * (Q_n_ - Q_0_) / (delta_t_ + TinyReal);
        Real p_star = p_0_ + dp_dt * delta_t_;
        Real dp_dt_star = - p_star / (C_ * R2_) + (R1_ + R2_) * Q_n_ / (C_ * R2_) + R1_ * (Q_n_ - Q_0_) / (delta_t_ + TinyReal);
        p_n_ = p_0_ + 0.5 * delta_t_ * (dp_dt + dp_dt_star);*/

        p_n_ = ((Q_n_ * (1.0 + R1_ / R2_) + C_ * R1_ * (Q_n_ - Q_0_) / delta_t_) * delta_t_ / C_ + p_0_) / (1.0 + delta_t_ / (C_ * R2_));

        writeOutletPressureData();
        writeOutletFlowRateData();
    }

    void writeOutletPressureData()
    {
        std::string output_folder = "./output";
        std::string filefullpath = output_folder + "/" + body_part_name_ + "_outlet_pressure.dat";
        std::ofstream out_file(filefullpath.c_str(), std::ios::app);
        out_file << GlobalStaticVariables::physical_time_ << "   " << p_n_ << "\n";
        out_file.close();
    }

    void writeOutletFlowRateData()
    {
        std::string output_folder = "./output";
        std::string filefullpath = output_folder + "/" + body_part_name_ + "_flow_rate.dat";
        std::ofstream out_file(filefullpath.c_str(), std::ios::app);
        out_file << GlobalStaticVariables::physical_time_ << "   " << Q_n_ << "\n";
        out_file.close();
    }
};

struct TargetPressureByWindekesselModel
{
    Real &p_n_;

    TargetPressureByWindekesselModel(BodyAlignedBoxByCell &aligned_box_part, WindkesselModel &windkessel_model)
        : p_n_(windkessel_model.getOutletPressure()) {}

    Real operator()(Real &p_)
    {
        return p_n_;
    }
};

using PressureConditionByWindekesselModel = PressureCondition<TargetPressureByWindekesselModel>;


//-------------------------------------------------------------------------------
//	Calculate flow rate by total particle volume of injection(-) and deletion(+).
//-------------------------------------------------------------------------------
class TargetOutletPressureWindkessel : public BaseLocalDynamics<BodyPartByCell>, public DataDelegateSimple
{
  public:
    explicit TargetOutletPressureWindkessel(BodyAlignedBoxByCell& aligned_box_part, const std::string &body_part_name)
        : BaseLocalDynamics<BodyPartByCell>(aligned_box_part),
          DataDelegateSimple(aligned_box_part.getSPHBody()),
          body_part_name_(body_part_name),
          R1_(0.0), R2_(0.0), C_(0.0), Q_ave_(0.0), delta_t_(0.0), 
          Q_n_(0.0), Q_0_(0.0), p_n_(0.0), p_0_(0.0),
          flow_rate_(*particles_->getSingleVariableByName<Real>(body_part_name+"FlowRate")),
          current_flow_rate_(0.0), previous_flow_rate_(0.0) {};
    virtual ~TargetOutletPressureWindkessel(){};

    void setWindkesselParams(Real R1, Real R2, Real C, Real dt, Real Q_ave)
    {
        R1_ = R1;
        R2_ = R2;
        C_ = C;
        Q_ave_ = Q_ave;
        delta_t_ = dt;
    }

    void updateNextPressure()
    {
        getFlowRate();

        Q_n_ = current_flow_rate_ / delta_t_ - Q_ave_;
        /*Real dp_dt = - p_0_ / (C_ * R2_) + (R1_ + R2_) * Q_n_ / (C_ * R2_) + R1_ * (Q_n_ - Q_0_) / (delta_t_ + TinyReal);
        Real p_star = p_0_ + dp_dt * delta_t_;
        Real dp_dt_star = - p_star / (C_ * R2_) + (R1_ + R2_) * Q_n_ / (C_ * R2_) + R1_ * (Q_n_ - Q_0_) / (delta_t_ + TinyReal);
        p_n_ = p_0_ + 0.5 * delta_t_ * (dp_dt + dp_dt_star);*/

        p_n_ = ((Q_n_ * (1.0 + R1_ / R2_) + C_ * R1_ * (Q_n_ - Q_0_) / delta_t_) * delta_t_ / C_ + p_0_) / (1.0 + delta_t_ / (C_ * R2_));

        std::cout << "p_n_ = " << p_n_ << std::endl;

        writeOutletPressureData();
        writeOutletFlowRateData();
    }

    Real operator()(Real &p_current)
    {
        return p_n_;
    }

  protected:
    std::string body_part_name_;
    Real R1_, R2_, C_, Q_ave_, delta_t_;
    Real Q_n_, Q_0_;
    Real p_n_, p_0_;
    Real &flow_rate_, current_flow_rate_, previous_flow_rate_;

    void getFlowRate()
    {
        Q_0_ = Q_n_;
        p_0_ = p_n_;
        current_flow_rate_ = flow_rate_ - previous_flow_rate_;
        previous_flow_rate_ = flow_rate_;
    }

    void writeOutletPressureData()
    {
        std::string output_folder = "./output";
        std::string filefullpath = output_folder + "/" + body_part_name_ + "_outlet_pressure.dat";
        std::ofstream out_file(filefullpath.c_str(), std::ios::app);
        out_file << GlobalStaticVariables::physical_time_ << "   " << p_n_ <<  "\n";
        out_file.close();
    }

    void writeOutletFlowRateData()
    {
        std::string output_folder = "./output";
        std::string filefullpath = output_folder + "/" + body_part_name_ + "_flow_rate.dat";
        std::ofstream out_file(filefullpath.c_str(), std::ios::app);
        out_file << GlobalStaticVariables::physical_time_ << "   " << Q_n_ <<  "\n";
        out_file.close();
    }
};

using WindkesselBoundaryCondition = PressureCondition<TargetOutletPressureWindkessel>;


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
      explicit TotalVelocityNormVal(BodyAlignedBoxByCell& disposer_part)
          : BaseLocalDynamicsReduce<ReduceSum<Real>, BodyPartByCell>(disposer_part),
          DataDelegateSimple(disposer_part.getSPHBody()),
          vel_(*particles_->getVariableDataByName<Vecd>("Velocity")),
          aligned_box_(disposer_part.getAlignedBoxShape()),
          alignment_axis_(aligned_box_.AlignmentAxis()),
          transform_(aligned_box_.getTransform()) {};

    virtual ~TotalVelocityNormVal(){};

    Real reduce(size_t index_i, Real dt = 0.0)
    {
        Vecd frame_velocity = Vecd::Zero();
        frame_velocity[alignment_axis_] = transform_.xformBaseVecToFrame(vel_[index_i])[alignment_axis_];
        return frame_velocity[alignment_axis_];
    }
};

template <class ReduceSumType>
class AreaAverageFlowRate : public ReduceSumType
{
  public:
    explicit AreaAverageFlowRate(BodyAlignedBoxByCell& aligned_box_part, const std::string &body_part_name, Real outlet_area)
          : ReduceSumType(aligned_box_part),
            tansient_flow_rate_(*this->particles_->registerSingleVariable<Real>(body_part_name+"TransientFlowRate")),
            outlet_area_(outlet_area) {};
    virtual ~AreaAverageFlowRate(){};

    virtual Real outputResult(Real reduced_value) override
    {
        Real average_velocity_norm = ReduceSumType::outputResult(reduced_value) / Real(this->getDynamicsIdentifier().SizeOfLoopRange());
        tansient_flow_rate_ = average_velocity_norm * outlet_area_;

        //std::cout << "tansient_flow_rate_ = " << tansient_flow_rate_ << std::endl;
        return tansient_flow_rate_;
    }

  private:
    Real &tansient_flow_rate_;
    Real outlet_area_;
};

using OutletTransientFlowRate = AreaAverageFlowRate<TotalVelocityNormVal>;

class RCRPressureByVel : public BaseLocalDynamics<BodyPartByCell>, public DataDelegateSimple
{
  public:
    explicit RCRPressureByVel(BodyAlignedBoxByCell& aligned_box_part, const std::string &body_part_name)
        : BaseLocalDynamics<BodyPartByCell>(aligned_box_part),
          DataDelegateSimple(aligned_box_part.getSPHBody()),
          body_part_name_(body_part_name),
          R1_(0.0), R2_(0.0), C_(0.0), Q_ave_(0.0), delta_t_(0.0), 
          Q_n_(0.0), Q_0_(0.0), p_n_(0.0), p_0_(0.0),
          transient_flow_rate_(*particles_->getSingleVariableByName<Real>(body_part_name+"TransientFlowRate")) {};
    virtual ~RCRPressureByVel(){};

    void setWindkesselParams(Real R1, Real R2, Real C, Real Q_ave)
    {
        R1_ = R1;
        R2_ = R2;
        C_ = C;
        Q_ave_ = Q_ave;
    }

    void accumulateFlowAndTime(Real dt)
    {
        accumulated_flow_ += transient_flow_rate_ * dt;
        delta_t_ += dt;
    }

    void updateNextPressure()
    {
        Q_n_ = accumulated_flow_ / delta_t_ - Q_ave_;
        //std::cout << "Q_n_ = " << Q_n_ << std::endl;
        //Real dp_dt = - p_0_ / (C_ * R2_) + (R1_ + R2_) * Q_n_ / (C_ * R2_) + R1_ * (Q_n_ - Q_0_) / (delta_t_ + TinyReal);
        //Real p_star = p_0_ + dp_dt * delta_t_;
        //Real dp_dt_star = - p_star / (C_ * R2_) + (R1_ + R2_) * Q_n_ / (C_ * R2_) + R1_ * (Q_n_ - Q_0_) / (delta_t_ + TinyReal);
        //p_n_ = p_0_ + 0.5 * delta_t_ * (dp_dt + dp_dt_star);


        p_n_ = ((Q_n_ * (1.0 + R1_ / R2_) + C_ * R1_ * (Q_n_ - Q_0_) / delta_t_) * delta_t_ / C_ + p_0_) / (1.0 + delta_t_ / (C_ * R2_));
        //std::cout << "p_n_ = " << p_n_ << std::endl;
        writeOutletPressureData();
        writeOutletFlowRateData();

        Q_0_ = Q_n_;
        p_0_ = p_n_;
        accumulated_flow_ = 0;
        delta_t_ = 0;
    }

    Real operator()(Real &p_current)
    {
        return p_n_;
    }

  protected:
    std::string body_part_name_;
    Real R1_, R2_, C_, Q_ave_, delta_t_;
    Real Q_n_, Q_0_;
    Real p_n_, p_0_;
    Real &transient_flow_rate_, accumulated_flow_;

    void writeOutletPressureData()
    {
        std::string output_folder = "./output";
        std::string filefullpath = output_folder + "/" + body_part_name_ + "_outlet_pressure.dat";
        std::ofstream out_file(filefullpath.c_str(), std::ios::app);
        out_file << GlobalStaticVariables::physical_time_ << "   " << p_n_ <<  "\n";
        out_file.close();
    }

    void writeOutletFlowRateData()
    {
        std::string output_folder = "./output";
        std::string filefullpath = output_folder + "/" + body_part_name_ + "_flow_rate.dat";
        std::ofstream out_file(filefullpath.c_str(), std::ios::app);
        out_file << GlobalStaticVariables::physical_time_ << "   " << Q_n_ <<  "\n";
        out_file.close();
    }
};

using WindkesselBoundaryConditionByVel = PressureCondition<RCRPressureByVel>;

} // namespace fluid_dynamics
} // namespace SPH
#endif // PRESSURE_BOUNDARY_H