#ifndef WINDKESSEL_BC_H
#define WINDKESSEL_BC_H

#include "sphinxsys.h"
#include "pressure_boundary.h"
#include "particle_generation_and_detection.h"

namespace SPH
{
namespace fluid_dynamics
{
class TargetOutletPressureWindkessel : public BaseLocalDynamics<BodyPartByCell>
{
  public:
    explicit TargetOutletPressureWindkessel(BodyAlignedBoxByCell& aligned_box_part)
        : BaseLocalDynamics<BodyPartByCell>(aligned_box_part),
          part_id_(aligned_box_part.getPartID()),
          Rp_(0.0), C_(0.0), Rd_(0.0), Q_ave_(0.0), delta_t_(0.0), 
          Q_n_(0.0), Q_0_(0.0), p_n_(80*133.32), p_0_(80*133.32),
          flow_rate_(*(this->particles_->registerSingularVariable<Real>("FlowRate" + std::to_string(part_id_ - 1))->ValueAddress())),
          current_flow_rate_(0.0), previous_flow_rate_(0.0),
          physical_time_(sph_system_.getSystemVariableDataByName<Real>("PhysicalTime")) {};
    virtual ~TargetOutletPressureWindkessel(){};

    void setWindkesselParams(Real Rp, Real C, Real Rd, Real dt, Real Q_ave)
    {
        Rp_ = Rp;
        C_ = C;
        Rd_ = Rd;
        Q_ave_ = Q_ave;
        delta_t_ = dt;
    }

    void updateNextPressure()
    {
        getFlowRate();

        Q_n_ = current_flow_rate_ / delta_t_ - Q_ave_;

        Real dp_dt = - p_0_ / (C_ * Rd_) + (Rp_ + Rd_) * Q_n_ / (C_ * Rd_) + Rp_ * (Q_n_ - Q_0_) / (delta_t_ + TinyReal);
        Real p_star = p_0_ + dp_dt * delta_t_;
        Real dp_dt_star = - p_star / (C_ * Rd_) + (Rp_ + Rd_) * Q_n_ / (C_ * Rd_) + Rp_ * (Q_n_ - Q_0_) / (delta_t_ + TinyReal);
        p_n_ = p_0_ + 0.5 * delta_t_ * (dp_dt + dp_dt_star);

        //p_n_ = ((Q_n_ * (1.0 + Rp_ / Rd_) + C_ * Rp_ * (Q_n_ - Q_0_) / delta_t_) * delta_t_ / C_ + p_0_) / (1.0 + delta_t_ / (C_ * Rd_));


        //p_n_ = ((Rd_ * delta_t_ + Rp_ * delta_t_ + C_ * Rp_ * Rd_) * Q_n_ - C_ * Rp_ * Rd_ * Q_0_ + C_ * Rd_ * p_0_) / (C_ * Rd_ + delta_t_);

        std::cout << "p_n_ = " << p_n_ / 133.32 << " mmHg" << std::endl;

        writeOutletPressureData();
        writeOutletFlowRateData();
    }

    Real operator()(Real p, Real current_time)
    {
        return p_n_ - 80*133.32;
    }

  protected:
    int part_id_;
    Real Rp_, C_, Rd_, Q_ave_, delta_t_;
    Real Q_n_, Q_0_;
    Real p_n_, p_0_;
    Real &flow_rate_, current_flow_rate_, previous_flow_rate_;
    Real *physical_time_;

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
        std::string filefullpath = output_folder + "/" + std::to_string(part_id_ - 1) + "_outlet_pressure.dat";
        std::ofstream out_file(filefullpath.c_str(), std::ios::app);
        out_file << *physical_time_ << "   " << p_n_ <<  "\n";
        out_file.close();
    }

    void writeOutletFlowRateData()
    {
        std::string output_folder = "./output";
        std::string filefullpath = output_folder + "/" + std::to_string(part_id_ - 1) + "_flow_rate.dat";
        std::ofstream out_file(filefullpath.c_str(), std::ios::app);
        out_file << *physical_time_ << "   " << Q_n_ <<  "\n";
        out_file.close();
    }
};

class NonPrescribedPressureForFlowRate : public BaseLocalDynamics<BodyPartByCell>
{
  public:
    explicit NonPrescribedPressureForFlowRate(BodyAlignedBoxByCell& aligned_box_part)
        : BaseLocalDynamics<BodyPartByCell>(aligned_box_part),
          part_id_(aligned_box_part.getPartID()),
          Q_n_(0.0), Q_0_(0.0),
          flow_rate_(*(this->particles_->registerSingularVariable<Real>("FlowRate" + std::to_string(part_id_ - 1))->ValueAddress())),
          current_flow_rate_(0.0), previous_flow_rate_(0.0),
          physical_time_(sph_system_.getSystemVariableDataByName<Real>("PhysicalTime")) {};
    virtual ~NonPrescribedPressureForFlowRate(){};


    void writeInletFlowRate(Real delta_t)
    {
        getFlowRate();
        Q_n_ = current_flow_rate_ / delta_t;
        writeOutletFlowRateData();
    }

    Real operator()(Real p, Real current_time)
    {
        return p;
    }

  protected:
    int part_id_;
    Real Q_n_, Q_0_;
    Real &flow_rate_, current_flow_rate_, previous_flow_rate_;
    Real *physical_time_;

    void getFlowRate()
    {
        Q_0_ = Q_n_;
        current_flow_rate_ = flow_rate_ - previous_flow_rate_;
        previous_flow_rate_ = flow_rate_;
    }

    void writeOutletFlowRateData()
    {
        std::string output_folder = "./output";
        std::string filefullpath = output_folder + "/" + std::to_string(part_id_ - 1) + "_flow_rate.dat";
        std::ofstream out_file(filefullpath.c_str(), std::ios::app);
        out_file << *physical_time_ << "   " << Q_n_ <<  "\n";
        out_file.close();
    }
};

using WindkesselBoundaryCondition = PressureCondition<TargetOutletPressureWindkessel>;

template <typename TargetPressure, class ExecutionPolicy = ParallelPolicy>
class BidirectionalBufferWindkessel
{
  protected:
    TargetPressure target_pressure_;

    class TagBufferParticles : public BaseLocalDynamics<BodyPartByCell>
    {
      public:
        TagBufferParticles(BodyAlignedBoxByCell &aligned_box_part)
            : BaseLocalDynamics<BodyPartByCell>(aligned_box_part),
              part_id_(aligned_box_part.getPartID()),
              pos_(particles_->getVariableDataByName<Vecd>("Position")),
              aligned_box_(aligned_box_part.getAlignedBoxShape()),
              buffer_particle_indicator_(particles_->registerStateVariable<int>("BufferParticleIndicator"))
        {
            particles_->addVariableToSort<int>("BufferParticleIndicator");
        };
        virtual ~TagBufferParticles() {};

        virtual void update(size_t index_i, Real dt = 0.0)
        {
            if (aligned_box_.checkInBounds(pos_[index_i]))
            {
                buffer_particle_indicator_[index_i] = part_id_;
            }
        };

      protected:
        int part_id_;
        Vecd *pos_;
        AlignedBoxShape &aligned_box_;
        int *buffer_particle_indicator_;
    };

    class Injection : public BaseLocalDynamics<BodyPartByCell>
    {
      public:
        Injection(BodyAlignedBoxByCell &aligned_box_part, ParticleBuffer<Base> &particle_buffer,
                  TargetPressure &target_pressure)
            : BaseLocalDynamics<BodyPartByCell>(aligned_box_part),
              part_id_(aligned_box_part.getPartID()),
              particle_buffer_(particle_buffer),
              aligned_box_(aligned_box_part.getAlignedBoxShape()),
              fluid_(DynamicCast<Fluid>(this, particles_->getBaseMaterial())),
              pos_(particles_->getVariableDataByName<Vecd>("Position")),
              rho_(particles_->getVariableDataByName<Real>("Density")),
              p_(particles_->getVariableDataByName<Real>("Pressure")),
              Vol_(particles_->getVariableDataByName<Real>("VolumetricMeasure")),
              previous_surface_indicator_(particles_->getVariableDataByName<int>("PreviousSurfaceIndicator")),
              buffer_particle_indicator_(particles_->getVariableDataByName<int>("BufferParticleIndicator")),
              upper_bound_fringe_(0.5 * sph_body_.getSPHBodyResolutionRef()),
              physical_time_(sph_system_.getSystemVariableDataByName<Real>("PhysicalTime")),
              target_pressure_(target_pressure),
              flow_rate_(*(this->particles_->template getSingularVariableByName<Real>("FlowRate" + std::to_string(part_id_ - 1))->ValueAddress()))
        {
            particle_buffer_.checkParticlesReserved();
        };
        virtual ~Injection() {};

        void update(size_t index_i, Real dt = 0.0)
        {
            if (!aligned_box_.checkInBounds(pos_[index_i]))
            {
                if (aligned_box_.checkUpperBound(pos_[index_i], upper_bound_fringe_) &&
                    buffer_particle_indicator_[index_i] == part_id_ &&
                    index_i < particles_->TotalRealParticles())
                {
                    mutex_switch.lock();
                    particle_buffer_.checkEnoughBuffer(*particles_);
                    size_t new_particle_index = particles_->createRealParticleFrom(index_i);
                    buffer_particle_indicator_[new_particle_index] = 0;

                    /** Periodic bounding. */
                    pos_[index_i] = aligned_box_.getUpperPeriodic(pos_[index_i]);
                    Real sound_speed = fluid_.getSoundSpeed(rho_[index_i]);
                    p_[index_i] = target_pressure_(p_[index_i], *physical_time_);
                    rho_[index_i] = p_[index_i] / pow(sound_speed, 2.0) + fluid_.ReferenceDensity();
                    previous_surface_indicator_[index_i] = 1;
                    mutex_switch.unlock();

                    flow_rate_ -= Vol_[index_i];
                }
            }
        }

      protected:
        int part_id_;
        std::mutex mutex_switch;
        ParticleBuffer<Base> &particle_buffer_;
        AlignedBoxShape &aligned_box_;
        Fluid &fluid_;
        Vecd *pos_;
        Real *rho_, *p_, *Vol_;
        int *previous_surface_indicator_, *buffer_particle_indicator_;
        Real upper_bound_fringe_;
        Real *physical_time_;
        Real &flow_rate_;

      private:
        TargetPressure &target_pressure_;
    };

    class Deletion : public BaseLocalDynamics<BodyPartByCell>
    {
      public:
        Deletion(BodyAlignedBoxByCell &aligned_box_part)
            : BaseLocalDynamics<BodyPartByCell>(aligned_box_part),
              part_id_(aligned_box_part.getPartID()),
              aligned_box_(aligned_box_part.getAlignedBoxShape()),
              pos_(particles_->getVariableDataByName<Vecd>("Position")),
              Vol_(particles_->getVariableDataByName<Real>("VolumetricMeasure")),
              buffer_particle_indicator_(particles_->getVariableDataByName<int>("BufferParticleIndicator")),
              flow_rate_(*(this->particles_->template getSingularVariableByName<Real>("FlowRate" + std::to_string(part_id_ - 1))->ValueAddress())) {};
        virtual ~Deletion() {};

        void update(size_t index_i, Real dt = 0.0)
        {
            if (!aligned_box_.checkInBounds(pos_[index_i]))
            {
                mutex_switch.lock();
                while (aligned_box_.checkLowerBound(pos_[index_i]) &&
                       buffer_particle_indicator_[index_i] == part_id_ &&
                       index_i < particles_->TotalRealParticles())
                {
                    particles_->switchToBufferParticle(index_i);
                    flow_rate_ += Vol_[index_i];
                }
                mutex_switch.unlock();
            }
        }

      protected:
        int part_id_;
        std::mutex mutex_switch;
        AlignedBoxShape &aligned_box_;
        Vecd *pos_;
        Real *Vol_;
        int *buffer_particle_indicator_;
        Real &flow_rate_;
    };

  public:
    BidirectionalBufferWindkessel(BodyAlignedBoxByCell &aligned_box_part, ParticleBuffer<Base> &particle_buffer)
        : target_pressure_(TargetPressure(aligned_box_part)),
          tag_buffer_particles(aligned_box_part),
          injection(aligned_box_part, particle_buffer, target_pressure_),
          deletion(aligned_box_part) {};
    virtual ~BidirectionalBufferWindkessel() {};

    SimpleDynamics<TagBufferParticles, ExecutionPolicy> tag_buffer_particles;
    SimpleDynamics<Injection, ExecutionPolicy> injection;
    SimpleDynamics<Deletion, ExecutionPolicy> deletion;
};

using WindkesselOutletBidirectionalBuffer = BidirectionalBufferWindkessel<TargetOutletPressureWindkessel>;

class TotalVelocityNormVal
    : public BaseLocalDynamicsReduce<ReduceSum<Real>, BodyPartByCell>
{
  protected:
    Vecd *vel_;
    AlignedBoxShape &aligned_box_;
    const int alignment_axis_;
    Transform &transform_;

  public:
      explicit TotalVelocityNormVal(BodyAlignedBoxByCell& aligned_box_part)
          : BaseLocalDynamicsReduce<ReduceSum<Real>, BodyPartByCell>(aligned_box_part),
          vel_(this->particles_->template getVariableDataByName<Vecd>("Velocity")),
          aligned_box_(aligned_box_part.getAlignedBoxShape()),
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
    explicit AreaAverageFlowRate(BodyAlignedBoxByCell& aligned_box_part, Real outlet_area)
          : ReduceSumType(aligned_box_part),
            part_id_(aligned_box_part.getPartID()),
            tansient_flow_rate_(*(this->particles_->template registerSingularVariable<Real>("TransientFlowRate" + std::to_string(part_id_ - 1))->ValueAddress())),
            outlet_area_(outlet_area), physical_time_(this->sph_system_.template getSystemVariableDataByName<Real>("PhysicalTime")) {};
    virtual ~AreaAverageFlowRate(){};

    virtual Real outputResult(Real reduced_value) override
    {
        Real average_velocity_norm = ReduceSumType::outputResult(reduced_value) / Real(this->getDynamicsIdentifier().SizeOfLoopRange());
        tansient_flow_rate_ = average_velocity_norm * outlet_area_;

        std::string output_folder = "./output";
        std::string filefullpath = output_folder + "/" + std::to_string(part_id_ - 1) + "_flow_rate.dat";
        std::ofstream out_file(filefullpath.c_str(), std::ios::app);
        out_file << *physical_time_ << "   " << tansient_flow_rate_ <<  "\n";
        out_file.close();

        //std::cout << "tansient_flow_rate_ = " << tansient_flow_rate_ << std::endl;
        return tansient_flow_rate_;
    }

  private:
    int part_id_;
    Real &tansient_flow_rate_;
    Real outlet_area_;
    Real *physical_time_;
};

using SectionTransientFlowRate = AreaAverageFlowRate<TotalVelocityNormVal>;

template <typename AlignedShapeType, typename TargetPressure, class ExecutionPolicy = ParallelPolicy>
class BidirectionalBufferArb
{
  protected:
    TargetPressure target_pressure_;

    class TagBufferParticles : public BaseLocalDynamics<BodyPartByCell>
    {
      public:
        TagBufferParticles(BaseAlignedRegion<BodyRegionByCell, AlignedShapeType>& aligned_region_part)
            : BaseLocalDynamics<BodyPartByCell>(aligned_region_part),
              part_id_(aligned_region_part.getPartID()),
              pos_(particles_->getVariableDataByName<Vecd>("Position")),
              aligned_shape_(aligned_region_part.getAlignedShape()),
              buffer_particle_indicator_(particles_->registerStateVariable<int>("BufferParticleIndicator"))
        {
            particles_->addVariableToSort<int>("BufferParticleIndicator");
        };
        virtual ~TagBufferParticles() {};

        virtual void update(size_t index_i, Real dt = 0.0)
        {
            if (aligned_shape_.checkInBounds(pos_[index_i]))
            {
                buffer_particle_indicator_[index_i] = part_id_;
            }
        };

      protected:
        int part_id_;
        Vecd *pos_;
        AlignedShapeType &aligned_shape_;
        int *buffer_particle_indicator_;
    };

    class Injection : public BaseLocalDynamics<BodyPartByCell>
    {
      public:
        Injection(BaseAlignedRegion<BodyRegionByCell, AlignedShapeType>& aligned_region_part, ParticleBuffer<Base> &particle_buffer,
                  TargetPressure &target_pressure)
            : BaseLocalDynamics<BodyPartByCell>(aligned_region_part),
              part_id_(aligned_region_part.getPartID()),
              particle_buffer_(particle_buffer),
              aligned_shape_(aligned_region_part.getAlignedShape()),
              fluid_(DynamicCast<Fluid>(this, particles_->getBaseMaterial())),
              pos_(particles_->getVariableDataByName<Vecd>("Position")),
              rho_(particles_->getVariableDataByName<Real>("Density")),
              p_(particles_->getVariableDataByName<Real>("Pressure")),
              previous_surface_indicator_(particles_->getVariableDataByName<int>("PreviousSurfaceIndicator")),
              buffer_particle_indicator_(particles_->getVariableDataByName<int>("BufferParticleIndicator")),
              upper_bound_fringe_(0.5 * sph_body_.getSPHBodyResolutionRef()),
              physical_time_(sph_system_.getSystemVariableDataByName<Real>("PhysicalTime")),
              target_pressure_(target_pressure)
        {
            particle_buffer_.checkParticlesReserved();
        };
        virtual ~Injection() {};

        void update(size_t index_i, Real dt = 0.0)
        {
            if (!aligned_shape_.checkInBounds(pos_[index_i]))
            {
                if (aligned_shape_.checkUpperBound(pos_[index_i], upper_bound_fringe_) &&
                    buffer_particle_indicator_[index_i] == part_id_ &&
                    index_i < particles_->TotalRealParticles())
                {
                    mutex_switch.lock();
                    particle_buffer_.checkEnoughBuffer(*particles_);
                    size_t new_particle_index = particles_->createRealParticleFrom(index_i);
                    buffer_particle_indicator_[new_particle_index] = 0;

                    /** Periodic bounding. */
                    pos_[index_i] = aligned_shape_.getUpperPeriodic(pos_[index_i]);
                    Real sound_speed = fluid_.getSoundSpeed(rho_[index_i]);
                    p_[index_i] = target_pressure_(p_[index_i], *physical_time_);
                    rho_[index_i] = p_[index_i] / pow(sound_speed, 2.0) + fluid_.ReferenceDensity();
                    previous_surface_indicator_[index_i] = 1;
                    mutex_switch.unlock();
                }
            }
        }

      protected:
        int part_id_;
        std::mutex mutex_switch;
        ParticleBuffer<Base> &particle_buffer_;
        AlignedShapeType &aligned_shape_;
        Fluid &fluid_;
        Vecd *pos_;
        Real *rho_, *p_;
        int *previous_surface_indicator_, *buffer_particle_indicator_;
        Real upper_bound_fringe_;
        Real *physical_time_;

      private:
        TargetPressure &target_pressure_;
    };

    class Deletion : public BaseLocalDynamics<BodyPartByCell>
    {
      public:
        Deletion(BaseAlignedRegion<BodyRegionByCell, AlignedShapeType>& aligned_region_part)
            : BaseLocalDynamics<BodyPartByCell>(aligned_region_part),
              part_id_(aligned_region_part.getPartID()),
              aligned_shape_(aligned_region_part.getAlignedShape()),
              pos_(particles_->getVariableDataByName<Vecd>("Position")),
              buffer_particle_indicator_(particles_->getVariableDataByName<int>("BufferParticleIndicator")) {};
        virtual ~Deletion() {};

        void update(size_t index_i, Real dt = 0.0)
        {
            if (!aligned_shape_.checkInBounds(pos_[index_i]))
            {
                mutex_switch.lock();
                while (aligned_shape_.checkLowerBound(pos_[index_i]) &&
                       buffer_particle_indicator_[index_i] == part_id_ &&
                       index_i < particles_->TotalRealParticles())
                {
                    particles_->switchToBufferParticle(index_i);
                }
                mutex_switch.unlock();
            }
        }

      protected:
        int part_id_;
        std::mutex mutex_switch;
        AlignedShapeType &aligned_shape_;
        Vecd *pos_;
        int *buffer_particle_indicator_;
    };

  public:
    BidirectionalBufferArb(BaseAlignedRegion<BodyRegionByCell, AlignedShapeType>& aligned_region_part, ParticleBuffer<Base> &particle_buffer)
        : target_pressure_(*this),
          tag_buffer_particles(aligned_region_part),
          injection(aligned_region_part, particle_buffer, target_pressure_),
          deletion(aligned_region_part) {};
    virtual ~BidirectionalBufferArb() {};

    SimpleDynamics<TagBufferParticles, ExecutionPolicy> tag_buffer_particles;
    SimpleDynamics<Injection, ExecutionPolicy> injection;
    SimpleDynamics<Deletion, ExecutionPolicy> deletion;
};

//-------------------------------------------------------------------------------
//	Resistance BC.
//-------------------------------------------------------------------------------
class ResistanceBCPressure : public BaseLocalDynamics<BodyPartByCell>
{
  public:
    explicit ResistanceBCPressure(BodyAlignedBoxByCell &aligned_box_part)
        : BaseLocalDynamics<BodyPartByCell>(aligned_box_part),
          part_id_(aligned_box_part.getPartID()),
          R_(0.0), delta_t_(0.0),
          Q_n_(0.0), p_n_(0.0), p_0_(0.0),
          flow_rate_(*(this->particles_->registerSingularVariable<Real>("FlowRate" + std::to_string(part_id_ - 1))->ValueAddress())),
          current_flow_rate_(0.0), previous_flow_rate_(0.0),
          physical_time_(sph_system_.getSystemVariableDataByName<Real>("PhysicalTime")) {};
    virtual ~ResistanceBCPressure(){};

    void setWindkesselParams(Real R, Real dt, Real p0)
    {
        R_ = R;
        delta_t_ = dt;
        p_0_ = p0;
    }

    void updateNextPressure()
    {
        getFlowRate();

        Q_n_ = current_flow_rate_ / delta_t_;
        p_n_ = p_0_ + R_ * Q_n_;

        writeOutletPressureData();
        writeOutletFlowRateData();
    }

    Real operator()(Real p, Real current_time)
    {
        return p_n_;
    }

  protected:
    int part_id_;
    Real R_, delta_t_;
    Real Q_n_;
    Real p_n_, p_0_;
    Real &flow_rate_, current_flow_rate_, previous_flow_rate_;
    Real *physical_time_;

    void getFlowRate()
    {
        current_flow_rate_ = flow_rate_ - previous_flow_rate_;
        previous_flow_rate_ = flow_rate_;
    }

    void writeOutletPressureData()
    {
        std::string output_folder = "./output";
        std::string filefullpath = output_folder + "/" + std::to_string(part_id_ - 1) + "_outlet_pressure.dat";
        std::ofstream out_file(filefullpath.c_str(), std::ios::app);
        out_file << *physical_time_ << "   " << p_n_ <<  "\n";
        out_file.close();
    }

    void writeOutletFlowRateData()
    {
        std::string output_folder = "./output";
        std::string filefullpath = output_folder + "/" + std::to_string(part_id_ - 1) + "_flow_rate.dat";
        std::ofstream out_file(filefullpath.c_str(), std::ios::app);
        out_file << *physical_time_ << "   " << Q_n_ <<  "\n";
        out_file.close();
    }
};

using ResistanceBoundaryCondition = PressureCondition<ResistanceBCPressure>;
} // namespace fluid_dynamics
} // namespace SPH
#endif // WINDKESSEL_BC_H