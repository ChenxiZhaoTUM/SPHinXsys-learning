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
 * @file 	general_reduce.h
 * @brief TBD
 * @author	Chi Zhang and Xiangyu Hu
 */

#ifndef GENERAL_REDUCE_H
#define GENERAL_REDUCE_H

#include "base_general_dynamics.h"
#include "near_wall_boundary.h"
#include <limits>

namespace SPH
{
/**
 * @class VariableNorm
 * @brief  obtained the maximum norm of a variable
 */
template <typename DataType, typename NormType, class DynamicsIdentifier = SPHBody>
class VariableNorm : public BaseLocalDynamicsReduce<NormType, DynamicsIdentifier>,
                     public DataDelegateSimple
{
  public:
    VariableNorm(DynamicsIdentifier &identifier, const std::string &variable_name)
        : BaseLocalDynamicsReduce<NormType, DynamicsIdentifier>(identifier),
          DataDelegateSimple(identifier.getSPHBody()),
          variable_(*particles_->getVariableByName<DataType>(variable_name)){};
    virtual ~VariableNorm(){};
    virtual Real outputResult(Real reduced_value) override { return std::sqrt(reduced_value); }
    Real reduce(size_t index_i, Real dt = 0.0) { return getSquaredNorm(variable_[index_i]); };

  protected:
    StdLargeVec<DataType> &variable_;

    template <typename Datatype>
    Real getSquaredNorm(const Datatype &variable) { return variable.squaredNorm(); };

    Real getSquaredNorm(const Real &variable) { return variable * variable; };
};

/**
 * @class VelocityBoundCheck
 * @brief  check whether particle velocity within a given bound
 */
class VelocityBoundCheck : public LocalDynamicsReduce<ReduceOR>,
                           public DataDelegateSimple
{
  protected:
    StdLargeVec<Vecd> &vel_;
    Real velocity_bound_;

  public:
    VelocityBoundCheck(SPHBody &sph_body, Real velocity_bound);
    virtual ~VelocityBoundCheck(){};

    bool reduce(size_t index_i, Real dt = 0.0);
};

/**
 * @class 	UpperFrontInAxisDirection
 * @brief 	Get the upper front in an axis direction for a body or body part
 */
template <class DynamicsIdentifier>
class UpperFrontInAxisDirection : public BaseLocalDynamicsReduce<ReduceMax, DynamicsIdentifier>,
                                  public DataDelegateSimple
{
  protected:
    int axis_;
    StdLargeVec<Vecd> &pos_;

  public:
    explicit UpperFrontInAxisDirection(DynamicsIdentifier &identifier, const std::string &name, int axis = lastAxis)
        : BaseLocalDynamicsReduce<ReduceMax, DynamicsIdentifier>(identifier),
          DataDelegateSimple(identifier.getSPHBody()), axis_(axis),
          pos_(*particles_->template getVariableByName<Vecd>("Position"))
    {
        this->quantity_name_ = name;
    }
    virtual ~UpperFrontInAxisDirection(){};

    Real reduce(size_t index_i, Real dt = 0.0) { return pos_[index_i][axis_]; };
};

/**
 * @class MaximumSpeed
 * @brief Get the maximum particle speed in a SPH body
 */
class MaximumSpeed : public LocalDynamicsReduce<ReduceMax>,
                     public DataDelegateSimple
{
  protected:
    StdLargeVec<Vecd> &vel_;

  public:
    explicit MaximumSpeed(SPHBody &sph_body);
    virtual ~MaximumSpeed(){};

    Real reduce(size_t index_i, Real dt = 0.0);
};

/**
 * @class	PositionLowerBound
 * @brief	the lower bound of a body by reduced particle positions.
 * 			TODO: a test using this method
 */
class PositionLowerBound : public LocalDynamicsReduce<ReduceLowerBound>,
                           public DataDelegateSimple
{
  protected:
    StdLargeVec<Vecd> &pos_;

  public:
    explicit PositionLowerBound(SPHBody &sph_body);
    virtual ~PositionLowerBound(){};

    Vecd reduce(size_t index_i, Real dt = 0.0);
};

/**
 * @class	PositionUpperBound
 * @brief	the upper bound of a body by reduced particle positions.
 * 			TODO: a test using this method
 */
class PositionUpperBound : public LocalDynamicsReduce<ReduceUpperBound>,
                           public DataDelegateSimple
{
  protected:
    StdLargeVec<Vecd> &pos_;

  public:
    explicit PositionUpperBound(SPHBody &sph_body);
    virtual ~PositionUpperBound(){};

    Vecd reduce(size_t index_i, Real dt = 0.0);
};

/**
 * @class QuantitySummation
 * @brief Compute the summation of  a particle variable in a body
 */
template <typename DataType, class DynamicsIdentifier = SPHBody>
class QuantitySummation : public BaseLocalDynamicsReduce<ReduceSum<DataType>, DynamicsIdentifier>,
                          public DataDelegateSimple
{
  public:
    explicit QuantitySummation(DynamicsIdentifier &identifier, const std::string &variable_name)
        : BaseLocalDynamicsReduce<ReduceSum<DataType>, DynamicsIdentifier>(identifier),
          DataDelegateSimple(identifier.getSPHBody()),
          variable_(*this->particles_->template getVariableByName<DataType>(variable_name))
    {
        this->quantity_name_ = "Total" + variable_name;
    };
    virtual ~QuantitySummation(){};

    DataType reduce(size_t index_i, Real dt = 0.0)
    {
        return variable_[index_i];
    };

  protected:
    StdLargeVec<DataType> &variable_;
};

/**
 * @class QuantityMoment
 * @brief Compute the moment of a body
 */
template <typename DataType, class DynamicsIdentifier>
class QuantityMoment : public QuantitySummation<DataType, DynamicsIdentifier>
{
  protected:
    StdLargeVec<Real> &mass_;

  public:
    explicit QuantityMoment(DynamicsIdentifier &identifier, const std::string &variable_name)
        : QuantitySummation<DataType, DynamicsIdentifier>(identifier, variable_name),
          mass_(*this->particles_->template getVariableByName<Real>("Mass"))
    {
        this->quantity_name_ = variable_name + "Moment";
    };
    virtual ~QuantityMoment(){};

    DataType reduce(size_t index_i, Real dt = 0.0)
    {
        return mass_[index_i] * this->variable_[index_i];
    };
};

class TotalKineticEnergy
    : public LocalDynamicsReduce<ReduceSum<Real>>,
      public DataDelegateSimple
{
  protected:
    StdLargeVec<Real> &mass_;
    StdLargeVec<Vecd> &vel_;

  public:
    explicit TotalKineticEnergy(SPHBody &sph_body);
    virtual ~TotalKineticEnergy(){};
    Real reduce(size_t index_i, Real dt = 0.0);
};

class TotalMechanicalEnergy : public TotalKineticEnergy
{
  protected:
    Gravity &gravity_;
    StdLargeVec<Vecd> &pos_;

  public:
    explicit TotalMechanicalEnergy(SPHBody &sph_body, Gravity &gravity);
    virtual ~TotalMechanicalEnergy(){};
    Real reduce(size_t index_i, Real dt = 0.0);
};

class FluidContactIndication : public LocalDynamics, public DataDelegateSimple
{
	public:
		FluidContactIndication(SPHBody& fluid_body, SPHBody& solid_body)
            : LocalDynamics(fluid_body), DataDelegateSimple(fluid_body),
			pos_(*particles_->getVariableByName<Vecd>("Position")), 
            fluid_contact_indicator_(*this->particles_->template registerSharedVariable<int>("FluidContactIndicator")), 
            solid_body_(solid_body),
			spacing_ref_(sph_body_.sph_adaptation_->ReferenceSpacing()){};
		virtual ~FluidContactIndication() {};

		void update(size_t index_i, Real dt = 0.0)
		{
			Real phi = solid_body_.getInitialShape().findSignedDistance(pos_[index_i]);
			fluid_contact_indicator_[index_i] = 1;
			if (phi > spacing_ref_)
				fluid_contact_indicator_[index_i] = 0;
		}

	protected:
		StdLargeVec<Vecd> &pos_;
		StdLargeVec<int> &fluid_contact_indicator_;
		SPHBody& solid_body_;
		Real spacing_ref_;
};

class FluidSurfaceIndicationByDistance : public fluid_dynamics::DistanceFromWall
{
public: 
    explicit FluidSurfaceIndicationByDistance(BaseContactRelation& wall_contact_relation)
        : DistanceFromWall(wall_contact_relation),
        distance_from_wall_(*particles_->getVariableByName<Vecd>("DistanceFromWall")),
        fluid_contact_indicator_(*this->particles_->template registerSharedVariable<int>("FluidContactIndicator")),
        spacing_ref_(sph_body_.sph_adaptation_->ReferenceSpacing()) {};

    virtual ~FluidSurfaceIndicationByDistance() {};

    void update(size_t index_i, Real dt = 0.0)
    {
        fluid_contact_indicator_[index_i] = 1;
        if (distance_from_wall_[index_i].squaredNorm() > pow(1.05 * spacing_ref_, 2))
            fluid_contact_indicator_[index_i] = 0;
    }

protected:
    StdLargeVec<Vecd> &distance_from_wall_;
    StdLargeVec<int> &fluid_contact_indicator_;
    Real spacing_ref_;
};

class SurfaceKineticEnergy
    : public LocalDynamicsReduce<ReduceSum<Real>>,
      public DataDelegateSimple
{
  protected:
    StdLargeVec<Real> &mass_;
    StdLargeVec<Vecd> &vel_;
	StdLargeVec<int> &fluid_contact_indicator_;
	StdLargeVec<Real> particle_energy_;

  public:
    explicit SurfaceKineticEnergy(SPHBody &sph_body)
          : LocalDynamicsReduce<ReduceSum<Real>>(sph_body),
      DataDelegateSimple(sph_body),
      mass_(*particles_->getVariableByName<Real>("Mass")),
      vel_(*particles_->getVariableByName<Vecd>("Velocity")),
      fluid_contact_indicator_(*particles_->getVariableByName<int>("FluidContactIndicator"))
    {
        quantity_name_ = "SurfaceKineticEnergy";
        particles_->registerVariable(particle_energy_, "ParticleEnergy");
    }
      virtual ~SurfaceKineticEnergy(){};

    Real reduce(size_t index_i, Real dt = 0.0)
    {
        Real particle_energy(0.0);

        if (fluid_contact_indicator_[index_i] == 1)
            particle_energy = 0.5 * mass_[index_i] * vel_[index_i].squaredNorm();
        else
            particle_energy = 0.0;

        particle_energy_[index_i] = particle_energy;

        return particle_energy;
    }
};

class AvgSurfaceKineticEnergy : public DataDelegateSimple
{
  protected:
    StdLargeVec<int> &fluid_contact_indicator_;
    StdLargeVec<Real> &particle_energy_;

  public:
    explicit AvgSurfaceKineticEnergy(SPHBody &sph_body)
        : DataDelegateSimple(sph_body),
          fluid_contact_indicator_(*particles_->getVariableByName<int>("FluidContactIndicator")),
          particle_energy_(*particles_->getVariableByName<Real>("ParticleEnergy")) {};
    virtual ~AvgSurfaceKineticEnergy(){};

    Real getAvgSurfaceKineticEnergy()
    {
        int fluid_surface_particle_num = 0;
        Real average_surface_kinetic_energy = 0;
        for (size_t i = 0; i != particles_->total_real_particles_; ++i)
        {
            if (fluid_contact_indicator_[i] == 1)
            {
                fluid_surface_particle_num++;
                average_surface_kinetic_energy += particle_energy_[i];
            }  
        }
        average_surface_kinetic_energy /= (fluid_surface_particle_num + TinyReal);
        return average_surface_kinetic_energy;
    }

    void writeToFile(size_t iteration_step = 0)
    {
        std::string output_folder = "./output";
        std::string filefullpath = output_folder + "/" +  "average_surface_kinetic_energy.dat";
        std::ofstream out_file(filefullpath.c_str(), std::ios::app);
        out_file << iteration_step << "   ";
        out_file << getAvgSurfaceKineticEnergy();
        out_file << "\n";
        out_file.close();
    };
};

class LocalDisorderMeasure : public LocalDynamics, public DataDelegateInner
{
  public:
    explicit LocalDisorderMeasure(BaseInnerRelation &inner_relation)
        : LocalDynamics(inner_relation.getSPHBody()),
          DataDelegateInner(inner_relation),
          spacing_ref_(sph_body_.sph_adaptation_->ReferenceSpacing()),
          distance_default_(100.0 * spacing_ref_),
          pos_(*particles_->getVariableByName<Vecd>("Position")),
          distance_1st_(*particles_->registerSharedVariable<Real>("FirstDistance")),
          distance_2nd_(*particles_->registerSharedVariable<Real>("SecondDistance")),
          local_disorder_measure_parameter_(*particles_->registerSharedVariable<Real>("LocalDisorderMeasureParameter"))
    {}
    virtual ~LocalDisorderMeasure(){};

    void interaction(size_t index_i, Real dt)
    {
        const Neighborhood &inner_neighborhood = inner_configuration_[index_i];

        Vecd distance_1st_vec = distance_default_ * Vecd::Ones();

        int cone_count_ = 8;
        Real cone_angle_ = 7.0 / 18.0 * Pi;
        std::vector<Vecd> distances_2nd_vec(cone_count_, distance_default_ * Vecd::Ones());

        for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
        {
            size_t index_j = inner_neighborhood.j_[n];

            Vecd temp = pos_[index_i] - pos_[index_j];
            if (temp.squaredNorm() < distance_1st_vec.squaredNorm())
            {
                distance_1st_vec = temp; // more reliable distance
            }

            if (Dimensions == 2)
            {
                Real angle = std::atan2(temp[1], temp[0]);
                if (angle < 0)
                    angle += 2.0 * Pi;
                int cone_index = static_cast<int>(angle / cone_angle_);

                if (temp.squaredNorm() < distances_2nd_vec[cone_index].squaredNorm())
                {
                    distances_2nd_vec[cone_index] = temp;
                }
            }
        }

        Real max_min_distance_2nd = 0.0;
        for (const Vecd &min_dist_vec : distances_2nd_vec)
        {
            if (min_dist_vec.norm() < distance_default_)
            {
                max_min_distance_2nd = std::max(max_min_distance_2nd, min_dist_vec.norm());
            }
        }

        distance_1st_[index_i] = distance_1st_vec.norm();
        distance_2nd_[index_i] = max_min_distance_2nd;

        if (distance_2nd_[index_i] > 0)
            local_disorder_measure_parameter_[index_i] = (distance_2nd_[index_i] - distance_1st_[index_i]) / (distance_2nd_[index_i] + distance_1st_[index_i]);
        else
            local_disorder_measure_parameter_[index_i] = 0;
    }

  protected:
    Real spacing_ref_, distance_default_;
    StdLargeVec<Vecd> &pos_;
    StdLargeVec<Real> &distance_1st_, &distance_2nd_;
    StdLargeVec<Real> &local_disorder_measure_parameter_;
};

class LocalDisorderMeasureForPeriodicCondition : public LocalDynamics, public DataDelegateInner
{
  public:
    explicit LocalDisorderMeasureForPeriodicCondition(BaseInnerRelation &inner_relation, Real width, Real height)
        : LocalDynamics(inner_relation.getSPHBody()),
          DataDelegateInner(inner_relation),
          spacing_ref_(sph_body_.sph_adaptation_->ReferenceSpacing()),
          distance_default_(100.0 * spacing_ref_),
          pos_(*particles_->getVariableByName<Vecd>("Position")),
          distance_1st_(*particles_->registerSharedVariable<Real>("FirstDistance")),
          distance_2nd_(*particles_->registerSharedVariable<Real>("SecondDistance")),
          local_disorder_measure_parameter_(*particles_->registerSharedVariable<Real>("LocalDisorderMeasureParameter")),
          width_(width), height_(height){};
    virtual ~LocalDisorderMeasureForPeriodicCondition(){};

    void interaction(size_t index_i, Real dt)
    {
        const Neighborhood &inner_neighborhood = inner_configuration_[index_i];

        Vecd distance_1st_vec = distance_default_ * Vecd::Ones();

        int cone_count_ = 8;
        Real cone_angle_ = 7.0 / 18.0 * Pi;
        std::vector<Vecd> distances_2nd_vec(cone_count_, distance_default_ * Vecd::Ones());

        for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
        {
            size_t index_j = inner_neighborhood.j_[n];

            Vecd temp = pos_[index_i] - pos_[index_j];
            if (temp.squaredNorm() < distance_1st_vec.squaredNorm())
            {
                distance_1st_vec = temp; // more reliable distance
            }

            if (Dimensions == 2)
            {
                Real angle = std::atan2(temp[1], temp[0]);
                if (angle < 0)
                    angle += 2.0 * Pi;
                int cone_index = static_cast<int>(angle / cone_angle_);

                if (temp.squaredNorm() < distances_2nd_vec[cone_index].squaredNorm())
                {
                    if (temp[0] > 0.5 * width_)
                    {
                        temp[0] -= width_;
                    }
                    else if (temp[0] < -0.5 * width_)
                    {
                        temp[0] += width_;
                    }

                    if (temp[1] > 0.5 * height_)
                    {
                        temp[1] -= height_;
                    }
                    else if (temp[1] < -0.5 * height_)
                    {
                        temp[1] += height_;
                    }

                    distances_2nd_vec[cone_index] = temp;
                }
            }
        }

        Real max_min_distance_2nd = 0.0;
        for (const Vecd &min_dist_vec : distances_2nd_vec)
        {
            if (min_dist_vec.norm() < distance_default_)
            {
                max_min_distance_2nd = std::max(max_min_distance_2nd, min_dist_vec.norm());
            }
        }

        distance_1st_[index_i] = distance_1st_vec.norm();
        distance_2nd_[index_i] = max_min_distance_2nd;

        if (distance_2nd_[index_i] > 0)
            local_disorder_measure_parameter_[index_i] = (distance_2nd_[index_i] - distance_1st_[index_i]) / (distance_2nd_[index_i] + distance_1st_[index_i]);
        else
            local_disorder_measure_parameter_[index_i] = 0;
    }

  protected:
    Real spacing_ref_, distance_default_;
    StdLargeVec<Vecd> &pos_;
    StdLargeVec<Real> &distance_1st_, &distance_2nd_;
    StdLargeVec<Real> &local_disorder_measure_parameter_;
    Real width_, height_;
};

class GlobalDisorderMeasure : public DataDelegateSimple
{
  private:
    StdLargeVec<Real> &local_disorder_measure_parameter_;

  public:
    explicit GlobalDisorderMeasure(SPHBody &sph_body)
        : DataDelegateSimple(sph_body),
          local_disorder_measure_parameter_(*particles_->getVariableByName<Real>("LocalDisorderMeasureParameter")) {};
    virtual ~GlobalDisorderMeasure(){};

    Real getGlobalDisorderMeasureValue()
    {
        Real global_disorder_measure_parameter = 0;
        for (size_t i = 0; i != particles_->total_real_particles_; ++i)
        {
            global_disorder_measure_parameter += local_disorder_measure_parameter_[i];
        }
        global_disorder_measure_parameter /= (particles_->total_real_particles_ + TinyReal);
        return global_disorder_measure_parameter;
    }

    void writeToFile(size_t iteration_step = 0)
    {
        std::string output_folder = "./output";
        std::string filefullpath = output_folder + "/" + "gobal_disorder_measure.dat";
        std::ofstream out_file(filefullpath.c_str(), std::ios::app);
        out_file << iteration_step << "   ";
        out_file << getGlobalDisorderMeasureValue();
        out_file << "\n";
        out_file.close();
    };
};
} // namespace SPH
#endif // GENERAL_REDUCE_H
