#include "sphinxsys.h"
using namespace SPH;

class FuncRelativeError : public LocalDynamics, public GeneralDataDelegateInner, public GeneralDataDelegateContactOnly
{
public:
	FuncRelativeError(ComplexRelation& complex_relation) :
		LocalDynamics(complex_relation.getSPHBody()),
		GeneralDataDelegateInner(complex_relation.getInnerRelation()),
		GeneralDataDelegateContactOnly(complex_relation.getContactRelation()),
		pos_(particles_->pos_), mass_(particles_->mass_), rho_(particles_->rho_)
	{
		particles_->registerVariable(error_, "FunctionRelativeError");

		for (size_t k = 0; k != contact_particles_.size(); ++k)
		{
			contact_mass_.push_back(&(contact_particles_[k]->mass_));
			contact_rho_.push_back(&(contact_particles_[k]->rho_));
			contact_pos_.push_back(&(contact_particles_[k]->pos_));
		}
	}

	virtual ~FuncRelativeError() {};

	void interaction(size_t index_i, Real dt = 0.0)
	{
		Real f_analytical = sin(pos_[index_i][0] * pos_[index_i][0] + pos_[index_i][1] * pos_[index_i][1]);
		Real f_sph = 0;
		const Neighborhood& inner_neighborhood = inner_configuration_[index_i];

		//std::string output_folder = "./output";
		//std::string filefullpath = output_folder + "/" + "neighborSize_" + std::to_string(dt) + ".dat";
		//std::ofstream out_file(filefullpath.c_str(), std::ios::app);
		//out_file << inner_neighborhood.current_size_ << " " << index_i << std::endl;

		for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
		{
			size_t index_j = inner_neighborhood.j_[n];
			Real f_sph_j = sin(pos_[index_j][0] * pos_[index_j][0] + pos_[index_j][1] * pos_[index_j][1]);
			f_sph += f_sph_j * inner_neighborhood.W_ij_[n] * mass_[index_j] / rho_[index_j];
		}

		for (size_t k = 0; k < this->contact_configuration_.size(); ++k)
		{
			StdLargeVec<Real>& contact_mass_k = *(contact_mass_[k]);
			StdLargeVec<Real>& contact_rho_k = *(contact_rho_[k]);
			StdLargeVec<Vecd>& contact_pos_k = *(contact_pos_[k]);

			Neighborhood& contact_neighborhood = (*contact_configuration_[k])[index_i];
			for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
			{
				size_t index_j = contact_neighborhood.j_[n];
				Real f_sph_j = sin(contact_pos_k[index_j][0] * contact_pos_k[index_j][0] + contact_pos_k[index_j][1] * contact_pos_k[index_j][1]);
				f_sph += f_sph_j * contact_neighborhood.W_ij_[n] * contact_mass_k[index_j] / contact_rho_k[index_j];
			}
		}

		error_[index_i] = abs(f_sph - f_analytical) * abs(f_sph - f_analytical) / abs(f_analytical) / abs(f_analytical) * mass_[index_i] / rho_[index_i];
	}

protected:
	StdLargeVec<Real> error_;
	StdLargeVec<Vecd>& pos_;
	StdLargeVec<Real>& mass_, & rho_;
	StdVec<StdLargeVec<Real>*> contact_mass_, contact_rho_;
	StdVec<StdLargeVec<Vecd>*> contact_pos_;
};

class WriteFuncRelativeErrorSum : public BaseIO
{
public:
	WriteFuncRelativeErrorSum(IOEnvironment& io_environment, RealBody& sphbody01, RealBody& sphbody02)
		: BaseIO(io_environment), plt_engine_(), reference_(0.0), error_sum_(0.0),
		sphbody01_complex_(sphbody01, { &sphbody02 }), sphbody02_complex_(sphbody02, { &sphbody01 }) {};

	virtual ~WriteFuncRelativeErrorSum() {};

	virtual void writeToFile(size_t iteration_step = 0) override
	{
		InteractionDynamics<FuncRelativeError> relative_error_for_sphbody01(sphbody01_complex_);
		InteractionDynamics<FuncRelativeError> relative_error_for_sphbody02(sphbody02_complex_);
		ReduceDynamics<QuantitySummation<Real>> compute_relative_error_sum_for_sphbody01(sphbody01_complex_.getSPHBody(), "FunctionRelativeError");
		ReduceDynamics<QuantitySummation<Real>> compute_relative_error_sum_for_sphbody02(sphbody02_complex_.getSPHBody(), "FunctionRelativeError");
		sphbody01_complex_.updateConfiguration();
		sphbody02_complex_.updateConfiguration();
		relative_error_for_sphbody01.exec();
		relative_error_for_sphbody02.exec();
		error_sum_ = sqrt(compute_relative_error_sum_for_sphbody01.exec() + compute_relative_error_sum_for_sphbody02.exec());

		filefullpath_output_ = io_environment_.output_folder_ + "/" + "FunctionRelativeError" + ".dat";
		std::ofstream out_file(filefullpath_output_.c_str(), std::ios::app);
		out_file << "\"run_time\"" << "   ";
		plt_engine_.writeAQuantityHeader(out_file, reference_, "FunctionRelativeError");
		out_file << "\n";
		out_file.close();

		std::ofstream out_file02(filefullpath_output_.c_str(), std::ios::app);
		out_file02 << iteration_step << "   ";
		plt_engine_.writeAQuantity(out_file02, error_sum_);
		out_file02 << "\n";
		out_file02.close();
	};

protected:
	Real error_sum_;
	PltEngine plt_engine_;
	std::string filefullpath_output_;
	Real reference_;
	ComplexRelation sphbody01_complex_;
	ComplexRelation sphbody02_complex_;
};

typedef DataDelegateInner<BaseParticles> ConsistencyDataDelegateInner;

class ZeroOrderConsistencyInteraction : public LocalDynamics, public ConsistencyDataDelegateInner
{
      public:
        explicit ZeroOrderConsistencyInteraction(BaseInnerRelation &inner_relation)
            : LocalDynamics(inner_relation.getSPHBody()), ConsistencyDataDelegateInner(inner_relation),
              pos_(particles_->pos_), sph_adaptation_(sph_body_.sph_adaptation_)
        {
                particles_->registerVariable(zero_order_consistency_value_, "ZeroOrderConsistencyValue");
                level_set_shape_ = DynamicCast<LevelSetShape>(this, sph_body_.body_shape_);
        }

        virtual ~ZeroOrderConsistencyInteraction(){};

        inline void interaction(size_t index_i, Real dt = 0.0)
        {
                Vecd sum_temp = Vecd::Zero();

                Neighborhood &inner_neighborhood = inner_configuration_[index_i];

                for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
                {
                        sum_temp += inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];
                }

                zero_order_consistency_value_[index_i] = sum_temp;
                zero_order_consistency_value_[index_i] += level_set_shape_->computeKernelGradientIntegral(
                    pos_[index_i], sph_adaptation_->SmoothingLengthRatio(index_i));
        };

      protected:
        StdLargeVec<Vecd> zero_order_consistency_value_;
        StdLargeVec<Vecd> &pos_;
        LevelSetShape *level_set_shape_;
        SPHAdaptation *sph_adaptation_;
};

typedef DataDelegateComplex<BaseParticles, BaseParticles> ConsistencyDataDelegateComplex;

class ZeroOrderConsistencyInteractionComplex : public LocalDynamics, public ConsistencyDataDelegateComplex
{
      public:
        explicit ZeroOrderConsistencyInteractionComplex(ComplexRelation &complex_relation, const std::string &shape_name)
            : LocalDynamics(complex_relation.getSPHBody()), ConsistencyDataDelegateComplex(complex_relation),
              pos_(particles_->pos_), sph_adaptation_(sph_body_.sph_adaptation_)
        {
                particles_->registerVariable(zero_order_consistency_value_, "ZeroOrderConsistencyValue");
                
				ComplexShape &complex_shape = DynamicCast<ComplexShape>(this, *sph_body_.body_shape_);
                level_set_shape_ = DynamicCast<LevelSetShape>(this, complex_shape.getShapeByName(shape_name));
        }

        virtual ~ZeroOrderConsistencyInteractionComplex(){};

        inline void interaction(size_t index_i, Real dt = 0.0)
        {
                Vecd sum_temp = Vecd::Zero();

                Neighborhood &inner_neighborhood = inner_configuration_[index_i];
                for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
                {
                        sum_temp += inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];
                }

				/** Contact interaction. */
                for (size_t k = 0; k < contact_configuration_.size(); ++k)
                {
                        Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
                        for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
                        {
                                sum_temp += contact_neighborhood.dW_ijV_j_[n] * contact_neighborhood.e_ij_[n];
                        }
                }

                zero_order_consistency_value_[index_i] = sum_temp;
                zero_order_consistency_value_[index_i] += level_set_shape_->computeKernelGradientIntegral(
                    pos_[index_i], sph_adaptation_->SmoothingLengthRatio(index_i));
        };

      protected:
        StdLargeVec<Vecd> zero_order_consistency_value_;
        StdLargeVec<Vecd> &pos_;
        LevelSetShape *level_set_shape_;
        SPHAdaptation *sph_adaptation_;
};

typedef DataDelegateComplex<BaseParticles, BaseParticles> FSIComplexData;

class FluidSurfaceIndication : public LocalDynamics, public FSIComplexData
{
  public:
	  explicit FluidSurfaceIndication(ComplexRelation &complex_relation, Real threshold = 0.75)
		  : LocalDynamics(complex_relation.getSPHBody()), FSIComplexData(complex_relation),
		  threshold_by_dimensions_(threshold* (Real)Dimensions),
		  indicator_(*particles_->getVariableByName<int>("Indicator")),
		  smoothing_length_(complex_relation.getSPHBody().sph_adaptation_->ReferenceSmoothingLength())
	  {
		  particles_->registerVariable(pos_div_, "PositionDivergence");
	  }

    virtual ~FluidSurfaceIndication(){};

	inline void interaction(size_t index_i, Real dt = 0.0)
	{
		Real pos_div = 0.0;
		const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
		for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
		{
			pos_div -= inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.r_ij_[n];
		}
		

		for (size_t k = 0; k < contact_configuration_.size(); ++k)
        {
            Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
			if (contact_neighborhood.current_size_ == 0)
			{
				pos_div = 2 * threshold_by_dimensions_;
			}
        }

		pos_div_[index_i] = pos_div;
	}

	void update(size_t index_i, Real dt = 0.0)
	{
		indicator_[index_i] = 1;
		if (pos_div_[index_i] > threshold_by_dimensions_)
			indicator_[index_i] = 0;
	}

  protected:
    Real threshold_by_dimensions_;
    StdLargeVec<int> &indicator_;
    StdLargeVec<Real> pos_div_;
    Real smoothing_length_;
};


class FluidSurfaceIndicationByDistance : public LocalDynamics, public GeneralDataDelegateSimple
{
	public:
		FluidSurfaceIndicationByDistance(SPHBody& fluid_body, SPHBody& solid_body) :
			LocalDynamics(fluid_body), GeneralDataDelegateSimple(fluid_body),
			pos_(particles_->pos_), indicator_(*particles_->getVariableByName<int>("Indicator")), solid_body_(solid_body),
			particle_spacing_min_(sph_body_.sph_adaptation_->MinimumSpacing()){};
		virtual ~FluidSurfaceIndicationByDistance() {};

		void update(size_t index_i, Real dt = 0.0)
		{
			Real phi = solid_body_.body_shape_->findSignedDistance(pos_[index_i]);
			indicator_[index_i] = 1;
			if (phi > particle_spacing_min_)
				indicator_[index_i] = 0;
		}

	protected:
		StdLargeVec<Vecd> &pos_;
		StdLargeVec<int> &indicator_;
		SPHBody& solid_body_;
		Real particle_spacing_min_;
};

class SurfaceKineticEnergy
    : public LocalDynamicsReduce<Real, ReduceSum<Real>>,
      public GeneralDataDelegateSimple
{

  protected:
    StdLargeVec<Real> &mass_;
    StdLargeVec<Vecd> &vel_, &pos_;
	StdLargeVec<int> &indicator_;
	StdLargeVec<Real> particle_energy_;

  public:
    SurfaceKineticEnergy(SPHBody &sph_body, const std::string &surface_energy_name)
		: LocalDynamicsReduce<Real, ReduceSum<Real>>(sph_body, Real(0)),
		  GeneralDataDelegateSimple(sph_body), mass_(particles_->mass_),
		  vel_(particles_->vel_), pos_(particles_->pos_),
		indicator_(*particles_->getVariableByName<int>("Indicator"))
	{
		quantity_name_ = surface_energy_name;
		particles_->registerVariable(particle_energy_, "ParticleEnergy");
	}
    virtual ~SurfaceKineticEnergy(){};

	Real reduce(size_t index_i, Real dt = 0.0)
	{
		Real particle_energy(0.0);

		if (indicator_[index_i] == 1)
			particle_energy = 0.5 * mass_[index_i] * vel_[index_i].squaredNorm();
		else
			particle_energy = 0.0;

		particle_energy_[index_i] = particle_energy;

		return particle_energy;
	}
};
