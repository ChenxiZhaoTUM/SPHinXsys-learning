//#include "sphinxsys.h"
//using namespace SPH;
//
//
//class ZeroOrderConsistencyInteraction : public LocalDynamics, public DataDelegateInner
//{
//      public:
//        explicit ZeroOrderConsistencyInteraction(BaseInnerRelation &inner_relation)
//            : LocalDynamics(inner_relation.getSPHBody()), DataDelegateInner(inner_relation),
//              pos_(*particles_->getVariableByName<Vecd>("Position")), sph_adaptation_(sph_body_.sph_adaptation_)
//        {
//                particles_->registerVariable(zero_order_consistency_value_, "ZeroOrderConsistencyValue");
//                level_set_shape_ = DynamicCast<LevelSetShape>(this, sph_body_.body_shape_);
//        }
//
//        virtual ~ZeroOrderConsistencyInteraction(){};
//
//        inline void interaction(size_t index_i, Real dt = 0.0)
//        {
//                Vecd sum_temp = Vecd::Zero();
//
//                Neighborhood &inner_neighborhood = inner_configuration_[index_i];
//
//                for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
//                {
//                        sum_temp += inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];
//                }
//
//                zero_order_consistency_value_[index_i] = sum_temp;
//                zero_order_consistency_value_[index_i] += level_set_shape_->computeKernelGradientIntegral(
//                    pos_[index_i], sph_adaptation_->SmoothingLengthRatio(index_i));
//        };
//
//      protected:
//        StdLargeVec<Vecd> zero_order_consistency_value_;
//        StdLargeVec<Vecd> &pos_;
//        LevelSetShape *level_set_shape_;
//        SPHAdaptation *sph_adaptation_;
//};
//
//typedef DataDelegateComplex<BaseParticles, BaseParticles> ConsistencyDataDelegateComplex;
//
//class ZeroOrderConsistencyInteractionComplex : public LocalDynamics, public ConsistencyDataDelegateComplex
//{
//      public:
//        explicit ZeroOrderConsistencyInteractionComplex(ComplexRelation &complex_relation, const std::string &shape_name)
//            : LocalDynamics(complex_relation.getSPHBody()), ConsistencyDataDelegateComplex(complex_relation),
//              pos_(*particles_->getVariableByName<Vecd>("Position")), sph_adaptation_(sph_body_.sph_adaptation_)
//        {
//                particles_->registerVariable(zero_order_consistency_value_, "ZeroOrderConsistencyValue");
//                
//				ComplexShape &complex_shape = DynamicCast<ComplexShape>(this, *sph_body_.body_shape_);
//                level_set_shape_ = DynamicCast<LevelSetShape>(this, complex_shape.getShapeByName(shape_name));
//        }
//
//        virtual ~ZeroOrderConsistencyInteractionComplex(){};
//
//        inline void interaction(size_t index_i, Real dt = 0.0)
//        {
//                Vecd sum_temp = Vecd::Zero();
//
//                Neighborhood &inner_neighborhood = inner_configuration_[index_i];
//                for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
//                {
//                        sum_temp += inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.e_ij_[n];
//                }
//
//				/** Contact interaction. */
//                for (size_t k = 0; k < contact_configuration_.size(); ++k)
//                {
//                        Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
//                        for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
//                        {
//                                sum_temp += contact_neighborhood.dW_ijV_j_[n] * contact_neighborhood.e_ij_[n];
//                        }
//                }
//
//                zero_order_consistency_value_[index_i] = sum_temp;
//                zero_order_consistency_value_[index_i] += level_set_shape_->computeKernelGradientIntegral(
//                    pos_[index_i], sph_adaptation_->SmoothingLengthRatio(index_i));
//        };
//
//      protected:
//        StdLargeVec<Vecd> zero_order_consistency_value_;
//        StdLargeVec<Vecd> &pos_;
//        LevelSetShape *level_set_shape_;
//        SPHAdaptation *sph_adaptation_;
//};
//
//typedef DataDelegateComplex<BaseParticles, BaseParticles> FSIComplexData;
//
//class FluidSurfaceIndication : public LocalDynamics, public FSIComplexData
//{
//  public:
//	  explicit FluidSurfaceIndication(ComplexRelation &complex_relation, Real threshold = 0.75)
//		  : LocalDynamics(complex_relation.getSPHBody()), FSIComplexData(complex_relation),
//		  threshold_by_dimensions_(threshold* (Real)Dimensions),
//		  indicator_(*particles_->getVariableByName<int>("Indicator")),
//		  smoothing_length_(complex_relation.getSPHBody().sph_adaptation_->ReferenceSmoothingLength())
//	  {
//		  particles_->registerVariable(pos_div_, "PositionDivergence");
//	  }
//
//    virtual ~FluidSurfaceIndication(){};
//
//	inline void interaction(size_t index_i, Real dt = 0.0)
//	{
//		Real pos_div = 0.0;
//		const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
//		for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
//		{
//			pos_div -= inner_neighborhood.dW_ijV_j_[n] * inner_neighborhood.r_ij_[n];
//		}
//		
//
//		for (size_t k = 0; k < contact_configuration_.size(); ++k)
//        {
//            Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
//			if (contact_neighborhood.current_size_ == 0)
//			{
//				pos_div = 2 * threshold_by_dimensions_;
//			}
//        }
//
//		pos_div_[index_i] = pos_div;
//	}
//
//	void update(size_t index_i, Real dt = 0.0)
//	{
//		indicator_[index_i] = 1;
//		if (pos_div_[index_i] > threshold_by_dimensions_)
//			indicator_[index_i] = 0;
//	}
//
//  protected:
//    Real threshold_by_dimensions_;
//    StdLargeVec<int> &indicator_;
//    StdLargeVec<Real> pos_div_;
//    Real smoothing_length_;
//};
//
//
//class FluidSurfaceIndicationByDistance : public LocalDynamics, public DataDelegateSimple
//{
//	public:
//		FluidSurfaceIndicationByDistance(SPHBody& fluid_body, SPHBody& solid_body) :
//			LocalDynamics(fluid_body), DataDelegateSimple(fluid_body),
//			pos_(*particles_->getVariableByName<Vecd>("Position")),
//      indicator_(*particles_->getVariableByName<int>("Indicator")), solid_body_(solid_body),
//			particle_spacing_min_(sph_body_.sph_adaptation_->MinimumSpacing()){};
//		virtual ~FluidSurfaceIndicationByDistance() {};
//
//		void update(size_t index_i, Real dt = 0.0)
//		{
//			Real phi = solid_body_.body_shape_->findSignedDistance(pos_[index_i]);
//			indicator_[index_i] = 1;
//			if (phi > particle_spacing_min_)
//				indicator_[index_i] = 0;
//		}
//
//	protected:
//		StdLargeVec<Vecd> &pos_;
//		StdLargeVec<int> &indicator_;
//		SPHBody& solid_body_;
//		Real particle_spacing_min_;
//};
//
//class SurfaceKineticEnergy
//    : public LocalDynamicsReduce<Real, ReduceSum<Real>>,
//      public DataDelegateSimple
//{
//
//  protected:
//    StdLargeVec<Real> &mass_;
//    StdLargeVec<Vecd> &vel_, &pos_;
//	StdLargeVec<int> &indicator_;
//	StdLargeVec<Real> particle_energy_;
//
//  public:
//    SurfaceKineticEnergy(SPHBody &sph_body, const std::string &surface_energy_name)
//		: LocalDynamicsReduce<Real, ReduceSum<Real>>(sph_body, Real(0)),
//		  DataDelegateSimple(sph_body), mass_(particles_->mass_),
//		  vel_(particles_->vel_), pos_(*particles_->getVariableByName<Vecd>("Position")),
//		indicator_(*particles_->getVariableByName<int>("Indicator"))
//	{
//		quantity_name_ = surface_energy_name;
//		particles_->registerVariable(particle_energy_, "ParticleEnergy");
//	}
//    virtual ~SurfaceKineticEnergy(){};
//
//	Real reduce(size_t index_i, Real dt = 0.0)
//	{
//		Real particle_energy(0.0);
//
//		if (indicator_[index_i] == 1)
//			particle_energy = 0.5 * mass_[index_i] * vel_[index_i].squaredNorm();
//		else
//			particle_energy = 0.0;
//
//		particle_energy_[index_i] = particle_energy;
//
//		return particle_energy;
//	}
//};
