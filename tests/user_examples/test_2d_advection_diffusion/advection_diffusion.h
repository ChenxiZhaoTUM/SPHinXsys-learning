/**
 * @file 	test_2d_advection_diffusion.h
 * @brief 	This is the head files used by test_2d_advection_diffusion.cpp.
 * @author	Chenxi Zhao and Xiangyu Hu
 */
#ifndef	ADVECTION_DIFFUSION_H
#define ADVECTION_DIFFUSION_H

#include "sphinxsys.h"
using namespace SPH;
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Vec2d insert_circle_center(0.0, 0.0);
Real insert_in_circle_radius = 0.75;
Real insert_out_circle_radius = 1.0;
Real insert_outer_wall_circle_radius = 1.2;
Real resolution_ref = 0.02;
BoundingBox system_domain_bounds(Vec2d(-1.2, -1.2), Vec2d(1.2, 1.2));
StdVec<Vecd> observation_location = { Vecd(0.8, 0.5) };
//----------------------------------------------------------------------
//	Basic parameters for material properties.
//----------------------------------------------------------------------
Real diff_cf_A_aqueous = 1.0;
Real diff_cf_B_aqueous = 1.0;
Real k_A_ad = 3.95e-1;
Real k_A_de = 1.44e-3;
Real k_B_ad = 3.95e-1;
Real k_B_de = 1.44e-3;
Real adsorption_sites_A = 2;
Real adsorption_sites_B = 3;
Real Y_A_max = 3.32e-4;
Real Y_B_max = 3.3e-11;
//----------------------------------------------------------------------
//	Initial and boundary conditions.
//----------------------------------------------------------------------
Real initial_XA_bc = 2.12e-3;
Real initial_XB_bc = 4.39e-12;
Real initial_XA_aqueous = 0.0;
Real initial_XB_aqueous = 0.0;
Real initial_YA_bc = 0.0;
Real initial_YB_bc = 0.0;
//----------------------------------------------------------------------
//	Geometric shapes used in the system.
//----------------------------------------------------------------------
class AqueousDomain : public ComplexShape
{
public:
	explicit AqueousDomain(const std::string& shape_name) : ComplexShape(shape_name)
	{
		MultiPolygon multi_polygon;
		multi_polygon.addACircle(insert_circle_center, insert_out_circle_radius, 100, ShapeBooleanOps::add);
		multi_polygon.addACircle(insert_circle_center, insert_in_circle_radius, 100, ShapeBooleanOps::sub);
		add<MultiPolygonShape>(multi_polygon);
	}
};

class InnerWall : public MultiPolygonShape
{
public:
	explicit InnerWall(const std::string& shape_name) : MultiPolygonShape(shape_name)
	{
		multi_polygon_.addACircle(insert_circle_center, insert_in_circle_radius, 100, ShapeBooleanOps::add);
	}
};

class OuterWall : public ComplexShape
{
public:
	explicit OuterWall(const std::string& shape_name) : ComplexShape(shape_name)
	{
		MultiPolygon multi_polygon;
		multi_polygon.addACircle(insert_circle_center, insert_outer_wall_circle_radius, 100, ShapeBooleanOps::add);
		multi_polygon.addACircle(insert_circle_center, insert_out_circle_radius, 100, ShapeBooleanOps::sub);
		add<MultiPolygonShape>(multi_polygon);
	}
};
//----------------------------------------------------------------------
//	Define reaction type 
//----------------------------------------------------------------------
class AqueousSpeciesReaction : public BaseReactionModel<2>
{
protected:
	size_t X_A_index_; //index of XA
	size_t X_B_index_; //index of XB

	virtual Real getProductionRateXA(LocalSpecies& species) = 0;
	virtual Real getLossRateXA(LocalSpecies& species) = 0;
	virtual Real getProductionRateXB(LocalSpecies& species) = 0;
	virtual Real getLossRateXB(LocalSpecies& species) = 0;

public:
	explicit AqueousSpeciesReaction()
		: BaseReactionModel<2>({ "AAqueousConcentration", "BAqueousConcentration"}),
		X_A_index_(species_indexes_map_["AAqueousConcentration"]),
		X_B_index_(species_indexes_map_["BAqueousConcentration"])
	{
		reaction_model_ = "AqueousSpeciesReaction";
		initializeAqueousSpeciesReaction();
	};

	virtual ~AqueousSpeciesReaction() {};

	void initializeAqueousSpeciesReaction()
	{
		get_production_rates_.push_back(std::bind(&AqueousSpeciesReaction::getProductionRateXA, this, _1));
		get_production_rates_.push_back(std::bind(&AqueousSpeciesReaction::getProductionRateXB, this, _1));

		get_loss_rates_.push_back(std::bind(&AqueousSpeciesReaction::getLossRateXA, this, _1));
		get_loss_rates_.push_back(std::bind(&AqueousSpeciesReaction::getLossRateXB, this, _1));
	}
};

class AdsorbedSpeciesReaction : public BaseReactionModel<2>
{
protected:
	size_t Y_A_index_; //index of YA
	size_t Y_B_index_; //index of YB

	virtual Real getProductionRateYA(LocalSpecies& species) = 0;
	virtual Real getLossRateYA(LocalSpecies& species) = 0;
	virtual Real getProductionRateYB(LocalSpecies& species) = 0;
	virtual Real getLossRateYB(LocalSpecies& species) = 0;

public:
	explicit AdsorbedSpeciesReaction()
		: BaseReactionModel<2>({ "AAdsorbedConcentration", "BAdsorbedConcentration"}),
		Y_A_index_(species_indexes_map_["AAdsorbedConcentration"]),
		Y_B_index_(species_indexes_map_["BAdsorbedConcentration"])
	{
		reaction_model_ = "AdsorbedSpeciesReaction";
		initializeAdsorbedSpeciesReaction();
	};

	virtual ~AdsorbedSpeciesReaction() {};

	void initializeAdsorbedSpeciesReaction()
	{
		get_production_rates_.push_back(std::bind(&AdsorbedSpeciesReaction::getProductionRateYA, this, _1));
		get_production_rates_.push_back(std::bind(&AdsorbedSpeciesReaction::getProductionRateYB, this, _1));

		get_loss_rates_.push_back(std::bind(&AdsorbedSpeciesReaction::getLossRateYA, this, _1));
		get_loss_rates_.push_back(std::bind(&AdsorbedSpeciesReaction::getLossRateYB, this, _1));
	}
};

class LangmuirAdsorptionModel : public AqueousSpeciesReaction,
								public AdsorbedSpeciesReaction
{
protected:
	Real k_A_ad_, k_A_de_;  //adsorption and desorption rate coefficients of species A
	Real k_B_ad_, k_B_de_;
	Real adsorption_sites_A_;
	Real adsorption_sites_B_;
	Real Y_A_max_;
	Real Y_B_max_;

	virtual Real getProductionRateXA(LocalSpecies& species) override
	{
		Real X_A = species[X_A_index_];
		Real theta_A = species[Y_A_index_] / Y_A_max_;
		Real theta_B = species[Y_B_index_] / Y_B_max_;
		return k_A_ad_ * X_A * pow(1 - theta_A - theta_B, adsorption_sites_A_);
	}

	virtual Real getLossRateXA(LocalSpecies& species) override
	{
		Real theta_A = species[Y_A_index_] / Y_A_max_;
		return k_A_de_ * pow(theta_A, adsorption_sites_A_);
	}

	virtual Real getProductionRateXB(LocalSpecies& species) override
	{
		Real X_B = species[X_B_index_];
		Real theta_A = species[Y_A_index_] / Y_A_max_;
		Real theta_B = species[Y_B_index_] / Y_B_max_;
		return k_B_ad_ * X_B * pow(1 - theta_A - theta_B, adsorption_sites_B_);
	}

	virtual Real getLossRateXB(LocalSpecies& species) override
	{
		Real theta_B = species[Y_B_index_] / Y_B_max_;
		return k_B_de_ * pow(theta_B, adsorption_sites_B_);
	}

	virtual Real getProductionRateYA(LocalSpecies& species) override
	{
		return getLossRateXA(species);
	}

	virtual Real getLossRateYA(LocalSpecies& species) override
	{
		return getProductionRateXA(species);
	}

	virtual Real getProductionRateYB(LocalSpecies& species) override
	{
		return getLossRateXB(species);
	}

	virtual Real getLossRateYB(LocalSpecies& species) override
	{
		return getProductionRateXB(species);
	}

public:
	explicit LangmuirAdsorptionModel(Real k_A_ad, Real k_A_de, Real k_B_ad, Real k_B_de, Real adsorption_sites_A, Real adsorption_sites_B, Real Y_A_max, Real Y_B_max)
		: AqueousSpeciesReaction(),
		AdsorbedSpeciesReaction(),
		k_A_ad_(k_A_ad), k_A_de_(k_A_de), k_B_ad_(k_B_ad), k_B_de_(k_B_de), adsorption_sites_A_(adsorption_sites_A), adsorption_sites_B_(adsorption_sites_A), Y_A_max_(Y_A_max), Y_B_max_(Y_B_max)
	{
		AqueousSpeciesReaction::reaction_model_ = "LangmuirAdsorptionModel";
		AdsorbedSpeciesReaction::reaction_model_ = "LangmuirAdsorptionModel";
	}
	virtual ~LangmuirAdsorptionModel() {};
};

//----------------------------------------------------------------------
//	Define material type 
//----------------------------------------------------------------------
class AqueousSpecies : public DiffusionReaction<Solid, 2>
{
public:
	AqueousSpecies(SharedPtr<AqueousSpeciesReaction> aqueous_species_reaction_ptr)
		: DiffusionReaction<Solid, 2>({ "AAqueousConcentration", "BAqueousConcentration" }, aqueous_species_reaction_ptr)
	{
		material_type_name_ = "AqueousSpecies";
		initializeAnDiffusion<IsotropicDiffusion>("AAqueousConcentration", "AAqueousConcentration", diff_cf_A_aqueous);
		initializeAnDiffusion<IsotropicDiffusion>("BAqueousConcentration", "BAqueousConcentration", diff_cf_B_aqueous);
	};

	virtual ~AqueousSpecies() {};
};

class AqueousSpeciesNoReaction : public DiffusionReaction<Solid>
{
public:
	AqueousSpeciesNoReaction()
		: DiffusionReaction<Solid>({ "AAqueousConcentration", "BAqueousConcentration" }, SharedPtr<NoReaction>())
	{
		material_type_name_ = "AqueousSpeciesNoReaction";
		initializeAnDiffusion<IsotropicDiffusion>("AAqueousConcentration", "AAqueousConcentration", diff_cf_A_aqueous);
		initializeAnDiffusion<IsotropicDiffusion>("BAqueousConcentration", "BAqueousConcentration", diff_cf_B_aqueous);
	};

	virtual ~AqueousSpeciesNoReaction() {};
};

class AdsorbedSpecies : public DiffusionReaction<Solid, 2>  //diff_cf_A = 0, diff_cf_B = 0
{
public:
	AdsorbedSpecies(SharedPtr<AdsorbedSpeciesReaction> adsorbed_species_reaction_ptr)
		: DiffusionReaction<Solid, 2>({ "AAdsorbedConcentration", "BAdsorbedConcentration" },
			adsorbed_species_reaction_ptr)
	{
		material_type_name_ = "AdsorbedSpecies";
	};
	virtual ~AdsorbedSpecies() {};
};

using AqueousParticles = DiffusionReactionParticles<SolidParticles, AqueousSpecies>;
using InnerWallBoundaryParticles = DiffusionReactionParticles<SolidParticles, AdsorbedSpecies>;
using OuterWallBoundaryParticles = DiffusionReactionParticles<SolidParticles, AqueousSpeciesNoReaction>;
//----------------------------------------------------------------------
//	Application dependent initial condition. 
//----------------------------------------------------------------------
class AqueousInitialCondition
	: public DiffusionReactionInitialCondition<AqueousParticles>
{
protected:
	Real X_A_;
	Real X_B_;

public:
	explicit AqueousInitialCondition(SPHBody& sph_body)
		: DiffusionReactionInitialCondition<AqueousParticles>(sph_body)
	{
		X_A_ = particles_->diffusion_reaction_material_.AllSpeciesIndexMap()["AAqueousConcentration"];
		X_B_ = particles_->diffusion_reaction_material_.AllSpeciesIndexMap()["BAqueousConcentration"];
	};

	void update(size_t index_i, Real dt)
	{
		all_species_[X_A_][index_i] = initial_XA_aqueous;
		all_species_[X_B_][index_i] = initial_XB_aqueous;
	};
};

class InnerWallBoundaryInitialCondition
	: public DiffusionReactionInitialCondition<InnerWallBoundaryParticles>
{
protected:
	Real Y_A_;
	Real Y_B_;

public:
	InnerWallBoundaryInitialCondition(SolidBody& diffusion_body) :
		DiffusionReactionInitialCondition<InnerWallBoundaryParticles>(diffusion_body)
	{
		Y_A_ = particles_->diffusion_reaction_material_.AllSpeciesIndexMap()["AAdsorbedConcentration"];
		Y_B_ = particles_->diffusion_reaction_material_.AllSpeciesIndexMap()["BAdsorbedConcentration"];
	}

	void update(size_t index_i, Real dt)
	{
		all_species_[Y_A_][index_i] = initial_YA_bc;
		all_species_[Y_B_][index_i] = initial_YB_bc;
	}
};

class OuterWallBoundaryInitialCondition
	: public DiffusionReactionInitialCondition<OuterWallBoundaryParticles>
{
protected:
	Real X_A_;
	Real X_B_;

public:
	OuterWallBoundaryInitialCondition(SolidBody& diffusion_body) :
		DiffusionReactionInitialCondition<OuterWallBoundaryParticles>(diffusion_body)
	{
		X_A_ = particles_->diffusion_reaction_material_.AllSpeciesIndexMap()["AAqueousConcentration"];
		X_B_ = particles_->diffusion_reaction_material_.AllSpeciesIndexMap()["BAqueousConcentration"];
	}

	void update(size_t index_i, Real dt)
	{
		all_species_[X_A_][index_i] = initial_XA_bc;
		all_species_[X_B_][index_i] = initial_XB_bc;
	}
};

using AqueousDiffusion = DiffusionRelaxationInner<AqueousParticles>;
using InnerWallBoundary = DiffusionRelaxationDirichlet<AqueousParticles, InnerWallBoundaryParticles>;
using OuterWallBoundary = DiffusionRelaxationDirichlet<AqueousParticles, OuterWallBoundaryParticles>;
//----------------------------------------------------------------------
//	Specify diffusion relaxation method. 
//----------------------------------------------------------------------
class DiffusionRelaxation
	: public DiffusionRelaxationRK2<ComplexInteraction<AqueousDiffusion, OuterWallBoundary>>
{
public:
	explicit DiffusionRelaxation(BaseInnerRelation& inner_relation,
									 BaseContactRelation& contact_outer_wall)
		: DiffusionRelaxationRK2<ComplexInteraction<AqueousDiffusion, OuterWallBoundary>>(
			inner_relation, contact_outer_wall) {};
	virtual ~DiffusionRelaxation() {};
};

template <class ParticlesType, class ContactParticlesType>
class SurfaceReaction : public DataDelegateContact<ParticlesType, ContactParticlesType>
{
protected:
	StdVec<StdVec<StdLargeVec<Real> *>> contact_reactive_species_;

public:

	SurfaceReaction(BaseContactRelation& contact_relation)
	{
		contact_reactive_species_.resize(this->contact_particles_.size());
		for (size_t k = 0; k != this->contact_particles_.size(); ++k)
		{
			StdVec<StdLargeVec<Real>> &contact_reactive_species_k = this->contact_particles_[k]->reactive_species_;
			contact_reactive_species_.push_back(&contact_reactive_species_k);
		}
	}
	virtual ~SurfaceReaction(){};
	inline void interaction(size_t index_i, Real dt = 0.0)
	{
		ParticlesType *particles = this->particles_;
		
		for (size_t k = 0; k < this->contact_configuration_.size(); ++k)
		{
			 Neighborhood &contact_neighborhood = (*this->contact_configuration_[k])[index_i];
			 for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
			 {
				 size_t index_j = contact_neighborhood.j_[n];
				 Real r_ij_ = contact_neighborhood.r_ij_[n];
				 Real dW_ijV_j_ = contact_neighborhood.dW_ijV_j_[n];
				 Vecd &e_ij = contact_neighborhood.e_ij_[n];
				 const Vecd &grad_ijV_j = particles->getKernelGradient(index_i, index_j, dW_ijV_j_, e_ij);


			 }


		}
	
	}

};









using AqueousReactionRelaxationForward =
    SimpleDynamics<ReactionRelaxationForward<AqueousParticles>>;
using AqueousReactionRelaxationBackward =
    SimpleDynamics<ReactionRelaxationBackward<AqueousParticles>>;
using AdsorbedReactionRelaxationForward =
    SimpleDynamics<ReactionRelaxationForward<InnerWallBoundaryParticles>>;
using AdsorbedReactionRelaxationBackward =
    SimpleDynamics<ReactionRelaxationBackward<InnerWallBoundaryParticles>>;
#endif //DIFFUSION_TEST_WITH_NEUMANNBC_H