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
Real L = 1.0;
Real H = 1.0;
Real resolution_ref = H / 100.0;
Real BW = resolution_ref * 2.0;
BoundingBox system_domain_bounds(Vec2d(-BW, -BW), Vec2d(L + BW, H + BW));
//----------------------------------------------------------------------
//	Basic parameters for material properties.
//----------------------------------------------------------------------
Real diff_cf_A_aqueous = 1;
Real diff_cf_B_aqueous = 1;
Real rho0_f = 1.0;					/**< Density. */
Real U_f = 1.0;						/**< Characteristic velocity. */
Real c_f = 10.0 * U_f;				/**< Speed of sound. */
Real Re = 100.0;					/**< Reynolds number100. */
Real mu_f = rho0_f * U_f * H / Re; /**< Dynamics viscosity. */
//----------------------------------------------------------------------
//	Initial and boundary conditions.
//----------------------------------------------------------------------
Real initial_temperature = 100.0;
Real left_temperature = 300.0;
Real right_temperature = 350.0;
Real k_A_ad;
Real k_A_de;
Real k_B_ad;
Real k_B_de;
Real adsorption_sites_A;
Real adsorption_sites_B;
Real Y_A_max;
Real Y_B_max;
//----------------------------------------------------------------------
//	Geometric shapes used in the system.
//----------------------------------------------------------------------
std::vector<Vecd> createThermalDomain()
{
	std::vector<Vecd> thermalDomainShape;
	thermalDomainShape.push_back(Vecd(0.0, 0.0));
	thermalDomainShape.push_back(Vecd(0.0, H));
	thermalDomainShape.push_back(Vecd(L, H));
	thermalDomainShape.push_back(Vecd(L, 0.0));
	thermalDomainShape.push_back(Vecd(0.0, 0.0));

	return thermalDomainShape;
}

std::vector<Vecd> left_temperature_region
{
	Vecd(0.3 * L, H), Vecd(0.3 * L, H + BW), Vecd(0.4 * L, H + BW),
	Vecd(0.4 * L, H), Vecd(0.3 * L, H)
};

std::vector<Vecd> right_temperature_region
{
	Vecd(0.6 * L, H), Vecd(0.6 * L, H + BW), Vecd(0.7 * L, H + BW),
	Vecd(0.7 * L, H), Vecd(0.6 * L, H)
};

std::vector<Vecd> convection_region
{
	Vecd(0.45 * L, -BW), Vecd(0.45 * L, 0), Vecd(0.55 * L, 0),
	Vecd(0.55 * L, -BW), Vecd(0.45 * L, -BW)
};

//----------------------------------------------------------------------
//	Define SPH bodies. 
//----------------------------------------------------------------------
class DiffusionBody : public MultiPolygonShape
{
public:
	explicit DiffusionBody(const std::string& shape_name) : MultiPolygonShape(shape_name)
	{
		multi_polygon_.addAPolygon(createThermalDomain(), ShapeBooleanOps::add);
	}
};

class DirichletWallBoundary : public MultiPolygonShape
{
public:
	explicit DirichletWallBoundary(const std::string& shape_name) : MultiPolygonShape(shape_name)
	{
		multi_polygon_.addAPolygon(left_temperature_region, ShapeBooleanOps::add);
		multi_polygon_.addAPolygon(right_temperature_region, ShapeBooleanOps::add);
	}
};

class RobinWallBoundary : public MultiPolygonShape
{
public:
	explicit RobinWallBoundary(const std::string& shape_name) : MultiPolygonShape(shape_name)
	{
		multi_polygon_.addAPolygon(convection_region, ShapeBooleanOps::add);
	}
};
//----------------------------------------------------------------------
//	Define reaction type 
//----------------------------------------------------------------------
class AqueousSpeciesReaction : public BaseReactionModel<2>
{
protected:
	Real X_A_; //aqueous concentration of species A
	Real X_B_; //aqueous concentration of species B

	virtual Real getProductionRateXA(LocalSpecies& species) = 0;
	virtual Real getLossRateXA(LocalSpecies& species) = 0;
	virtual Real getProductionRateXB(LocalSpecies& species) = 0;
	virtual Real getLossRateXB(LocalSpecies& species) = 0;

public:
	explicit AqueousSpeciesReaction()
		: BaseReactionModel<2>({ "AAqueousConcentration", "BAqueousConcentration"}),
		X_A_(species_indexes_map_["AAqueousConcentration"]),
		X_B_(species_indexes_map_["BAqueousConcentration"])
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
	Real Y_A_; //aqueous concentration of species A
	Real Y_B_; //aqueous concentration of species B

	virtual Real getProductionRateYA(LocalSpecies& species) = 0;
	virtual Real getLossRateYA(LocalSpecies& species) = 0;
	virtual Real getProductionRateYB(LocalSpecies& species) = 0;
	virtual Real getLossRateYB(LocalSpecies& species) = 0;

public:
	explicit AdsorbedSpeciesReaction()
		: BaseReactionModel<2>({ "AAdsorbedConcentration", "BAdsorbedConcentration"}),
		Y_A_(species_indexes_map_["AAdsorbedConcentration"]),
		Y_B_(species_indexes_map_["BAdsorbedConcentration"])
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
		Real X_A = species[X_A_];
		Real theta_A = species[Y_A_] / Y_A_max_;
		Real theta_B = species[Y_B_] / Y_B_max_;
		return k_A_ad_ * X_A * pow(1 - theta_A - theta_B, adsorption_sites_A_);
	}

	virtual Real getLossRateXA(LocalSpecies& species) override
	{
		Real theta_A = species[Y_A_] / Y_A_max_;
		return k_A_de_ * pow(theta_A, adsorption_sites_A_);
	}

	virtual Real getProductionRateXB(LocalSpecies& species) override
	{
		Real X_B = species[X_B_];
		Real theta_A = species[Y_A_] / Y_A_max_;
		Real theta_B = species[Y_B_] / Y_B_max_;
		return k_B_ad_ * X_B * pow(1 - theta_A - theta_B, adsorption_sites_B_);
	}

	virtual Real getLossRateXB(LocalSpecies& species) override
	{
		Real theta_B = species[Y_B_] / Y_B_max_;
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
class AqueousSpecies : public DiffusionReaction<WeaklyCompressibleFluid, 2>
{
public:
	AqueousSpecies(SharedPtr<AqueousSpeciesReaction> aqueous_species_reaction_ptr, Real diff_cf_A_aqueous, Real diff_cf_B_aqueous)
		: DiffusionReaction<WeaklyCompressibleFluid, 2>({ "AAqueousConcentration", "BAqueousConcentration" }, aqueous_species_reaction_ptr, rho0_f, c_f, mu_f)
	{
		material_type_name_ = "AqueousSpecies";
		initializeAnDiffusion<IsotropicDiffusion>("AAqueousConcentration", "AAqueousConcentration", diff_cf_A_aqueous);
		initializeAnDiffusion<IsotropicDiffusion>("BAqueousConcentration", "BAqueousConcentration", diff_cf_B_aqueous);
	};

	virtual ~AqueousSpecies() {};
};

//class AdsorbedSpecies : public DiffusionReaction<Fluid, 2>
//{
//public:
//	template <class DiffusionType>
//	AdsorbedSpecies(SharedPtr<AdsorbedSpeciesReaction> adsorbed_species_reaction_ptr,
//		TypeIdentity<DiffusionType> empty_object, Real diff_cf_A_adsorbed, Real diff_cf_B_adsorbed)
//		: DiffusionReaction<Fluid, 2>({ "AAdsorbedConcentration", "BAdsorbedConcentration" },
//			adsorbed_species_reaction_ptr)
//	{
//		material_type_name_ = "AdsorbedSpecies";
//		initializeAnDiffusion<DiffusionType>("AAdsorbedConcentration", "AAdsorbedConcentration", diff_cf_A_adsorbed);
//		initializeAnDiffusion<DiffusionType>("BAdsorbedConcentration", "BAdsorbedConcentration", diff_cf_B_adsorbed);
//	};
//
//	virtual ~AdsorbedSpecies() {};
//};

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
using AqueousParticles = DiffusionReactionParticles<FluidParticles, AqueousSpecies>;
using WallBoundaryParticles = DiffusionReactionParticles<SolidParticles, AdsorbedSpecies>;

//----------------------------------------------------------------------
//	Application dependent initial condition. 
//----------------------------------------------------------------------
class DiffusionInitialCondition
	: public DiffusionReactionInitialCondition<AqueousParticles>
{
protected:
	Real X_A_;
	Real X_B_;

public:
	explicit DiffusionInitialCondition(SPHBody& sph_body)
		: DiffusionReactionInitialCondition<AqueousParticles>(sph_body)
	{
		X_A_ = particles_->diffusion_reaction_material_.AllSpeciesIndexMap()["AAqueousConcentration"];
		X_B_ = particles_->diffusion_reaction_material_.AllSpeciesIndexMap()["BAqueousConcentration"];
	};

	void update(size_t index_i, Real dt)
	{
		all_species_[X_A_][index_i] = initial_temperature;
	};
};

class DirichletWallBoundaryInitialCondition
	: public DiffusionReactionInitialCondition<WallBoundaryParticles>
{
protected:
	Real Y_A_;
	Real Y_B_;

public:
	DirichletWallBoundaryInitialCondition(SolidBody& diffusion_body) :
		DiffusionReactionInitialCondition<WallBoundaryParticles>(diffusion_body)
	{
		Y_A_ = particles_->diffusion_reaction_material_.AllSpeciesIndexMap()["AAdsorbedConcentration"];
		Y_B_ = particles_->diffusion_reaction_material_.AllSpeciesIndexMap()["BAdsorbedConcentration"];
	}

	void update(size_t index_i, Real dt)
	{
		all_species_[Y_A_][index_i] = -0.0;
		all_species_[Y_B_][index_i] = -0.0;

		if (pos_[index_i][1] > H && pos_[index_i][0] > 0.3 * L && pos_[index_i][0] < 0.4 * L)
		{
			all_species_[Y_A_][index_i] = left_temperature;
		}
		if (pos_[index_i][1] > H && pos_[index_i][0] > 0.6 * L && pos_[index_i][0] < 0.7 * L)
		{
			all_species_[Y_B_][index_i] = right_temperature;
		}
	}
};

using AqueousDiffusionInner = DiffusionRelaxationInner<AqueousParticles>;
using WallBoundaryDirichlet = DiffusionRelaxationDirichlet<AqueousParticles, WallBoundaryParticles>;
//----------------------------------------------------------------------
//	Specify diffusion relaxation method. 
//----------------------------------------------------------------------
class DiffusionBodyRelaxation
	: public DiffusionRelaxationRK2<ComplexInteraction<AqueousDiffusionInner, WallBoundaryDirichlet>>
{
public:
	explicit DiffusionBodyRelaxation(BaseInnerRelation& inner_relation,
									 BaseContactRelation& body_contact_relation_Dirichlet)
		: DiffusionRelaxationRK2<ComplexInteraction<AqueousDiffusionInner, WallBoundaryDirichlet>>(
			inner_relation, body_contact_relation_Dirichlet) {};
	virtual ~DiffusionBodyRelaxation() {};
};
//----------------------------------------------------------------------
//	An observer body to measure temperature at given positions. 
//----------------------------------------------------------------------
class TemperatureObserverParticleGenerator : public ObserverParticleGenerator
{
public:
	TemperatureObserverParticleGenerator(SPHBody& sph_body) : ObserverParticleGenerator(sph_body)
	{
		/** A line of measuring points at the middle line. */
		size_t number_of_observation_points = 5;
		Real range_of_measure = L;
		Real start_of_measure = 0;

		for (size_t i = 0; i < number_of_observation_points; ++i)
		{
			Vec2d point_coordinate(0.5 * L, range_of_measure * Real(i) /
                                                    Real(number_of_observation_points - 1) +
                                                start_of_measure);
			positions_.push_back(point_coordinate);
		}
	}
};
#endif //DIFFUSION_TEST_WITH_NEUMANNBC_H