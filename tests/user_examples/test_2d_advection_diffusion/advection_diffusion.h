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
Real diffusion_coff = 1;
std::array<std::string, 1> species_name_list{ "Phi" };
//----------------------------------------------------------------------
//	Initial and boundary conditions.
//----------------------------------------------------------------------
Real initial_temperature = 100.0;
Real left_temperature = 300.0;
Real right_temperature = 350.0;
Real convection = 100.0;
Real T_infinity = 400.0;
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
class CompetitiveAdsorptionReaction : public BaseReactionModel<4>
{
protected:
	Real X_A_; //aqueous concentration of species A
	Real Y_A_; //adsorpted concentration of species A

	Real X_B_; //aqueous concentration of species B
	Real Y_B_; //adsorpted concentration of species B

	virtual Real getProductionRateXA(LocalSpecies& species) = 0;
	virtual Real getLossRateXA(LocalSpecies& species) = 0;
	virtual Real getProductionRateXB(LocalSpecies& species) = 0;
	virtual Real getLossRateXB(LocalSpecies& species) = 0;

public:
	explicit CompetitiveAdsorptionReaction()
		: BaseReactionModel<4>({ "AAqueousConcentration", "AAdsorptedConcentration","BAqueousConcentration","BAdsorptedConcentration" }),
		X_A_(species_indexes_map_["AAqueousConcentration"]),
		Y_A_(species_indexes_map_["AAdsorptedConcentration"]),
		X_B_(species_indexes_map_["BAqueousConcentration"]),
		Y_B_(species_indexes_map_["BAdsorptedConcentration"])
	{
		reaction_model_ = "CompetitiveAdsorptionReaction";
		initializeCompetitiveAdsorptionReaction();
	};

	virtual ~CompetitiveAdsorptionReaction() {};

	void initializeCompetitiveAdsorptionReaction()
	{
		get_production_rates_.push_back(std::bind(&CompetitiveAdsorptionReaction::getProductionRateXA, this, _1));
		get_production_rates_.push_back(std::bind(&CompetitiveAdsorptionReaction::getProductionRateXB, this, _1));

		get_loss_rates_.push_back(std::bind(&CompetitiveAdsorptionReaction::getLossRateXA, this, _1));
		get_loss_rates_.push_back(std::bind(&CompetitiveAdsorptionReaction::getLossRateXB, this, _1));
	}
};

class LangmuirAdsorptionModel : public CompetitiveAdsorptionReaction
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

public:
	explicit LangmuirAdsorptionModel(Real k_A_ad, Real k_A_de, Real k_B_ad, Real k_B_de, Real adsorption_sites_A, Real adsorption_sites_B, Real Y_A_max, Real Y_B_max)
		: CompetitiveAdsorptionReaction(),
		k_A_ad_(k_A_ad), k_A_de_(k_A_de), k_B_ad_(k_B_ad), k_B_de_(k_B_de), adsorption_sites_A_(adsorption_sites_A), adsorption_sites_B_(adsorption_sites_A), Y_A_max_(Y_A_max), Y_B_max_(Y_B_max)
	{
		reaction_model_ = "LangmuirAdsorptionModel";
	}
	virtual ~LangmuirAdsorptionModel() {};
};

//----------------------------------------------------------------------
//	Setup diffusion material properties. 
//----------------------------------------------------------------------
class DiffusionMaterial : public DiffusionReaction<Solid>
{
public:
	DiffusionMaterial() : DiffusionReaction<Solid>({ "Phi" }, SharedPtr<NoReaction>())
	{
		initializeAnDiffusion<IsotropicDiffusion>("Phi", "Phi", diffusion_coff);
	}
};
using DiffusionParticles = DiffusionReactionParticles<SolidParticles, DiffusionMaterial>;
using WallParticles = DiffusionReactionParticles<SolidParticles, DiffusionMaterial>;
//----------------------------------------------------------------------
//	Application dependent initial condition. 
//----------------------------------------------------------------------
class DiffusionInitialCondition
	: public DiffusionReactionInitialCondition<DiffusionParticles>
{
protected:
	size_t phi_;

public:
	explicit DiffusionInitialCondition(SPHBody& sph_body)
		: DiffusionReactionInitialCondition<DiffusionParticles>(sph_body)
	{
		phi_ = particles_->diffusion_reaction_material_.AllSpeciesIndexMap()["Phi"];
	};

	void update(size_t index_i, Real dt)
	{
		all_species_[phi_][index_i] = initial_temperature;
	};
};

class DirichletWallBoundaryInitialCondition
	: public DiffusionReactionInitialCondition<WallParticles>
{
protected:
	size_t phi_;

public:
	DirichletWallBoundaryInitialCondition(SolidBody& diffusion_body) :
		DiffusionReactionInitialCondition<WallParticles>(diffusion_body)
	{
		phi_ = particles_->diffusion_reaction_material_.AllSpeciesIndexMap()["Phi"];
	}

	void update(size_t index_i, Real dt)
	{
		all_species_[phi_][index_i] = -0.0;

		if (pos_[index_i][1] > H && pos_[index_i][0] > 0.3 * L && pos_[index_i][0] < 0.4 * L)
		{
			all_species_[phi_][index_i] = left_temperature;
		}
		if (pos_[index_i][1] > H && pos_[index_i][0] > 0.6 * L && pos_[index_i][0] < 0.7 * L)
		{
			all_species_[phi_][index_i] = right_temperature;
		}
	}
};

class RobinWallBoundaryInitialCondition
	: public DiffusionReactionInitialCondition<WallParticles>
{
protected:
	size_t phi_;
	StdLargeVec<Real>& convection_;
	Real& T_infinity_;

public:
	RobinWallBoundaryInitialCondition(SolidBody& diffusion_body) :
		DiffusionReactionInitialCondition<WallParticles>(diffusion_body),
		convection_(*(this->particles_->template getVariableByName<Real>("Convection"))),
		T_infinity_(*(this->particles_->template getGlobalVariableByName<Real>("T_infinity")))
	{
		phi_ = particles_->diffusion_reaction_material_.AllSpeciesIndexMap()["Phi"];
	}

	void update(size_t index_i, Real dt)
	{
		all_species_[phi_][index_i] = -0.0;

		if (pos_[index_i][1] < 0 && pos_[index_i][0] > 0.45 * L && pos_[index_i][0] < 0.55 * L)
		{
			convection_[index_i] = convection;
			T_infinity_ = T_infinity;
		}
	}
};

using SolidDiffusionInner = DiffusionRelaxationInner<DiffusionParticles>;
using SolidDiffusionDirichlet = DiffusionRelaxationDirichlet<DiffusionParticles, WallParticles>;
using SolidDiffusionRobin = DiffusionRelaxationRobin<DiffusionParticles, WallParticles>;
//----------------------------------------------------------------------
//	Specify diffusion relaxation method. 
//----------------------------------------------------------------------
class DiffusionBodyRelaxation
	: public DiffusionRelaxationRK2<ComplexInteraction<SolidDiffusionInner, SolidDiffusionDirichlet, SolidDiffusionRobin>>
{
public:
	explicit DiffusionBodyRelaxation(BaseInnerRelation& inner_relation,
									 BaseContactRelation& body_contact_relation_Dirichlet,
									 BaseContactRelation& body_contact_relation_Robin)
		: DiffusionRelaxationRK2<ComplexInteraction<SolidDiffusionInner, SolidDiffusionDirichlet, SolidDiffusionRobin>>(
			inner_relation, body_contact_relation_Dirichlet, body_contact_relation_Robin) {};
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