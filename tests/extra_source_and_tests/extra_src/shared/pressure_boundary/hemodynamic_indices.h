#ifndef HEMODYNAMICS_INDICES_H
#define HEMODYNAMICS_INDICES_H

#include "all_particle_dynamics.h"
#include "base_material.h"
#include "elastic_dynamics.h"
#include "force_prior.hpp"
#include "riemann_solver.h"
#include "fluid_structure_interaction.h"
#include "base_fluid_dynamics.h"
#include "near_wall_boundary.h"
#include "velocity_gradient.h"

namespace SPH
{
namespace fluid_dynamics
{
class TwoLayersFromWall : public DistanceFromWall
{
  public:
    explicit TwoLayersFromWall(BaseContactRelation &wall_contact_relation);
    virtual ~TwoLayersFromWall(){};
    void update(size_t index_i, Real dt = 0.0);

  protected:
    int *two_layers_indicatior_;
};

class FluidWSS : public LocalDynamics
{
  public:
    explicit FluidWSS(SPHBody &sph_body);
    virtual ~FluidWSS(){};

    void update(size_t index_i, Real dt = 0.0);

  protected:
    int *two_layers_indicatior_;
    Matd *vel_grad_;
    Matd *wall_shear_stress_;
    Vecd *n_;
};

class FluidWSSTraction : public LocalDynamics
{
  public:
    explicit FluidWSSTraction(SPHBody &sph_body);
    virtual ~FluidWSSTraction(){};

    void update(size_t index_i, Real dt = 0.0);

  protected:
    int *two_layers_indicatior_;
    Matd *vel_grad_;
    Vecd *wall_shear_stress_;
    Vecd *n_;
    Real* p_;
};
} // namespace fluid_dynamics

namespace solid_dynamics
{
class WallShearStress : public BaseForceFromFluid
{
  public:
    explicit WallShearStress(BaseContactRelation &contact_relation);
    virtual ~WallShearStress(){};
    void interaction(size_t index_i, Real dt = 0.0);

  protected:
    Vecd *vel_ave_;
    Real spacing_ref_;
    Vecd *n_;
    Vecd *wall_shear_stress_;
    StdVec<Real *> contact_Vol_;
    StdVec<Vecd *> contact_vel_;
    StdVec<Real> mu_;
    StdVec<Real> smoothing_length_;
};

class HemodynamicIndiceCalculation : public LocalDynamics
{
  public:
    explicit HemodynamicIndiceCalculation(SPHBody &sph_body, Real cardiac_cycle_time);
    virtual ~HemodynamicIndiceCalculation(){};
    void update(size_t index_i, Real dt = 0.0);

  protected:
    Real cardiac_cycle_time_;
    Real *acc_time_;
    Vecd *wall_shear_stress_;
    Real *acc_wall_shear_stress_;
    Real *time_averaged_wall_shear_stress_;
    Vecd *acc_vector_wall_shear_stress_;
    Real *oscillatory_shear_index_;
};

class FirstLayerFromFluid : public LocalDynamics
{
  public:
    FirstLayerFromFluid(SPHBody &solid_body, SPHBody &fluid_body)
        : LocalDynamics(solid_body),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          solid_contact_indicator_(this->particles_->template registerStateVariable<int>("SolidFirstLayerIndicator")),
          fluid_body_(fluid_body),
          spacing_ref_(sph_body_.sph_adaptation_->ReferenceSpacing()){};
    virtual ~FirstLayerFromFluid(){};

    void update(size_t index_i, Real dt = 0.0)
    {
        Real phi = fluid_body_.getInitialShape().findSignedDistance(pos_[index_i]);
        solid_contact_indicator_[index_i] = 1;
        if (phi > 1.01 * spacing_ref_)
            solid_contact_indicator_[index_i] = 0;
    }

  protected:
    Vecd *pos_;
    int *solid_contact_indicator_;
    SPHBody &fluid_body_;
    Real spacing_ref_;
};

class TwoLayersFromFluid : public LocalDynamics
{
  public:
    TwoLayersFromFluid(SPHBody &solid_body, SPHBody &fluid_body)
        : LocalDynamics(solid_body),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          solid_contact_indicator_(this->particles_->template registerStateVariable<int>("SolidTwoLayersIndicator")),
          fluid_body_(fluid_body),
          spacing_ref_(sph_body_.sph_adaptation_->ReferenceSpacing()){};
    virtual ~TwoLayersFromFluid(){};

    void update(size_t index_i, Real dt = 0.0)
    {
        Real phi = fluid_body_.getInitialShape().findSignedDistance(pos_[index_i]);
        solid_contact_indicator_[index_i] = 1;
        if (phi > 2.0 * 1.15 * spacing_ref_)
            solid_contact_indicator_[index_i] = 0;
    }

  protected:
    Vecd *pos_;
    int *solid_contact_indicator_;
    SPHBody &fluid_body_;
    Real spacing_ref_;
};

class SolidWSSFromFluid : public LocalDynamics, public DataDelegateInner, public DataDelegateContact
{
  public:
    explicit SolidWSSFromFluid(BaseInnerRelation &inner_relation, BaseContactRelation &contact_relation);
    virtual ~SolidWSSFromFluid(){};

    void interaction(size_t index_i, Real dt = 0.0);
    
    Real getWSSMagnitudeFromMatrix(const Mat2d &sigma)
    {
        Real sigmaxx = sigma(0, 0);
        Real sigmayy = sigma(1, 1);
        Real sigmaxy = sigma(0, 1);

        return sqrt(sigmaxx * sigmaxx + sigmayy * sigmayy + sigmaxy * sigmaxy);
    }
    
    Real getWSSMagnitudeFromMatrix(const Mat3d &sigma)
    {
        Real sigmaxx = sigma(0, 0);
        Real sigmayy = sigma(1, 1);
        Real sigmazz = sigma(2, 2);
        Real sigmaxy = sigma(0, 1);
        Real sigmaxz = sigma(0, 2);
        Real sigmayz = sigma(1, 2);

        return sqrt(sigmaxx * sigmaxx + sigmayy * sigmayy + sigmazz * sigmazz + sigmaxy * sigmaxy + sigmaxz * sigmaxz + sigmayz * sigmayz);
    }

  protected:
    int *solid_contact_indicator_;
    Real *Vol_;
    StdVec<Real *> contact_Vol_;
    Matd *wall_shear_stress_;
    StdVec<Matd *> fluid_wall_shear_stress_;
    Matd *total_wall_shear_stress_;
    Real *WSS_magnitude_;
};

class SolidWSSFromFluidTraction : public LocalDynamics, public DataDelegateInner, public DataDelegateContact
{
  public:
    explicit SolidWSSFromFluidTraction(BaseInnerRelation &inner_relation, BaseContactRelation &contact_relation);
    virtual ~SolidWSSFromFluidTraction(){};

    void interaction(size_t index_i, Real dt = 0.0);

  protected:
    int *solid_contact_indicator_;
    Real *Vol_;
    StdVec<Real *> contact_Vol_;
    Vecd *wall_shear_stress_;
    StdVec<Vecd *> fluid_wall_shear_stress_;
    Vecd *total_wall_shear_stress_;
    Real *WSS_magnitude_;
};

class SolidWSSFromFluidTangent : public LocalDynamics, public DataDelegateInner, public DataDelegateContact
{
  public:
    explicit SolidWSSFromFluidTangent(BaseInnerRelation &inner_relation, BaseContactRelation &contact_relation);
    virtual ~SolidWSSFromFluidTangent(){};

    void interaction(size_t index_i, Real dt = 0.0);
    
    Real getWSSMagnitudeFromMatrix(const Mat2d &sigma)
    {
        Real sigmaxx = sigma(0, 0);
        Real sigmayy = sigma(1, 1);
        Real sigmaxy = sigma(0, 1);

        return sqrt(sigmaxx * sigmaxx + sigmayy * sigmayy + sigmaxy * sigmaxy);
    }
    
    Real getWSSMagnitudeFromMatrix(const Mat3d &sigma)
    {
        Real sigmaxx = sigma(0, 0);
        Real sigmayy = sigma(1, 1);
        Real sigmazz = sigma(2, 2);
        Real sigmaxy = sigma(0, 1);
        Real sigmaxz = sigma(0, 2);
        Real sigmayz = sigma(1, 2);

        return sqrt(sigmaxx * sigmaxx + sigmayy * sigmayy + sigmazz * sigmazz + sigmaxy * sigmaxy + sigmaxz * sigmaxz + sigmayz * sigmayz);
    }

  protected:
    int *solid_contact_indicator_;
    Real *Vol_;
    Vecd* n_;
    StdVec<Real *> contact_Vol_;
    Matd *wall_shear_stress_;
    StdVec<Matd *> fluid_wall_shear_stress_;
    Matd *total_wall_shear_stress_;
    Real *WSS_magnitude_;
};
} // namespace solid_dynamics

} // namespace SPH
#endif // HEMODYNAMICS_INDICES_H