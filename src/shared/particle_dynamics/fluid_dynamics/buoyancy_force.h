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
 * Portions copyright (c) 2017-2025 Technical University of Munich and       *
 * the authors' affiliations.                                                *
 *                                                                           *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may   *
 * not use this file except in compliance with the License. You may obtain a *
 * copy of the License at http://www.apache.org/licenses/LICENSE-2.0.        *
 *                                                                           *
 * ------------------------------------------------------------------------- */
/**
 * @file 	buoyancy_force.h
 * @brief   
 * @details  
 * @author	 
 */

#ifndef BUOYANCY_FORCE_H
#define BUOYANCY_FORCE_H

#include "base_fluid_dynamics.h"
#include "force_prior.h"

namespace SPH
{
namespace fluid_dynamics
{
class BuoyancyForce : public ForcePrior
{
  protected:
    Vecd gravity_;
    Real thermal_expansion_coeff_, phi_ref_;
    Real* phi_;
    
  public:
    BuoyancyForce(SPHBody &sph_body, const Real thermal_expansion_coeff, const Real phi_ref);
    virtual ~BuoyancyForce() {};
    void update(size_t index_i, Real dt = 0.0);
};

/**
 * @class NusseltNumInner
 * @brief  compute Nusselt number in the fluid field
 */
//class NusseltNumInner : public LocalDynamics, public DataDelegateInner
//{
//  public:
//    explicit NusseltNumInner(BaseInnerRelation &inner_relation);
//    virtual ~NusseltNumInner() {};
//
//    void interaction(size_t index_i, Real dt = 0.0);
//
//  protected:
//    Real *Vol_;
//    Vecd *vel_;
//    AngularVecd *vorticity_;
//};
} // namespace fluid_dynamics
} // namespace SPH
#endif // BUOYANCY_FORCE_H