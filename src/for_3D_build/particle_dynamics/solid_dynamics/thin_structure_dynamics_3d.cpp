#include "thin_structure_dynamics.h"
#include "base_particles.hpp"

namespace SPH
{
//=====================================================================================================//
namespace thin_structure_dynamics
{
void PrincipalStrains::update(size_t index_i, Real dt)
{
    Mat3d F = F_[index_i];
    Mat3d green_lagrange_strain = 0.5 * (F.transpose() * F - Matd::Identity());

    Vec3d principal_strains = getPrincipalValuesFromMatrix(green_lagrange_strain);
    Real eps_1 = principal_strains[0];
    Real eps_2 = principal_strains[1];
    Real eps_3 = principal_strains[2];

    derived_variable_[index_i] = Vecd(eps_1, eps_2, eps_3);
    max_principal_strain_[index_i] = std::max(std::max(eps_1, eps_2), eps_3);
}
//=================================================================================================//
} // namespace thin_structure_dynamics
} // namespace SPH