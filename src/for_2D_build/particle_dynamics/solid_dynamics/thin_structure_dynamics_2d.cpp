#include "thin_structure_dynamics.h"
#include "base_particles.hpp"

namespace SPH
{
//=====================================================================================================//
namespace thin_structure_dynamics
{
void PrincipalStrains::update(size_t index_i, Real dt)
{
    Mat2d F = F_[index_i];
    Mat2d green_lagrange_strain = 0.5 * (F.transpose() * F - Matd::Identity());

    Vec2d principal_strains = getPrincipalValuesFromMatrix(green_lagrange_strain);
    Real eps_1 = principal_strains[0];
    Real eps_2 = principal_strains[1];

    derived_variable_[index_i] = Vecd(eps_1, eps_2);
    max_principal_strain_[index_i] = std::max(eps_1, eps_2);
}
//=================================================================================================//
} // namespace thin_structure_dynamics
} // namespace SPH