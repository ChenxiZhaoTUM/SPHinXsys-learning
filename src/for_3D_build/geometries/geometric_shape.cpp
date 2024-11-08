#include "geometric_shape.h"

namespace SPH
{
//=================================================================================================//
SimTK::ContactGeometry *GeometricShape::getContactGeometry()
{
    if (contact_geometry_ == nullptr)
    {
        std::cout << "\n Error: ContactGeometry not setup yet! \n";
        std::cout << __FILE__ << ':' << __LINE__ << std::endl;
        exit(1);
    }
    return contact_geometry_;
};
//=================================================================================================//
bool GeometricShape::checkContain(const Vec3d &probe_point, bool BOUNDARY_INCLUDED)
{
    SimTK::UnitVec3 normal;
    bool inside = false;
    contact_geometry_->findNearestPoint(SimTKVec3(probe_point[0], probe_point[1], probe_point[2]), inside, normal);

    return inside;
}
//=================================================================================================//
Vec3d GeometricShape::findClosestPoint(const Vec3d &probe_point)
{
    SimTK::UnitVec3 normal;
    bool inside = false;
    SimTKVec3 out_pnt = contact_geometry_->findNearestPoint(SimTKVec3(probe_point[0], probe_point[1], probe_point[2]), inside, normal);

    return Vecd(out_pnt[0], out_pnt[1], out_pnt[2]);
}
//=================================================================================================//
GeometricShapeBox::
    GeometricShapeBox(const Vecd &halfsize, const std::string &shape_name)
    : GeometricShape(shape_name), brick_(EigenToSimTK(halfsize)), halfsize_(halfsize)
{
    contact_geometry_ = &brick_;
}
//=================================================================================================//
bool GeometricShapeBox::checkContain(const Vec3d &probe_point, bool BOUNDARY_INCLUDED)
{
    return brick_.getGeoBox().containsPoint(SimTKVec3(probe_point[0], probe_point[1], probe_point[2]));
}
//=================================================================================================//
Vec3d GeometricShapeBox::findClosestPoint(const Vec3d &probe_point)
{
    bool inside = false;
    SimTKVec3 out_pnt = brick_.getGeoBox().findClosestPointOnSurface(SimTKVec3(probe_point[0], probe_point[1], probe_point[2]), inside);

    return Vecd(out_pnt[0], out_pnt[1], out_pnt[2]);
}
//=================================================================================================//
BoundingBox GeometricShapeBox::findBounds()
{
    return BoundingBox(-halfsize_, halfsize_);
}
//=================================================================================================//
GeometricShapeBall::
    GeometricShapeBall(const Vecd &center, const Real &radius, const std::string &shape_name)
    : GeometricShape(shape_name), center_(center), sphere_(radius)
{
    contact_geometry_ = &sphere_;
}
//=================================================================================================//
bool GeometricShapeBall::checkContain(const Vec3d &probe_point, bool BOUNDARY_INCLUDED)
{
    return (probe_point - center_).norm() < sphere_.getRadius();
}
//=================================================================================================//
Vec3d GeometricShapeBall::findClosestPoint(const Vec3d &probe_point)
{
    Vec3d displacement = probe_point - center_;
    Real distance = displacement.norm();
    Real cosine0 = (SGN(displacement[0]) * (ABS(displacement[0])) + TinyReal) / (distance + TinyReal);
    Real cosine1 = displacement[1] / (distance + TinyReal);
    Real cosine2 = displacement[2] / (distance + TinyReal);
    return probe_point + (sphere_.getRadius() - distance) * Vec3d(cosine0, cosine1, cosine2);
}
//=================================================================================================//
BoundingBox GeometricShapeBall::findBounds()
{
    Vecd shift = Vecd(sphere_.getRadius(), sphere_.getRadius(), sphere_.getRadius());
    return BoundingBox(center_ - shift, center_ + shift);
}
//=================================================================================================//
GeometricShapeCylinder::
    GeometricShapeCylinder(const Real &radius, const Real &halflength, const std::string &shape_name)
    : GeometricShape(shape_name), cylinder_(radius), halflength_(halflength), radius_(radius)
{
    contact_geometry_ = &cylinder_;
    // default center (0, 0, 0), default axis direction (1, 0, 0)
}
//=================================================================================================//
bool GeometricShapeCylinder::checkContain(const Vec3d &probe_point, bool BOUNDARY_INCLUDED)
{
    Vec3d radial_vector = Vec3d(0, probe_point[1], probe_point[2]);
    Real radial_distance = radial_vector.norm();
    Real axial_distance = ABS(probe_point[0]);

    return (radial_distance < cylinder_.getRadius() && axial_distance < halflength_);
}
//=================================================================================================//
Vec3d GeometricShapeCylinder::findClosestPoint(const Vec3d &probe_point)
{
    bool ptWasInside = checkContain(probe_point, true);
    Real axial_distance = probe_point[0];
    Real radial_distance = sqrt(probe_point[1] * probe_point[1] + probe_point[2] * probe_point[2]);
    Vecd closest_point = probe_point;

    Real distance_to_caps = halflength_ - ABS(axial_distance); // Distance to the nearest end cap
    Real distance_to_side = cylinder_.getRadius() - radial_distance; // Distance to the cylindrical surface

    if (!ptWasInside)
    {
        if (distance_to_caps < 0)
        {
            closest_point[0] = (axial_distance < 0) ? -halflength_ : halflength_;  // Set to nearest end cap
        }

        if (distance_to_side < 0)
        {
            Real scaling_factor = cylinder_.getRadius() / (radial_distance + TinyReal); // Scale the radial vector to lie on the surface
            closest_point[1] = probe_point[1] * scaling_factor;
            closest_point[2] = probe_point[2] * scaling_factor;
        }
    }
    else
    {
        if (distance_to_side < distance_to_caps)
        {
            // Closer to the cylindrical surface, project onto the surface
            Real scaling_factor = cylinder_.getRadius() / (radial_distance + TinyReal);
            closest_point[1] = probe_point[1] * scaling_factor;
            closest_point[2] = probe_point[2] * scaling_factor;
        }
        else
        {
            // Closer to the flat ends, project onto the nearest cap
            closest_point[0] = (axial_distance < 0) ? -halflength_ : halflength_;
        }
    }

    return closest_point;
}
//=================================================================================================//
BoundingBox GeometricShapeCylinder::findBounds()
{
    Vecd lower_bound(-halflength_, -cylinder_.getRadius(), -cylinder_.getRadius());
    Vecd upper_bound(halflength_, cylinder_.getRadius(), cylinder_.getRadius());
    return BoundingBox(lower_bound, upper_bound);
}
//=================================================================================================//
} // namespace SPH