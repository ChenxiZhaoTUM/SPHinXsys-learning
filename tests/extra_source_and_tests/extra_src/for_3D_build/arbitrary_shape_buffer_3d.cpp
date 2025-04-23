#include "arbitrary_shape_buffer_3d.h"

namespace SPH
{
//=================================================================================================//
bool AlignedCylinderShape::checkInBounds(const Vecd &probe_point, Real lower_bound_fringe, Real upper_bound_fringe)
{
    Vecd position_in_frame = transform_.shiftBaseStationToFrame(probe_point);
    Real radial_distance = sqrt(position_in_frame[1] * position_in_frame[1] + position_in_frame[2] * position_in_frame[2]);
    return (ABS(position_in_frame[0]) <= halflength_ && radial_distance <= radius_)
        ? true : false;
}
//=================================================================================================//
bool AlignedCylinderShape::checkUpperBound(const Vecd &probe_point, Real upper_bound_fringe)
{
    Vecd position_in_frame = transform_.shiftBaseStationToFrame(probe_point);
    return position_in_frame[0] > halflength_ ? true : false;
}
//=================================================================================================//
bool AlignedCylinderShape::checkLowerBound(const Vecd &probe_point, Real lower_bound_fringe)
{
    Vecd position_in_frame = transform_.shiftBaseStationToFrame(probe_point);
    return position_in_frame[0] < halflength_ ? true : false;
}
//=================================================================================================//
bool AlignedCylinderShape::checkNearUpperBound(const Vecd &probe_point, Real threshold)
{
    Vecd position_in_frame = transform_.shiftBaseStationToFrame(probe_point);
    return ABS(position_in_frame[0] - halflength_) <= threshold ? true : false;
}
//=================================================================================================//
bool AlignedCylinderShape::checkNearLowerBound(const Vecd &probe_point, Real threshold)
{
    Vecd position_in_frame = transform_.shiftBaseStationToFrame(probe_point);
    return ABS(position_in_frame[0] + halflength_) <= threshold ? true : false;
}
//=================================================================================================//
Vecd AlignedCylinderShape::getUpperPeriodic(const Vecd &probe_point)
{
    Vecd position_in_frame = transform_.shiftBaseStationToFrame(probe_point);
    Vecd shift = Vecd::Zero();
    shift[0] -= 2.0 * halflength_;
    return transform_.shiftFrameStationToBase(position_in_frame + shift);
}
//=================================================================================================//
Vecd AlignedCylinderShape::getLowerPeriodic(const Vecd &probe_point)
{
    Vecd position_in_frame = transform_.shiftBaseStationToFrame(probe_point);
    Vecd shift = Vecd::Zero();
    shift[0] += 2.0 * halflength_;
    return transform_.shiftFrameStationToBase(position_in_frame + shift);
}
//=================================================================================================//
bool AlignedCylinderShapeByTriangleMesh::checkInBounds(const Vecd &probe_point, Real lower_bound_fringe, Real upper_bound_fringe)
{
    SimTKVec3 probe_point_simtk(probe_point[0], probe_point[1], probe_point[2]);
    SimTKVec3 translation_simtk(translation_[0], translation_[1], translation_[2]);
    SimTKVec3 relative_position = probe_point_simtk - translation_simtk;
    
    Real distance_along_axis = dot(relative_position, cylinder_length_axis_);
    if (distance_along_axis < -halflength_ || distance_along_axis > halflength_)
        return false;

    SimTKVec3 perpendicular_position = relative_position - distance_along_axis * cylinder_length_axis_;
    Real distance_perpendicular = perpendicular_position.norm();

    return distance_perpendicular <= radius_ ? true : false;
}
//=================================================================================================//
bool AlignedCylinderShapeByTriangleMesh::checkUpperBound(const Vecd &probe_point, Real upper_bound_fringe)
{
    SimTKVec3 probe_point_simtk(probe_point[0], probe_point[1], probe_point[2]);
    SimTKVec3 translation_simtk(translation_[0], translation_[1], translation_[2]);
    SimTKVec3 relative_position = probe_point_simtk - translation_simtk;
    
    Real distance_along_axis = dot(relative_position, cylinder_length_axis_);

    return distance_along_axis > halflength_ ? true : false;
}
//=================================================================================================//
bool AlignedCylinderShapeByTriangleMesh::checkLowerBound(const Vecd &probe_point, Real lower_bound_fringe)
{
    SimTKVec3 probe_point_simtk(probe_point[0], probe_point[1], probe_point[2]);
    SimTKVec3 translation_simtk(translation_[0], translation_[1], translation_[2]);
    SimTKVec3 relative_position = probe_point_simtk - translation_simtk;
    
    Real distance_along_axis = dot(relative_position, cylinder_length_axis_);

    return distance_along_axis < halflength_ ? true : false;
}
//=================================================================================================//
bool AlignedCylinderShapeByTriangleMesh::checkNearUpperBound(const Vecd &probe_point, Real threshold)
{
    SimTKVec3 probe_point_simtk(probe_point[0], probe_point[1], probe_point[2]);
    SimTKVec3 translation_simtk(translation_[0], translation_[1], translation_[2]);
    SimTKVec3 relative_position = probe_point_simtk - translation_simtk;
    
    Real distance_along_axis = dot(relative_position, cylinder_length_axis_);
    return ABS(distance_along_axis - halflength_) <= threshold ? true : false;
}
//=================================================================================================//
bool AlignedCylinderShapeByTriangleMesh::checkNearLowerBound(const Vecd &probe_point, Real threshold)
{
    SimTKVec3 probe_point_simtk(probe_point[0], probe_point[1], probe_point[2]);
    SimTKVec3 translation_simtk(translation_[0], translation_[1], translation_[2]);
    SimTKVec3 relative_position = probe_point_simtk - translation_simtk;
    
    Real distance_along_axis = dot(relative_position, cylinder_length_axis_);
    return ABS(distance_along_axis + halflength_) <= threshold ? true : false;
}
//=================================================================================================//
Vecd AlignedCylinderShapeByTriangleMesh::getUpperPeriodic(const Vecd &probe_point)
{
    SimTKVec3 probe_point_simtk(probe_point[0], probe_point[1], probe_point[2]);
    SimTKVec3 translation_simtk(translation_[0], translation_[1], translation_[2]);
    SimTKVec3 relative_position = probe_point_simtk - translation_simtk;

    SimTKVec3 offset = -2.0 * halflength_ * cylinder_length_axis_;
    SimTKVec3 wrapped_position = relative_position + offset;
    return Vecd(wrapped_position[0] + translation_[0], wrapped_position[1] + translation_[1], wrapped_position[2] + translation_[2]);
}
//=================================================================================================//
Vecd AlignedCylinderShapeByTriangleMesh::getLowerPeriodic(const Vecd &probe_point)
{
    SimTKVec3 probe_point_simtk(probe_point[0], probe_point[1], probe_point[2]);
    SimTKVec3 translation_simtk(translation_[0], translation_[1], translation_[2]);
    SimTKVec3 relative_position = probe_point_simtk - translation_simtk;

    SimTKVec3 offset = 2.0 * halflength_ * cylinder_length_axis_;
    SimTKVec3 wrapped_position = relative_position + offset;
    return Vecd(wrapped_position[0] + translation_[0], wrapped_position[1] + translation_[1], wrapped_position[2] + translation_[2]);
}
//=================================================================================================//
} // namespace SPH