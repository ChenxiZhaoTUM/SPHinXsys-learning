#include "complex_shape.h"

namespace SPH
{
//=================================================================================================//
bool AlignedBoxShape::checkInBounds(const Vecd &probe_point)
{
    Vecd position_in_frame = transform_.shiftBaseStationToFrame(probe_point);
    return position_in_frame[alignment_axis_] >= -halfsize_[alignment_axis_] &&
                   position_in_frame[alignment_axis_] <= halfsize_[alignment_axis_]
               ? true
               : false;
}
//=================================================================================================//
bool AlignedBoxShape::checkUpperBound(const Vecd &probe_point)
{
    Vecd position_in_frame = transform_.shiftBaseStationToFrame(probe_point);
    return position_in_frame[alignment_axis_] > halfsize_[alignment_axis_] ? true : false;
}
//=================================================================================================//
bool AlignedBoxShape::checkLowerBound(const Vecd &probe_point)
{
    Vecd position_in_frame = transform_.shiftBaseStationToFrame(probe_point);
    return position_in_frame[alignment_axis_] < -halfsize_[alignment_axis_] ? true : false;
}
//=================================================================================================//
bool AlignedBoxShape::checkNearUpperBound(const Vecd &probe_point, Real threshold)
{
    Vecd position_in_frame = transform_.shiftBaseStationToFrame(probe_point);
    return ABS(position_in_frame[alignment_axis_] - halfsize_[alignment_axis_]) <= threshold ? true : false;
}
//=================================================================================================//
bool AlignedBoxShape::checkNearLowerBound(const Vecd &probe_point, Real threshold)
{
    Vecd position_in_frame = transform_.shiftBaseStationToFrame(probe_point);
    return ABS(position_in_frame[alignment_axis_] + halfsize_[alignment_axis_]) <= threshold ? true : false;
}
//=================================================================================================//
Vecd AlignedBoxShape::getUpperPeriodic(const Vecd &probe_point)
{
    Vecd position_in_frame = transform_.shiftBaseStationToFrame(probe_point);
    Vecd shift = Vecd::Zero();
    shift[alignment_axis_] -= 2.0 * halfsize_[alignment_axis_];
    return transform_.shiftFrameStationToBase(position_in_frame + shift);
}
//=================================================================================================//
Vecd AlignedBoxShape::getLowerPeriodic(const Vecd &probe_point)
{
    Vecd position_in_frame = transform_.shiftBaseStationToFrame(probe_point);
    Vecd shift = Vecd::Zero();
    shift[alignment_axis_] += 2.0 * halfsize_[alignment_axis_];
    return transform_.shiftFrameStationToBase(position_in_frame + shift);
}
//=================================================================================================//
bool AlignedCylinderShape::checkInBounds(const Vecd &probe_point)
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

//=================================================================================================//
} // namespace SPH