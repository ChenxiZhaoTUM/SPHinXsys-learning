#include "arbitrary_shape_buffer.h"

namespace SPH
{
//=================================================================================================//
bool AlignedBoxShape02::checkInBounds(const Vecd &probe_point)
{
    Vecd position_in_frame = transform_.shiftBaseStationToFrame(probe_point);
    return position_in_frame[alignment_axis_] >= -halfsize_[alignment_axis_] &&
                   position_in_frame[alignment_axis_] <= halfsize_[alignment_axis_]
               ? true
               : false;
}
//=================================================================================================//
bool AlignedBoxShape02::checkUpperBound(const Vecd &probe_point)
{
    Vecd position_in_frame = transform_.shiftBaseStationToFrame(probe_point);
    return position_in_frame[alignment_axis_] > halfsize_[alignment_axis_] ? true : false;
}
//=================================================================================================//
bool AlignedBoxShape02::checkLowerBound(const Vecd &probe_point)
{
    Vecd position_in_frame = transform_.shiftBaseStationToFrame(probe_point);
    return position_in_frame[alignment_axis_] < -halfsize_[alignment_axis_] ? true : false;
}
//=================================================================================================//
bool AlignedBoxShape02::checkNearUpperBound(const Vecd &probe_point, Real threshold)
{
    Vecd position_in_frame = transform_.shiftBaseStationToFrame(probe_point);
    return ABS(position_in_frame[alignment_axis_] - halfsize_[alignment_axis_]) <= threshold ? true : false;
}
//=================================================================================================//
bool AlignedBoxShape02::checkNearLowerBound(const Vecd &probe_point, Real threshold)
{
    Vecd position_in_frame = transform_.shiftBaseStationToFrame(probe_point);
    return ABS(position_in_frame[alignment_axis_] + halfsize_[alignment_axis_]) <= threshold ? true : false;
}
//=================================================================================================//
Vecd AlignedBoxShape02::getUpperPeriodic(const Vecd &probe_point)
{
    Vecd position_in_frame = transform_.shiftBaseStationToFrame(probe_point);
    Vecd shift = Vecd::Zero();
    shift[alignment_axis_] -= 2.0 * halfsize_[alignment_axis_];
    return transform_.shiftFrameStationToBase(position_in_frame + shift);
}
//=================================================================================================//
Vecd AlignedBoxShape02::getLowerPeriodic(const Vecd &probe_point)
{
    Vecd position_in_frame = transform_.shiftBaseStationToFrame(probe_point);
    Vecd shift = Vecd::Zero();
    shift[alignment_axis_] += 2.0 * halfsize_[alignment_axis_];
    return transform_.shiftFrameStationToBase(position_in_frame + shift);
}
//=================================================================================================//
} // namespace SPH