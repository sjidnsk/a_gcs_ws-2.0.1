"""Ackermann plotting components."""

from importlib import import_module

_LAZY_EXPORTS = {
    'PathComparator': ('.paths', 'PathComparator'),
    'Plot2DTrajectory': ('.trajectory_2d', 'Plot2DTrajectory'),
    'Plot3DTrajectory': ('.trajectory_3d', 'Plot3DTrajectory'),
    'PlotProfiles': ('.profiles', 'PlotProfiles'),
    'RegionRenderer': ('.regions', 'RegionRenderer'),
    'visualize_3d_trajectory': ('.trajectory_3d', 'visualize_3d_trajectory'),
}

__all__ = [
    'PathComparator',
    'Plot2DTrajectory',
    'Plot3DTrajectory',
    'PlotProfiles',
    'RegionRenderer',
    'visualize_3d_trajectory',
]


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
