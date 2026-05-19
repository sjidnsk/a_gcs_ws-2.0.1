"""Experimental visualization helpers."""


def visualize_environment_with_bezier(*args, **kwargs):
    """Render an experimental environment/Bezier view on demand."""
    from .environment_with_bezier import (
        visualize_environment_with_bezier as _visualize_environment_with_bezier,
    )

    return _visualize_environment_with_bezier(*args, **kwargs)

__all__ = ['visualize_environment_with_bezier']
