"""
Optional dependency checks for IrisZo.
"""


def check_drake_availability() -> bool:
    """Return True when the Drake geometry primitives required by IrisZo exist."""
    try:
        from pydrake.geometry.optimization import HPolyhedron, Hyperellipsoid  # noqa: F401
    except ImportError:
        return False
    return True


__all__ = ['check_drake_availability']
