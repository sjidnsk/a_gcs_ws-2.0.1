"""
Project-level configuration entry points.

Use submodules for algorithm-specific dataclasses, for example
``config.iris.IrisNpConfig`` or ``config.solver.AdaptiveSolverConfig``.
"""

from .project import ProjectConfig, dump_resolved_config, load_project_config

__all__ = [
    "ProjectConfig",
    "load_project_config",
    "dump_resolved_config",
]
