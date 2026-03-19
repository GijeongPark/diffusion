"""Utilities for PEH inverse-design dataset preparation."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "GeometryBuildConfig",
    "PipelineArtifacts",
    "PipelineConfig",
    "VolumeMeshConfig",
    "aggregate_response_directory",
    "build_geometry_dataset",
    "build_unit_cell_solid_polygons",
    "convert_msh_to_xdmf",
    "mesh_tiled_plate_sample",
    "mesh_tiled_plate_volume_sample",
    "run_pipeline",
    "save_fem_response",
    "tile_unit_cell_polygons",
]

_ATTRIBUTE_TO_MODULE = {
    "GeometryBuildConfig": ("peh_inverse_design.geometry_pipeline", "GeometryBuildConfig"),
    "PipelineArtifacts": ("peh_inverse_design.pipeline_runner", "PipelineArtifacts"),
    "PipelineConfig": ("peh_inverse_design.pipeline_runner", "PipelineConfig"),
    "VolumeMeshConfig": ("peh_inverse_design.volume_mesh", "VolumeMeshConfig"),
    "aggregate_response_directory": ("peh_inverse_design.response_dataset", "aggregate_response_directory"),
    "build_geometry_dataset": ("peh_inverse_design.geometry_pipeline", "build_geometry_dataset"),
    "build_unit_cell_solid_polygons": ("peh_inverse_design.geometry_pipeline", "build_unit_cell_solid_polygons"),
    "convert_msh_to_xdmf": ("peh_inverse_design.geometry_pipeline", "convert_msh_to_xdmf"),
    "mesh_tiled_plate_sample": ("peh_inverse_design.geometry_pipeline", "mesh_tiled_plate_sample"),
    "mesh_tiled_plate_volume_sample": ("peh_inverse_design.volume_mesh", "mesh_tiled_plate_volume_sample"),
    "run_pipeline": ("peh_inverse_design.pipeline_runner", "run_pipeline"),
    "save_fem_response": ("peh_inverse_design.response_dataset", "save_fem_response"),
    "tile_unit_cell_polygons": ("peh_inverse_design.geometry_pipeline", "tile_unit_cell_polygons"),
}


def __getattr__(name: str):
    if name not in _ATTRIBUTE_TO_MODULE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _ATTRIBUTE_TO_MODULE[name]
    module = import_module(module_name)
    return getattr(module, attribute_name)
