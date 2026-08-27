#!/usr/bin/env python3
"""
Run just the interactive articulation refinement UI on existing results.

The workflow publishes each object's articulation to
<root_dir>/<scene_name>/<object>/results/{mobility.urdf, meshes/}. This script
opens the refinement UI (joint limits, pivots, axes, dynamics) on those
published results without re-running any articulation step — the counterpart
of run_interactive_correction.py for the segmentation step.

Usage (with the config generated for a scene by the SimFoundry pipeline):
    python run_articulation_refinement.py --config-name=generated_<scene>

Refine a subset of the configured objects:
    python run_articulation_refinement.py --config-name=generated_<scene> \
        '+refine_objects=[toaster_oven]'
"""

import os

import hydra
from omegaconf import DictConfig

from refine_articulation import interactive_articulation_refinement


@hydra.main(config_path="cfg", config_name="hunyuan_template", version_base=None)
def main(cfg: DictConfig):
    if hasattr(cfg, 'objects') and cfg.objects:
        object_names = []
        for obj in cfg.objects:
            try:
                object_names.append(obj['name'])
            except (TypeError, KeyError) as exc:
                raise ValueError(
                    f"Config entry in 'objects' has no usable 'name' ({obj!r}). "
                    "Pass the config generated for your scene, e.g. "
                    "--config-name=generated_<scene>"
                ) from exc
    elif 's1_render' in cfg:
        # Single-object mode: same default name the workflow publishes under.
        object_names = [cfg.s1_render.get('object_name', 'object')]
    else:
        raise ValueError("No objects specified in config")

    wanted = cfg.get('refine_objects', None)
    if wanted:
        if isinstance(wanted, str):  # tolerate +refine_objects=name (no brackets)
            wanted = [wanted]
        wanted = set(wanted)
        object_names = [name for name in object_names if name in wanted]
        missing = wanted - set(object_names)
        if missing:
            raise ValueError(
                f"refine_objects entries not in this config's objects: {sorted(missing)}. "
                f"Configured objects: {[o['name'] for o in cfg.objects] if cfg.get('objects') else object_names}"
            )

    results_dirs = {}
    for name in object_names:
        results_dir = f"{cfg.root_dir}/{cfg.scene_name}/{name}/results"
        if os.path.exists(f"{results_dir}/mobility.urdf"):
            results_dirs[name] = results_dir
        else:
            print(f"Skipping '{name}': no published result at {results_dir}/mobility.urdf")

    if not results_dirs:
        raise FileNotFoundError(
            f"No published articulation results under {cfg.root_dir}/{cfg.scene_name}. "
            "Run complete_workflow.py first."
        )

    s6_cfg = cfg.get('s6_refine_articulation', None)
    result = interactive_articulation_refinement(
        results_dirs,
        port=s6_cfg.get('port', 8060) if s6_cfg else 8060,
        open_browser=s6_cfg.get('open_browser', True) if s6_cfg else True,
        max_faces_per_link=s6_cfg.get('max_faces_per_link', 40000) if s6_cfg else 40000,
        verbose=cfg.get('verbose', False),
    )

    for name, summary in result['saved'].items():
        print(f"  {name}: v{summary['version']} -> {summary['urdf_path']} "
              f"(changed: {summary['changed_joints'] or 'physics only'})")


if __name__ == "__main__":
    main()
