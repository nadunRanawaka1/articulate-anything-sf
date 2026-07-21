#!/usr/bin/env python3
"""
Run just the interactive correction step (Step 4) with existing VLM results.
Usage: python run_interactive_correction.py --config-name=bamboo_cabinet_partfield objects.0.name=bamboo_cabinet
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import json
from pathlib import Path

from postprocess_segmentation import merge_and_center_segmented_mesh
from generate_urdf import generate_base_urdf


@hydra.main(config_path="cfg", config_name="bamboo_cabinet_partfield", version_base=None)
def main(cfg: DictConfig):
    # Get object config
    if hasattr(cfg, 'objects') and cfg.objects:
        object_config = cfg.objects[0]
        object_name = object_config['name']
    else:
        raise ValueError("No objects specified in config")
    
    object_root = f"{cfg.root_dir}/{cfg.scene_name}/{object_name}"
    
    # Load the existing VLM articulation tree
    s2_cfg = OmegaConf.create(cfg.s2_generate_articulation_tree)
    s2_cfg.out_dir = f"{object_root}/{s2_cfg.out_dirname}"
    articulation_tree_path = Path(s2_cfg.out_dir) / "result_generate_articulation_tree.json"
    
    if not articulation_tree_path.exists():
        raise FileNotFoundError(f"VLM result not found at: {articulation_tree_path}")
    
    with open(articulation_tree_path) as f:
        articulation_tree_dict = json.load(f)
    print(f"Loaded articulation tree from: {articulation_tree_path}")
    print(f"Parts: {[p['part_name'] for p in articulation_tree_dict.get('part_list', [])]}")
    
    # Setup s3 config (for paths)
    s3_cfg = OmegaConf.create(cfg.s3_segment_mesh)
    s3_cfg.out_dir = f"{object_root}/{s3_cfg.out_dirname}"
    
    # Setup s4 config
    s1_cfg = OmegaConf.create(cfg.s1_render)
    s1_cfg.out_dir = f"{object_root}/{s1_cfg.out_dirname}"
    s1_cfg.object_path = f"{s1_cfg.out_dir}/{object_name}.glb"
    
    s4_cfg = OmegaConf.create(cfg.s4_merge_mesh_parts)
    s4_cfg.out_dir = f"{object_root}/{s4_cfg.out_dirname}"
    s4_cfg.image_dir = f"{s3_cfg.out_dir}/rendered_parts"
    s4_cfg.object_name = object_name
    s4_cfg.object_path = s1_cfg.object_path
    s4_cfg.gcloud_project = cfg.gcloud_project
    s4_cfg.gcloud_location = cfg.gcloud_location
    s4_cfg.mesh_parts_dir = f"{s4_cfg.out_dir}/meshes"
    
    print(f"\nRunning interactive correction for: {object_name}")
    print(f"  Segment results: {s3_cfg.out_dir}")
    print(f"  Output: {s4_cfg.out_dir}")
    
    # Run the interactive correction
    merge_and_center_segmented_mesh(
        s4_cfg, 
        articulation_tree_dict, 
        verbose=cfg.verbose, 
        interactive=True
    )
    
    # Generate base URDF
    urdf_path = generate_base_urdf(s4_cfg, articulation_tree_dict, verbose=cfg.verbose)
    print(f"\nGenerated URDF: {urdf_path}")


if __name__ == "__main__":
    main()

