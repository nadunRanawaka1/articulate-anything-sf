"""
Helpers for querying VLM models.
"""

import os
import json
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from articulate_anything.utils.prompts import recognize_parts_from_image, generate_articulation_tree_known_parts, merge_parts
from articulate_anything.utils.vlm import Claude, Gemini, GPT
from articulate_anything.utils.utils import IMAGE_EXTENSIONS
from articulate_anything.utils.prompt_utils import extract_json_from_response, is_claude_model


def make_vlm(cfg: DictConfig, verbose: bool = False):
    """
    Build the VLM client for a step, chosen by `cfg.model_name`.

    Gemini takes either a Developer API key (`cfg.api_key`, else GEMINI_API_KEY /
    GOOGLE_API_KEY) or Vertex AI + gcloud ADC via `cfg.gcloud_project`. It prefers
    the key when one is configured; force a route with `cfg.gemini_backend`
    ("api_key" or "vertex"). See prompt_utils.resolve_gemini_auth.

    Claude runs on Vertex AI and authenticates with `cfg.gcloud_project` + gcloud
    ADC. Claude is not served in every Gemini region, so `cfg.claude_location`
    can override `cfg.gcloud_location` for the Claude path ("global" is the safe
    choice). Setting `cfg.vlm_backend: anthropic` sends Claude to the direct
    Anthropic API instead, which bypasses Vertex quota entirely.

    These steps only ever send images and text, both of which Claude accepts.
    """
    if "gemini" in cfg.model_name:
        return Gemini(project=cfg.gcloud_project, location=cfg.gcloud_location,
                      model=cfg.model_name, verbose=verbose,
                      api_key=cfg.get("api_key", None),
                      backend=cfg.get("gemini_backend", None))
    elif is_claude_model(cfg.model_name):
        location = cfg.get("claude_location", None) or cfg.gcloud_location
        return Claude(project=cfg.gcloud_project, location=location,
                      model=cfg.model_name, verbose=verbose,
                      backend=cfg.get("vlm_backend", None) or "vertex",
                      api_key=cfg.get("api_key", None))
    elif "gpt" in cfg.model_name:
        return GPT(model_name=cfg.model_name, verbose=verbose)
    else:
        raise ValueError(
            f"Unsupported model_name '{cfg.model_name}': expected a 'gemini', "
            "'claude' or 'gpt' model."
        )


def recognize_parts(cfg: DictConfig, verbose: bool = False):
    """
    Recognize the parts of an object from a set of images.
    Inspired from https://github.com/UMass-Embodied-AGI/Articulate-Anymesh.git
    """
    os.makedirs(cfg.out_dir, exist_ok=True)
    if Path(os.path.join(cfg.out_dir, "result_recognize_parts.json")).exists() and not cfg.rerun:
        return json.load(open(os.path.join(cfg.out_dir, "result_recognize_parts.json")))
    if verbose:
        print(f"Recognizing parts")
    image_dir = cfg.image_dir
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(tuple(IMAGE_EXTENSIONS))]
    user_prompt, system_prompt = recognize_parts_from_image(image_paths, cfg.object_name)

    model = make_vlm(cfg, verbose=verbose)

    result = model(user_prompt, system_prompt, image_paths=image_paths)
    
    result_text = model.get_result_text(result)

    with open(os.path.join(cfg.out_dir, "result_recognize_parts.txt"), "w") as f:
        f.write(result_text)
    
    result_json = extract_json_from_response(result_text)
    with open(os.path.join(cfg.out_dir, "result_recognize_parts.json"), "w") as f:
        json.dump(result_json, f, indent=4)

    vlm_query = {
        "user_prompt": user_prompt,
        "system_prompt": system_prompt,
        "image_paths": image_paths,
    }

    with open(os.path.join(cfg.out_dir, "vlm_query_recognize_parts.json"), "w") as f:
        json.dump(vlm_query, f, indent=4)

    return extract_json_from_response(result_text)

def generate_articulation_tree(cfg: DictConfig, parts_dict, verbose: bool = False):
    """
    Generate the articulation tree of an object from a set of images.
    Inspired from https://github.com/UMass-Embodied-AGI/Articulate-Anymesh.git
    """
    os.makedirs(cfg.out_dir, exist_ok=True)
    if Path(os.path.join(cfg.out_dir, "result_generate_articulation_tree.json")).exists() and not cfg.rerun:
        return json.load(open(os.path.join(cfg.out_dir, "result_generate_articulation_tree.json")))
    
    if verbose:
        print(f"Generating articulation tree")
    image_dir = cfg.image_dir
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(tuple(IMAGE_EXTENSIONS))]
    parts_list = [part['part_name'] for part in parts_dict['part_list']] + [parts_dict['fixed_part_name']]
    user_prompt, system_prompt = generate_articulation_tree_known_parts(cfg.object_name, parts_list)

    model = make_vlm(cfg, verbose=verbose)

    result = model(user_prompt, system_prompt, image_paths=image_paths)
    result_text = model.get_result_text(result)
    result_json = extract_json_from_response(result_text)
    result_json['fixed_part_name'] = parts_dict['fixed_part_name']
    with open(os.path.join(cfg.out_dir, "result_generate_articulation_tree.json"), "w") as f:
        json.dump(result_json, f, indent=4)
    
    with open(os.path.join(cfg.out_dir, "result_generate_articulation_tree.txt"), "w") as f:
        f.write(result_text)


        vlm_query = {
        "user_prompt": user_prompt,
        "system_prompt": system_prompt,
        "image_paths": image_paths,
    }

    with open(os.path.join(cfg.out_dir, "vlm_query_generate_articulation_tree.json"), "w") as f:
        json.dump(vlm_query, f, indent=4)

    return result_json

def merge_mesh_parts(cfg: DictConfig, parts_list, verbose: bool = False):
    """
    Merge the mesh parts of an object from a set of images.
    """
    os.makedirs(cfg.out_dir, exist_ok=True)
    
    # Check for cached result
    cache_path = os.path.join(cfg.out_dir, "merge_result.json")
    if Path(cache_path).exists() and not cfg.rerun:
        if verbose:
            print(f"Loading cached merge result from: {cache_path}")
        return json.load(open(cache_path))
    
    # Which render variant the merge sees. Both variants carry the same numeric tags and leader
    # lines; they differ only in surface colour:
    #   "segmented" (cfg.image_dir)        - each primitive a flat distinct colour, so its
    #                                        extent, boundary and relative size are visible
    #   "original_colors" (subdirectory)   - the object's true appearance
    # The merge question is "which primitives belong to the same part?", which is about regions
    # and adjacency. On a dark or uniformly coloured object the original-colour render shows the
    # tags but not the regions, so the model is asked about areas while looking at points.
    # Default to "segmented"; set merge_render_variant to override.
    variant = cfg.get("merge_render_variant", "segmented")
    image_dir = cfg.image_dir if variant == "segmented" else os.path.join(cfg.image_dir, variant)
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(
            f"merge_render_variant='{variant}' resolved to '{image_dir}', which does not exist. "
            f"Expected 'segmented' (renders directly in {cfg.image_dir}) or a subdirectory name "
            f"such as 'original_colors'."
        )
    image_paths = sorted(
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith(tuple(IMAGE_EXTENSIONS))
    )
    if not image_paths:
        raise FileNotFoundError(f"No renders found in '{image_dir}' for the part merge.")
    if verbose:
        print(f"  Merge renders: {len(image_paths)} from '{variant}' ({image_dir})")
    user_prompt, system_prompt = merge_parts(cfg.object_name, parts_list)

    model = make_vlm(cfg, verbose=verbose)

    result = model(user_prompt, system_prompt, image_paths=image_paths)
    result_text = model.get_result_text(result)
    result_json = extract_json_from_response(result_text)
    with open(os.path.join(cfg.out_dir, "merge_result.json"), "w") as f:
        json.dump(result_json, f, indent=4)
    
    with open(os.path.join(cfg.out_dir, "merge_result.txt"), "w") as f:
        f.write(result_text)

        vlm_query = {
        "user_prompt": user_prompt,
        "system_prompt": system_prompt,
        "image_paths": image_paths,
    }

    with open(os.path.join(cfg.out_dir, "vlm_query.json"), "w") as f:
        json.dump(vlm_query, f, indent=4)

    return result_json