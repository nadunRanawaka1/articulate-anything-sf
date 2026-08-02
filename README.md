<p align="center">
  <img src="assets/Figures/emojis/monkey_video.png" width="150px" />
</p>


<div align="center">

# Articulate Anything: Automatic Modeling of Articulated Objects via a Vision-Language Foundation Model
# ICLR 2025


[![Python Version](https://img.shields.io/badge/Python-3.9-blue.svg)](https://github.com/vlongle/articulate-anything)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2401.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2410.13882)
[![Dataset](https://img.shields.io/badge/🤗_Dataset-Hugging_Face-orange.svg)](https://huggingface.co/datasets/vlongle/articulate-anything-dataset-preprocessed/tree/main)
[![Twitter follow](https://img.shields.io/twitter/follow/LongLeRobot?style=social&label=follow)](https://twitter.com/LongLeRobot)


 <div align="center" margin-bottom="6em">
         <span class="author-block">
                <a target="_blank" rel="noopener noreferrer" href="https://vlongle.github.io">Long Le</a>
              </span>
            <a target="_blank" rel="noopener noreferrer" href="https://www.jchunx.dev/">Jason Xie</a>,</span>
              <span class="author-block">
                <a target="_blank" rel="noopener noreferrer" href="https://willjhliang.github.io/">William
                  Liang</a>,</span>
              <span class="author-block">
                <a target="_blank" rel="noopener noreferrer" href="https://johnny-wang16.github.io/">Hung-Ju
                  Wang</a>,</span>
              <span class="author-block">
                <a target="_blank" rel="noopener noreferrer" href="https://yueyang1996.github.io/">Yue Yang</a>,
              </span> <br>
              <span class="author-block">
                <a target="_blank" rel="noopener noreferrer" href="https://jasonma2016.github.io/">Jason Ma</a>,
              </span>
              <span class="author-block">
                <a target="_blank" rel="noopener noreferrer" href="https://vedder.io/">Kyle Vedder</a>,
              </span>
              <span class="author-block">
                <a target="_blank" rel="noopener noreferrer" href="https://arjun-krishna.github.io/">Arjun Krishna</a>,
              </span>
              <span class="author-block">
                <a target="_blank" rel="noopener noreferrer" href="https://www.seas.upenn.edu/~dineshj/">Dinesh
                  Jayaraman</a>,
              </span>
              <span class="author-block">
                <a target="_blank" rel="noopener noreferrer" href="https://www.seas.upenn.edu/~eeaton/">Eric Eaton</a>
              </span>
   <br>
   University of Pennsylvania
</div>


[[Project Website]](https://articulate-anything.github.io)
[[Paper]](https://arxiv.org/abs/2410.13882)
[[Twitter threads]](https://x.com/int64_le/status/1866519866934714623)

______________________________________________________________________

> [!NOTE]
> ### SimFoundry fork
> This is the [SimFoundry](https://github.com/NVlabs/simfoundry) fork of Articulate-Anything. It adds a
> **SimFoundry integration** (under [`simfoundry/`](simfoundry/)) that runs articulation as a stage of the SimFoundry real2sim
> pipeline on in-the-wild reconstructed meshes, on top of the original PartNet-Mobility workflow.
>
> What differs from upstream:
> - **VLM backend:** all VLM calls run on **Google Vertex AI** — either **Gemini** or **Anthropic Claude**
>   (see [Choosing a VLM](#choosing-a-vlm)). Set `GCLOUD_PROJECT` and pick a model with `model_name`
>   (see [`conf/config_simfoundry.yaml`](conf/config_simfoundry.yaml)). One set of gcloud credentials covers both;
>   no separate API keys or proprietary/hosted inference services are required.
> - **Mesh part-segmentation** has three interchangeable backends selected by `segment_method`:
>   [`samesh`](https://github.com/gtangg12/samesh), [`Hunyuan3D-Part`](https://github.com/Tencent-Hunyuan/Hunyuan3D-Part),
>   and [`PartField`](https://github.com/nv-tlabs/PartField).
> - **Install** uses the per-backend scripts described in [Installation](#installation) (separate conda envs).
>
> The upstream documentation below is preserved and still applies to the standalone
> text / image / video articulation workflow.

Articulate Anything is a powerful VLM system for articulating 3D objects using various input modalities.



https://github.com/user-attachments/assets/3c23f423-3bdd-4843-a4e3-e1c7f26bfc42




</div>

## Features

- <img src="assets/Figures/emojis/monkey_text.png" alt="Text Input" width="50" style="vertical-align: middle;"/> Articulate 3D objects from text 🖋 descriptions
- <img src="assets/Figures/emojis/monkey_image.png" alt="Image Input" width="50" style="vertical-align: middle;"/> Articulate 3D objects from 🖼 images
- <img src="assets/Figures/emojis/monkey_video.png" alt="Video Input" width="50" style="vertical-align: middle;"/> Articulate 3D objects from 🎥  videos

We use [Hydra](https://hydra.cc/) for configuration management. You can easily customize the system by modifying the configuration files in `configs/` or overload parameters from the command line. We can automatically articulate a variety of input modalities from a single command

```bash
   python articulate.py modality={partnet, text, image, video} prompt={prompt} out_dir={output_dir}
```
Articulate-anything uses a **actor-critic** system, allowing for self-correction and self-improvement over iterations. 


## 🚀 QUICK START
1. Download preprocessed PartNet-Mobility dataset from 🤗 [Articulate-Anything Dataset on Hugging Face](https://huggingface.co/datasets/vlongle/articulate-anything-dataset-preprocessed/tree/main).

2. To use an interactive demo, run
   ```bash
   python gradio_app.py
   ```




https://github.com/user-attachments/assets/3b8edddd-3c26-4691-a9b6-8bfc6c8f4a8d




See below for more detailed guides.

## Table of Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Usage](#usage)
  - [Demo](#demo)
  - [PartNet-Mobility Masked Reconstruction](#partnet-mobility-masked-reconstruction)
  - [Text Articulation](#text-articulation)
  - [Visual Articulation](#visual-articulation)
- [Notes](#notes)
- [Contact](#contact)
- [Citation](#citation)

## Installation

> [!NOTE]  
> Skip the downloading raw dataset step if you have already downloaded our dataset from 🤗 [Articulate-Anything Dataset on Hugging Face](https://huggingface.co/datasets/vlongle/articulate-anything-dataset-preprocessed/tree/main).


### SimFoundry integration install

The SimFoundry integration builds a separate conda env per mesh-segmentation backend. From the repo root:

```bash
bash installation_hunyuan.sh     # env: articulate-anything-hunyuan  (py3.10) — Hunyuan3D-Part  [DEFAULT]
bash installation_samesh.sh      # env: articulate-anything-samesh   (py3.11) — SAMesh + SAM2 + CoTracker  (optional)
bash installation_partfield.sh   # env: articulate-anything-partfield(py3.10) — PartField  (optional)
```

**Hunyuan3D-Part is the default segmentation backend**; SAMesh and PartField are optional and only run
when explicitly selected via `segment_method` (`hunyuan` / `samesh` / `partfield`). Install just the
backend(s) you need.

> [!NOTE]
> **The segmentation backends are fetched from their public upstreams, then patched.** Each
> `installation_*.sh` clones the public upstream at a pinned commit and applies a patch from
> [`patches/`](patches/) (see [`patches/README.md`](patches/README.md)).
> Model weights and large demo assets are fetched separately, not shipped in the patches.

Set your Vertex AI project (read by [`conf/config_simfoundry.yaml`](conf/config_simfoundry.yaml)) and authenticate:

```bash
export GCLOUD_PROJECT=<your-gcp-project>
gcloud auth application-default login
```

### Upstream (standalone) install
> [!NOTE] We have kept these instructions, but you do not need to run them for SimFoundry workflows.
For the standalone text / image / video articulation workflow:

1. Set up the Python environment:
   ```bash
   conda create -n articulate-anything python=3.9
   conda activate articulate-anything
   pip install -e .
   ```

2. Download and extract the PartNet-Mobility dataset:
   ```bash
   # Download from https://sapien.ucsd.edu/downloads
   mkdir datasets
   mv partnet-mobility-v0.zip datasets/partnet-mobility-v0.zip
   cd datasets
   mkdir partnet-mobility-v0
   unzip partnet-mobility-v0 -d partnet-mobility-v0
   ```

## Getting Started

### Choosing a VLM

Our system supports **Google Gemini**, **Anthropic Claude**, and OpenAI GPT. Gemini and Claude both run on
**Vertex AI** under the same gcloud credentials, so switching between them is a one-line config change.

Set your project and authenticate with Application Default Credentials:

   ```bash
   export GCLOUD_PROJECT=<your-gcp-project>
   gcloud auth application-default login
   ```

Then set `model_name` in the config file. There are three independent `model_name` settings:

| Setting | Config file | Drives |
| --- | --- | --- |
| `s2_generate_articulation_tree.model_name` | `simfoundry/cfg/<your-config>.yaml` | Step 2 — part recognition + articulation tree |
| `s4_merge_mesh_parts.model_name` | `simfoundry/cfg/<your-config>.yaml` | Step 4 — merging segmented parts |
| `s5_articulate.model_name` | `simfoundry/cfg/<your-config>.yaml` | Step 5 — the joint actor / critic loop (optional; overrides the row below) |
| `model_name` | [`conf/config_simfoundry.yaml`](conf/config_simfoundry.yaml) | Step 5 default, shared by all scene configs |

Step 5 normally takes its model from `conf/config_simfoundry.yaml`, which is shared by every scene. Setting
`s5_articulate.model_name` in a scene config overrides it for that scene only — useful for running the same
object on two models without duplicating the articulation config. `s5_articulate.vlm_backend` and
`s5_articulate.claude_location` can be overridden the same way.
[`simfoundry/cfg/red_mailbox_claude.yaml`](simfoundry/cfg/red_mailbox_claude.yaml) is a worked example: the
same mailbox as `red_mailbox.yaml`, articulated by Claude Opus 5 instead of Gemini.

For the standalone (non-SimFoundry) workflow, use `model_name` in [`conf/config.yaml`](conf/config.yaml).

Supported values:

| Provider | Model ids | Notes |
| --- | --- | --- |
| Gemini | `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-3.1-pro-preview` | Default. Accepts video. |
| Claude | `claude-opus-5`, `claude-opus-4-8`, `claude-sonnet-5`, `claude-haiku-4-5` | Text / images / PDF only — **no video** (see below). |
| GPT | `gpt-4o`, `o3` | Needs `OPENAI_API_KEY`. |

#### Using Claude Opus on Vertex AI

```yaml
# conf/config_simfoundry.yaml  (step 5: joint actor + critic)
model_name: claude-opus-5
vlm_backend: vertex     # gcloud ADC + gcloud_project; no Anthropic API key
claude_location: global # Claude is not served in every Gemini region
```

Notes:

- **Region.** `claude_location` overrides `gcloud_location` for Claude only, because the two model families
  are not served in the same set of Vertex regions (`gcloud_location: us-west1` serves Gemini but not Claude).
  `global` is the safe default.
- **Quota.** Claude models on Vertex have their own per-project quota
  (`aiplatform.googleapis.com/global_online_prediction_requests_per_base_model`), which is **0 until you
  request an increase**. A `429 RESOURCE_EXHAUSTED ... base model: anthropic-claude-opus` means quota, not a
  misconfiguration — request it from
  [Vertex AI generative AI quotas](https://cloud.google.com/vertex-ai/docs/generative-ai/quotas-genai).
- **No video input.** Claude accepts text, images and PDFs, but not video. The `simfoundry_video` joint critic
  normally uploads the prediction MP4; on a Claude model it automatically samples frames from that video
  instead (and switches to the matching frame-by-frame system prompt), so `joint_critic.type` needs no change.
  Ground-truth input for the SimFoundry pipeline is already a still image, so nothing else is affected.
- **Sampling parameters.** Claude Opus 5 / Opus 4.8 / Sonnet 5 reject `temperature` / `top_p` / `top_k`.
  Response style is steered with the prompt, and cost/quality with `effort` (`GEN_CONFIG_CLAUDE` in
  [`articulate_anything/agent/agent.py`](articulate_anything/agent/agent.py)).
- **Direct Anthropic API.** Set `vlm_backend: anthropic` and `ANTHROPIC_API_KEY` to bypass Vertex.

## Usage

We support reconstruction from in-the-wild text, images, or videos, or masked reconstruction from PartNet-Mobility dataset.

> [!NOTE]  
> Skip all the processing steps if you have downloaded our preprocessed dataset from 🤗 [Articulate-Anything Dataset on Hugging Face](https://huggingface.co/datasets/vlongle/articulate-anything-dataset-preprocessed/tree/main).



<h3 id="demo">Demo</h3>

1. First, preprocess the parntet dataset by running
   ```bash
   python preprocess_partnet.py parallel={int} modality={}
   ```
2. Run the interactive demo
   ```bash
   python gradio_app.py
   ```



<h3 id="partnet-mobility-masked-reconstruction">💾 PartNet-Mobility Masked Reconstruction</h3>


🐒 It's articulation time! For a step-by-step guide on articulating a PartNet-Mobility object, see the notebook:

   [<img align="center" src="assets/Figures/jupyter-logo.svg" width="20"/> Open in Jupyter Notebook](examples/articulate_partnet.ipynb)

   or run

   ```bash
     python articulate.py modality=partnet prompt=45384 out_dir=results additional_prompt=joint_0
   ```
to run for `object_id`=149.

<h3 id="text-articulation">🖋 Text Articulation </h3>

1. Preprocess the dataset:
   ```bash
   python articulate_anything/preprocess/preprocess_partnet.py parallel={int} modality=text
   ```

Our precomputed CLIP embeddings is available from our repo in `partnet_mobility_embeddings.csv`. If you prefer to generate your own embeddings, follow these steps:

1. Run the preprocessing with `render_aprt_views=true` to render part views for later part annotation.
```bash
   python articulate_anything/preprocess/preprocess_partnet.py parallel={int} modality=text render_part_views=true 
```
2. Annotate mesh parts using VLM (skip if using our precomputed embeddings):
   ```bash
   python articulate_anything/preprocess/annotate_partnet_parts.py parallel={int}
   ```
3. Extract CLIP embeddings (skip if using our precomputed embeddings):
   ```bash
   python articulate_anything/preprocess/create_partnet_embeddings.py
   ```

4. 🐒 It's articulation time!  For a detailed guide, see:

   [<img align="center" src="assets/Figures/jupyter-logo.svg" width="20"/> Open in Jupyter Notebook](examples/articulate_text.ipynb)

   or run 

   ```bash
   python articulate.py modality=text  prompt="suitcase with a retractable handle" out_dir=results/text/suitcase joint_actor.targetted_affordance=false
   ```

<h3 id="visual-articulation">🖼 / 🎥 Visual Articulation </h3>

1. Render images for each object:
   ```bash
   python articulate_anything/preprocess/preprocess_partnet.py parallel={int} modality={image}
   ```
   This renders a front-view image for each object in the PartNet-Mobility dataset. This is necessary for our mesh retrieval as we will compare the visual similarity between the input image or video against each rendered template object.


2. 🐒 It's articulation time!  For a detailed guide, see:

   [<img align="center" src="assets/Figures/jupyter-logo.svg" width="20"/> Open in Jupyter Notebook](examples/articulate_video.ipynb)

   or run 

   ```bash
   python articulate.py modality=video prompt="datasets/in-the-wild-dataset/videos/suitcase.mp4" out_dir=results/video/suitcase
   ```

Note: Please download a checkpoint of [cotracker](https://github.com/facebookresearch/co-tracker) for video articulation to visualize the motion traces.

## Notes

Some implementation pecularity with the PartNet-Mobility dataset.
- __Raise above ground__: The meshes are centered at origin `(0,0,0)`. We use `pybullet` to raise the links above the ground. Done automatically in `sapien_simulate`.
- __Rotate meshes__: All the meshes will be on the ground. We have to get them in the upright orientation. Specifically, we need to add a fixed joint `<origin rpy="1.570796326794897 0 1.570796326794897" xyz="0 0 0"/>` between the first link and the `base` link. This is almost done in the original PartNet-Mobility dataset. `render_partnet_obj` which calls `rotate_urdf` saves the original urdf under `mobility.urdf.backup` and get the correct rotation under `mobility.urdf`. Then, for our generated python program we need to make sure that the compiled python program also has this joint. This is done automatically by the compiler `odio_urdf.py` using `align_robot_orientation` function.

## Contact

Feel free to reach me at vlongle@seas.upenn.edu if you'd like to collaborate, or have any questions. You can also open a Github issue if you encounter any problems.

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{le2024articulate,
  title={Articulate-Anything: Automatic Modeling of Articulated Objects via a Vision-Language Foundation Model},
  author={Le, Long and Xie, Jason and Liang, William and Wang, Hung-Ju and Yang, Yue and Ma, Yecheng Jason and Vedder, Kyle and Krishna, Arjun and Jayaraman, Dinesh and Eaton, Eric},
  journal={arXiv preprint arXiv:2410.13882},
  year={2024}
}
```

For more information, visit our [project website](https://articulate-anything.github.io).
