# ReMindView-Bench
This repository contains the data generation pipeline and evaluation scripts for ReMindView-Bench for paper [Reasoning Path and Latent State Analysis for Multi-view Visual Spatial Reasoning](https://arxiv.org/abs/2512.02340)

![Pipeline overview](figures/figure1.png)

## Table of Contents
- [ReMindView-Bench](#remindview-bench)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Requirements](#requirements)
  - [Environment Setup](#environment-setup)
  - [Quickstart](#quickstart)
  - [Pipeline Overview](#pipeline-overview)
  - [Repository Map](#repository-map)
  - [Evaluation](#evaluation)
  - [Using the Generated Dataset](#using-the-generated-dataset)
  - [Citation](#citation)
  - [License](#license)

## Project Overview
- Paper: Reasoning Path and Latent State Analysis for Multi-view Visual Spatial Reasoning ([arXiv:2512.02340](https://arxiv.org/abs/2512.02340)).
- Dataset: ReMindView-Bench hosted on Hugging Face at [Xue0823/ReMindView-Bench](https://huggingface.co/datasets/Xue0823/ReMindView-Bench); scripts here reproduce the synthetic data.
- Figures in the repository mirror the organization of the generation pipeline (scene generation, rendering, QA construction).

## Requirements
- Blender: 3.6 LTS (bundled Python 3.10) or newer with `bpy==3.6.0` (included in infinigen environment).
- Python: 3.10 (matches Blender’s bundled interpreter).
- GPU: NVIDIA GPU with at least 12 GB VRAM recommended for rendering.
- OS: Linux or macOS; commands below assume Bash.
- Python dependencies: see `requirements.txt` (mirrors `environment.yml`) with tested versions for reproducibility.

## Environment Setup
1) Clone the repo and switch to the root directory.
2) Point to Blender’s bundled Python and install dependencies:
   ```bash
   export BLENDER_ROOT=/path/to/blender-3.6
   export BLENDER_BIN="$BLENDER_ROOT/blender"
   export BLENDER_PYTHON="$BLENDER_ROOT/3.6/python/bin/python3.10"

   "$BLENDER_PYTHON" -m pip install --upgrade pip
   "$BLENDER_PYTHON" -m pip install -r requirements.txt
   ```
3) Ensure imports resolve when running through Blender by adding the repo to `PYTHONPATH`:
   ```bash
   export PYTHONPATH="$PWD:$PYTHONPATH"
   ```
4) Example: run a script with Blender’s interpreter so `bpy` is available:
   ```bash
   "$BLENDER_BIN" --background --python view_centric_view_frame_generation.py -- \
     --config view_centric_view_frame_generation_config.json \
     --output_dir view_centric_view_frame_outputs/Bedroom/Bedroom_0 \
     --room_type bedroom \
     --scene_path outputs/indoors/Bedroom_0/scene.blend
   ```
   Paths above are the defaults produced by the generation script; command-line arguments stay exposed to let you override locations if you customize the pipeline.

## Quickstart
1) Generate scenes and renders (object-centric and view-centric) from the repo root:  
   ```bash
   bash scene_generation.sh
   ```  
   Scenes land in `outputs/indoors/<ROOM>_<SEED>`, object-centric frames in `object_centric_view_frame_outputs/<ROOM>/<ROOM>_<SEED>`, and view-centric frames in `view_centric_view_frame_outputs/<ROOM>/<ROOM>_<SEED>`.
2) Clean empty or invalid views (already invoked inside `scene_generation.sh`, rerun if you tweak outputs):  
   ```bash
   python clean_visual_data.py --dir_path object_centric_view_frame_outputs
   python clean_visual_data.py --dir_path view_centric_view_frame_outputs
   ```
3) Build QA CSVs (choose one of `view_view`, `view_object`, `object_object`):  
   ```bash
   python ground_truth_generation.py \
     --image_folder object_centric_view_frame_outputs \
     --qa_type object_object
   ```  
   Output is stored next to the image folder (e.g., `object_centric_view_frame_outputs/object_object_qa.csv`).
4) (Optional) Adjust outputs without editing scripts by passing flags (e.g., `--output_dir`, `--scene_path`). Defaults match the repo layout to minimize user error.

## Pipeline Overview
![Sample outputs](figures/figure2.png)
| Stage | Script(s) | Inputs | Outputs |
| --- | --- | --- | --- |
| Scene generation | `scene_generation.sh` (+ `infinigen_examples.generate_indoors`) | Seeds, room types, gin configs | Blender scenes under `outputs/indoors/<ROOM>_<SEED>` |
| View rendering | `object_centric_view_frame_generation.py`, `view_centric_view_frame_generation.py` | `scene.blend`, camera configs (`*_generation_config.json`) | Rendered frames and metadata under `object_centric_view_frame_outputs/` and `view_centric_view_frame_outputs/` |
| Visibility pruning | `clean_visual_data.py` | Rendered frame dirs | Pruned frame dirs without empty renders |
| Metadata extraction | `generate_object_list.py` | Rendered frame dirs | Object visibility summaries used for QA |
| VQA construction | `ground_truth_generation.py`, `ground_truth_generation.sh` | Visibility metadata, rename maps in `object_rename/`, `query_template.json` | Multiple-choice QA CSVs for view-view, view-object, and object-object tasks |

## Repository Map
```
ReMindView-Bench/
├─ scene_generation.sh                 # End-to-end loop over seeds/room types
├─ object_centric_view_frame_generation.py
├─ view_centric_view_frame_generation.py
├─ object_centric_view_frame_generation_config.json
├─ view_centric_view_frame_generation_config.json
├─ clean_visual_data.py
├─ generate_object_list.py
├─ ground_truth_generation.py
├─ ground_truth_generation.sh
├─ object_rename/                      # Rename/orientation metadata per room
├─ evaluation/                         # Model evaluation entrypoints (see notes)
├─ figures/                            # Paper figures used above
├─ environment.yml
└─ requirements.txt
```
The organization mirrors the paper sections (scene generation → rendering → QA). Files are listed once to avoid duplication between sections.

## Evaluation
Evaluation entrypoints live in `evaluation/` (`eval_models.sh`, `eval_internvl35.py`, `eval_llava_onevision.py`, `eval_qwen_vl.py`). Some evaluation code paths are intentionally left minimal; add checkpoints and dataset paths before use. Additional evaluation scripts will be added here as they are released.

## Using the Generated Dataset
Download the dataset from [Hugging Face](https://huggingface.co/datasets/Xue0823/ReMindView-Bench) or produce it locally via the pipeline above. Simple example to load QA pairs for a VQA task:
```python
import pandas as pd

qa = pd.read_csv("object_centric_view_frame_outputs/object_object_qa.csv")
print(qa.head())
```
Each row includes the frame folder, query text, ground truth, and choices, so you can pair images from the same folder for VQA training or evaluation.

## Citation
If you use ReMindView-Bench, please cite the accompanying paper:
```bibtex
@article{xue2025reasoning,
  title={Reasoning Path and Latent State Analysis for Multi-view Visual Spatial Reasoning: A Cognitive Science Perspective},
  author={Xue, Qiyao and Liu, Weichen and Wang, Shiqi and Wang, Haoming and Wu, Yuyang and Gao, Wei},
  journal={arXiv preprint arXiv:2512.02340},
  year={2025}
}
```

## License
See `LICENSE` for usage terms. A permissive research-friendly license will be finalized; until then, reach out to the authors for questions about reuse.
