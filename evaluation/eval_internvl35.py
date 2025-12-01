#!/usr/bin/env python3
import os, re, time, argparse, warnings, json
from pathlib import Path
from typing import List, Tuple, Dict, Optional, NamedTuple

import torch
import pandas as pd
from PIL import Image
from transformers import pipeline

# ============================
# Speed knobs (safe defaults)
# ============================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", category=UserWarning)

# ============================
# CSV / parsing helpers
# ============================
LABELS = ["A", "B", "C", "D"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CHOICE_LINE_RE = re.compile(r"\b([ABCD])\s*\.\s*([^\n\r]+)")

def validate_csv(df: pd.DataFrame):
    req = ["folder_path","query_type","query","ground_truth","choices","cross_frame","perspective_changing","object_num"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    def blank(x): return (pd.isna(x)) or (str(x).strip() == "")
    bad = []
    for i, r in df.iterrows():
        errs = []
        if blank(r["folder_path"]): errs.append("folder_path")
        if blank(r["query_type"]):  errs.append("query_type")
        if blank(r["query"]):       errs.append("query")
        if blank(r["choices"]):     errs.append("choices")
        if errs: bad.append((i, errs))
    if bad:
        msg = "\n".join([f"  Row {i}: missing {errs}" for i, errs in bad[:10]])
        raise ValueError("Bad rows detected:\n" + msg + ("\n... truncated)" if len(bad) > 10 else ""))

def parse_query_type(qt_raw: str):
    parts = [p for p in str(qt_raw or "").split("|") if p]
    scope = parts[0] if len(parts) > 0 else ""
    kind  = parts[1] if len(parts) > 1 else ""
    return scope, kind, "|".join([p for p in (scope, kind) if p])

def split_choices(choices_str: str) -> List[str]:
    cs = [c.strip() for c in str(choices_str or "").split(",") if c.strip()]
    if len(cs) < 2:
        raise ValueError(f"Need at least 2 choices; got: {cs}")
    return cs[:4]

def normalize_ground_truth(gt: str, choices: List[str]) -> Tuple[str, str]:
    s = str(gt or "").strip()
    m = CHOICE_LINE_RE.fullmatch(s)
    if m: return m.group(1), m.group(2).strip()
    for i, c in enumerate(choices[:4]):
        if s.lower() == c.lower():
            return LABELS[i], c
    return "", ""

# ============================
# System prompts (unchanged content)
# ============================
SYSTEM_PROMPT = {
    "object_centric_object_object":
    """You are a reasoning-focused vision-language model. You will be given multiple image frames from different viewpoints and a spatial reasoning question. Your goal is to reason step-by-step across these frames before giving the final answer. Please strictly follow the structured reasoning path below. Each stage should be clearly written and numbered.

    ### Reasoning Path Template###
    (1) General visual description in each frame
    - Describe what you see in every frame separately. Focus on horizontial spatial relationships, avoid style or non-spatial details. Use consistent object names on the same object. The direction should only discribed in 8 relationships: front, front-right, right, back-right, back, back-left, left, front-left.
    (2) Intermediate reasoning step for connecting cross-frame relationships
    - Describe frame taken position's horizontial spatial relationships based on object spatial relationship observed in (1).
    (3) Query-specific spatial reasoning step
    - Use the above information to perform spatial reasoning about the specific question by fully comprehend the horizontial spatial relationships of all objects in the entire scene.
    (4) Final answer
    - Only output the final choice label in the format: 
    A.<answer> or B.<answer> or C.<answer> or D.<answer>
    - Do NOT include any explanation, text, or additional reasoning in this step.

    ### Example Output###
    (1) In frame1: the chair is in front of the table. In frame2: the chair is to the right of the table. In frame3: the cell shelf is in front of the table.
    (2) By analyzing the spatial changes of the objects across frames, I establish their relationships: frame2 is taken from the front-right of frame1; frame3 is taken from the front-right of frame2; overall, frame3 is positioned in front of frame1.
    (3) Based on the spatial information in (1) and (2), I construct a complete spatial understanding. From frame2 perspective, the chair is to the right of the table, and the cell shelf is to the left of the table.
    (4) A.left""",

    "object_centric_view_object":
    """You are a reasoning-focused vision-language model. You will be given multiple image frames from different viewpoints and a spatial reasoning question. Your goal is to reason step-by-step across these frames before giving the final answer. Please strictly follow the structured reasoning path below. Each stage should be clearly written and numbered.

    ### Reasoning Path Template###
    (1) General visual description in each frame
    - Describe what you see in every frame separately. Focus on horizontial spatial relationships, avoid style or non-spatial details. Use consistent object names on the same object. The direction should only discribed in 8 relationships: front, front-right, right, back-right, back, back-left, left, front-left.
    (2) Intermediate reasoning step for connecting cross-frame relationships
    - Describe frame taken position's horizontial spatial relationships based on object spatial relationship observed in (1).
    (3) Query-specific spatial reasoning step
    - Use the above information to perform spatial reasoning about the specific question by fully comprehend the horizontial spatial relationships of all objects in the entire scene.
    (4) Final answer
    - Only output the final choice label in the format: 
    A.<answer> or B.<answer> or C.<answer> or D.<answer>
    - Do NOT include any explanation, text, or additional reasoning in this step.

    ### Example Output###
    (1) In frame1: the chair is in front of the table. In frame2: the chair is to the right of the table. In frame3: the cell shelf is in front of the table.
    (2) By analyzing the spatial changes of the objects across frames, I establish their relationships: frame2 is taken from the front-right of frame1; frame3 is taken from the front-right of frame2; overall, frame3 is positioned in front of frame1.
    (3) Based on the spatial information in (1) and (2), I construct a complete spatial understanding. From frame2 perspective, the chair is to the right of my position, and the cell shelf is to the left of my position.
    (4) A.left""",

    "object_centric_view_view":
    """You are a reasoning-focused vision-language model. You will be given multiple image frames from different viewpoints and a spatial reasoning question. Your goal is to reason step-by-step across these frames before giving the final answer. Please strictly follow the structured reasoning path below. Each stage should be clearly written and numbered.

    ### Reasoning Path Template###
    (1) General visual description in each frame
    - Describe what you see in every frame separately. Focus on horizontial spatial relationships, avoid style or non-spatial details. Use consistent object names on the same object. The direction should only discribed in 8 relationships: front, front-right, right, back-right, back, back-left, left, front-left.
    (2) Intermediate reasoning step for connecting cross-frame relationships
    - Describe frame taken position's horizontial spatial relationships based on object spatial relationship observed in (1).
    (3) Query-specific spatial reasoning step
    - Use the above information to perform spatial reasoning about the specific question by fully comprehend the horizontial spatial relationships of all objects in the entire scene.
    (4) Final answer
    - Only output the final choice label in the format: 
    A.<answer> or B.<answer> or C.<answer> or D.<answer>
    - Do NOT include any explanation, text, or additional reasoning in this step.

    ### Example Output###
    (1) In frame1: the chair is in front of the table. In frame2: the chair is to the right of the table. In frame3: the cell shelf is in front of the table.
    (2) By analyzing the spatial changes of the objects across frames, I establish their relationships: frame2 is taken from the front-right of frame1; frame3 is taken from the front-right of frame2; overall, frame3 is positioned in front of frame1.
    (3) Based on the spatial information in (1) and (2), I construct a complete spatial understanding. Frame 1 is opposite to frame 3, frame 1 is front-left of frame 2, and frame 2 is front-left of frame 3.
    (4) A.left""",

    "view_centric_view_object":
    """You are a reasoning-focused vision-language model. You will be given multiple image frames from different viewpoints and a spatial reasoning question. The provided images are taken at the same position, frame1 is taken by turning 90 degree counter-clockwise from frame0 view, frame2 is taken by turning 180 degree counter-clockwise from frame0 view, frame3 is taken by turning 90 degree clockwise from frame0 view. Your goal is to reason step-by-step across these frames before giving the final answer. Please strictly follow the structured reasoning path below. Each stage should be clearly written and numbered.

    ### Reasoning Path Template###
    (1) General visual description in each frame
    - Describe what you see in every frame separately. Focus on horizontial spatial relationships, avoid style or non-spatial details. Use consistent object names on the same object. The direction should only discribed in 8 relationships: front, front-right, right, back-right, back, back-left, left, front-left.
    (2) Intermediate reasoning step for connecting cross-frame relationships
    - We know that the provided images are taken at the same position, frame1 is taken by turning 90 degree counter-clockwise from frame0 view, frame2 is taken by turning 180 degree counter-clockwise from frame0 view, frame3 is taken by turning 90 degree clockwise from frame0 view.
    (3) Query-specific spatial reasoning step
    - Use the above information to perform spatial reasoning about the specific question by fully comprehend the horizontial spatial relationships of all objects in the entire scene.
    (4) Final answer
    - Only output the final choice label in the format: 
    A.<answer> or B.<answer> or C.<answer> or D.<answer>
    - Do NOT include any explanation, text, or additional reasoning in this step.

    ### Example Output###
    (1) In frame1: the chair is in front of the table. In frame2: the chair is to the right of the table. In frame3: the cell shelf is in front of the table.
    (2) The provided images are taken at the same position, frame1 is taken by turning 90 degree counter-clockwise from frame0 view, frame2 is taken by turning 180 degree counter-clockwise from frame0 view, frame3 is taken by turning 90 degree clockwise from frame0 view.
    (3) Based on the spatial information in (1) and (2), I construct a complete spatial understanding. From frame2 perspective, the chair is to the right of my position, and the cell shelf is to the left of my position.
    (4) A.left""",

    "view_centric_object_object":
    """You are a reasoning-focused vision-language model. You will be given multiple image frames from different viewpoints and a spatial reasoning question. The provided images are taken at the same position, frame1 is taken by turning 90 degree counter-clockwise from frame0 view, frame2 is taken by turning 180 degree counter-clockwise from frame0 view, frame3 is taken by turning 90 degree clockwise from frame0 view. Your goal is to reason step-by-step across these frames before giving the final answer. Please strictly follow the structured reasoning path below. Each stage should be clearly written and numbered.

    ### Reasoning Path Template###
    (1) General visual description in each frame
    - Describe what you see in every frame separately. Focus on horizontial spatial relationships, avoid style or non-spatial details. Use consistent object names on the same object. The direction should only discribed in 8 relationships: front, front-right, right, back-right, back, back-left, left, front-left.
    (2) Intermediate reasoning step for connecting cross-frame relationships
    - We know that the provided images are taken at the same position, frame1 is taken by turning 90 degree counter-clockwise from frame0 view, frame2 is taken by turning 180 degree counter-clockwise from frame0 view, frame3 is taken by turning 90 degree clockwise from frame0 view.
    (3) Query-specific spatial reasoning step
    - Use the above information to perform spatial reasoning about the specific question by fully comprehend the horizontial spatial relationships of all objects in the entire scene.
    (4) Final answer
    - Only output the final choice label in the format: 
    A.<answer> or B.<answer> or C.<answer> or D.<answer>
    - Do NOT include any explanation, text, or additional reasoning in this step.

    ### Example Output###
    (1) In frame1: the chair is in front of the table. In frame2: the chair is to the right of the table. In frame3: the cell shelf is in front of the table.
    (2) The provided images are taken at the same position, frame1 is taken by turning 90 degree counter-clockwise from frame0 view, frame2 is taken by turning 180 degree counter-clockwise from frame0 view, frame3 is taken by turning 90 degree clockwise from frame0 view.
    (3) Based on the spatial information in (1) and (2), I construct a complete spatial understanding. From frame2 perspective, the chair is to the right of the table, and the cell shelf is to the left of the table.
    (4) A.left""",
}

def choose_system_prompt(csv_path: str) -> str:
    for k in SYSTEM_PROMPT.keys():
        if k in csv_path:
            return SYSTEM_PROMPT[k]
    return list(SYSTEM_PROMPT.values())[0]

# ============================
# Images (bind names directly)
# ============================
class ImgWithName(NamedTuple):
    image: Image.Image
    name: str

def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]

def list_images_cached(folder_path: str, cache: Dict[str, List[ImgWithName]]) -> List[ImgWithName]:
    if folder_path in cache:
        return cache[folder_path]
    p = Path(folder_path)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"folder_path not found or not a directory: {folder_path}")
    files = [x for x in p.iterdir() if x.suffix.lower() in IMG_EXTS]
    files.sort(key=lambda x: natural_key(x.name))
    if not files:
        raise FileNotFoundError(f"No images found under: {folder_path}")
    out: List[ImgWithName] = []
    for f in files:
        with Image.open(f) as im:
            out.append(ImgWithName(im.convert("RGB").copy(), f.stem + ":"))
    cache[folder_path] = out
    return out

# ============================
# Prompt builders -> HF messages
# ============================
def build_user_text(question: str, choices: List[str], query_type: str, cross_frame, perspective_changing, object_num) -> str:
    choice_lines = "\n".join(choices)
    tail = "Follow the Reasoning Path Template strictly. Your last line must match ^[ABCD]\\.[^\\n\\r]+$ and contain only one choice."
    return f"Question: {question}\nChoices:\n{choice_lines}\n\n{tail}"

def build_messages_for_pipeline(pil_images_with_names: List[ImgWithName], user_text: str, csv_path: str):
    """
    Build HF pipeline messages:
    [
      {"role": "system", "content": [{"type":"text","text": <system_prompt>}]},
      {"role": "user", "content": [{"type":"image","image": PIL}, ..., {"type":"text","text": user_prompt}]}
    ]
    """
    sys_prompt = choose_system_prompt(csv_path)
    content = []
    for it in pil_images_with_names:
        content.append({"type": "image", "image": it.image})
    content.append({"type": "text", "text": user_text})
    return [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {"role": "user",   "content": content},
    ]

# ============================
# Output normalization & parsing
# ============================
def _ensure_string_text(obj) -> str:
    """
    Robustly extract a human string from varied pipeline outputs.
    Handles: str, dict with 'generated_text'/'text'/'answer'/'content',
    lists (including [{"generated_text": ...}] or [{"type":"text","text":...}]).
    """
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj

    if isinstance(obj, list):
        if not obj:
            return ""
        return _ensure_string_text(obj[0])

    if isinstance(obj, dict):
        if "generated_text" in obj:
            return _ensure_string_text(obj["generated_text"])
        for k in ("text", "answer", "output"):
            if k in obj:
                return _ensure_string_text(obj[k])
        if "content" in obj and isinstance(obj["content"], list):
            parts = []
            for it in obj["content"]:
                if isinstance(it, dict) and it.get("type") == "text" and "text" in it:
                    parts.append(str(it["text"]))
            if parts:
                return "\n".join(parts)
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return str(obj)

    return str(obj)

def parse_prediction(generated) -> Tuple[str, str]:
    if generated is None:
        return "INVALID", ""
    if not isinstance(generated, str):
        generated = _ensure_string_text(generated)

    matches = list(CHOICE_LINE_RE.finditer(generated))
    if matches:
        m = matches[-1]; return m.group(1), m.group(2).strip()
    single_labels = re.findall(r"\b([ABCD])\b", generated)
    if single_labels: return single_labels[-1], "unknown"
    return "INVALID", ""

# ============================
# HF pipeline loader
# ============================
def _parse_device_arg(device_str: str):
    ds = (device_str or "").lower()
    if ds.startswith("cuda"):
        if ":" in ds:
            try:
                return int(ds.split(":")[1])
            except Exception:
                return 0
        return 0
    if ds.startswith("mps"):
        return "mps"
    return "cpu"

def load_pipe(model: str, device: str, torch_dtype: str):
    dtype_map = {"auto": None, "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map.get(torch_dtype, None)
    dev = _parse_device_arg(device)
    return pipeline(
        task="image-text-to-text",
        model=model,
        trust_remote_code=True,
        device=dev,
        torch_dtype=dtype if dtype is not None else None,
    )

# ============================
# Main
# ============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV with: folder_path,query_type,query,ground_truth,choices,cross_frame,perspective_changing,object_num")
    parser.add_argument("--model", default="OpenGVLab/InternVL3_5-1B-HF")
    parser.add_argument("--model_name", required=True, help="Short tag used for output filenames")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto", choices=["auto","bf16","fp16","fp32"])
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows for quick test")
    parser.add_argument("--out", default=None, help="Output CSV (default: <model_name>_<input>_pred.csv)")
    parser.add_argument("--stage-logits-out", default=None, help="Placeholders (empty dicts) for compatibility")
    parser.add_argument("--entropy-stats-out", default=None, help="Placeholders (empty dicts) for compatibility")
    args = parser.parse_args()

    print(f"[info] HF pipeline eval | model={args.model} device={args.device} dtype={args.dtype}", flush=True)

    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit).copy()
    validate_csv(df)

    pipe = load_pipe(args.model, args.device, args.dtype)

    img_cache: Dict[str, List[ImgWithName]] = {}
    preds_label, preds_text, raw_out, latency, correct = [], [], [], [], []

    base = os.path.splitext(args.csv)[0]
    base_name = Path(base).name
    stage_logits_path = args.stage_logits_out or (args.model_name + "_" + base_name + "_stage_end_logits.jsonl")
    entropy_path = args.entropy_stats_out or (args.model_name + "_" + base_name + "_entropy_stats.jsonl")
    stage_f = open(stage_logits_path, "w", encoding="utf-8")
    entropy_f = open(entropy_path, "w", encoding="utf-8")

    for i, row in df.iterrows():
        folder_path = str(row["folder_path"])
        scope, kind, qt_clean = parse_query_type(row["query_type"])
        question = str(row["query"])
        choices  = split_choices(row["choices"])
        gt_label, gt_text = normalize_ground_truth(row.get("ground_truth",""), choices)

        # Load images
        try:
            imgs_named = list_images_cached(folder_path, img_cache)
        except Exception as e:
            print(f"[WARN] Row {i}: {e}", flush=True)
            imgs_named = []

        # Build messages
        user_text = build_user_text(
            question, choices, qt_clean,
            row.get("cross_frame",""), row.get("perspective_changing",""), row.get("object_num","")
        )
        messages = build_messages_for_pipeline(imgs_named, user_text, args.csv)

        # Run pipeline
        t0 = time.time()
        try:
            result = pipe(text=messages, max_new_tokens=args.max_new_tokens, do_sample=False)
        except TypeError:
            result = pipe(text=messages, generate_kwargs={"max_new_tokens": args.max_new_tokens, "do_sample": False})
        dt = time.time() - t0

        # Normalize output -> string
        result = result[0]["generated_text"][-1]["content"]
        out_text = _ensure_string_text(result)

        label, text = parse_prediction(out_text)
        is_ok = False
        if label in LABELS and gt_label in LABELS:
            is_ok = (label == gt_label)
        elif text and gt_text:
            is_ok = (text.strip().lower() == gt_text.strip().lower())

        preds_label.append(label)
        preds_text.append(text)
        raw_out.append(out_text)
        latency.append(dt)
        correct.append(bool(is_ok))

        # Placeholders for compatibility (pipeline doesn't expose logits/entropy)
        stage_f.write(json.dumps({"row_index": int(i), "stage_end_token_logits": {}}, ensure_ascii=False) + "\n")
        entropy_f.write(json.dumps({"row_index": int(i), "stage_entropy_stats": {}}, ensure_ascii=False) + "\n")

        print(f"[{i}] {folder_path} -> pred={label}.{text} | gt={gt_label}.{gt_text} | ok={is_ok} | {dt:.2f}s", flush=True)

    stage_f.close()
    entropy_f.close()
    print(f"[done] stage logits -> {stage_logits_path}", flush=True)
    print(f"[done] per-sample stage entropies (JSONL) -> {entropy_path}", flush=True)

    out_path = args.out or (args.model_name + "_" + base_name + "_pred.csv")
    df_out = df.copy()
    df_out["prediction_label"] = preds_label
    df_out["prediction_text"]  = preds_text
    df_out["prediction_raw"]   = raw_out
    df_out["latency_sec"]      = latency
    df_out["correct"]          = correct

    acc = (sum(correct) / len(correct)) if len(correct) else 0.0
    df_out.to_csv(out_path, index=False)
    print(f"[done] saved -> {out_path} | accuracy={acc:.4f}", flush=True)

if __name__ == "__main__":
    main()
