#!/usr/bin/env python3
import os, re, time, argparse, warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional, NamedTuple
import json
import string

import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# ============================
# Speed knobs (safe defaults)
# ============================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", category=UserWarning)

# ============================
# Model utilities
# ============================
def maybe_enable_flash_attn(model):
    try:
        import flash_attn  # noqa: F401
        model.config.attn_implementation = "flash_attention_2"
        return True
    except Exception:
        return False

def load_model(model_name: str, device: str, dtype: str, use_4bit: bool, compile_model: bool):
    dtype_map = {"auto": None, "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    base_dtype = dtype_map.get(dtype, None)
    torch_dtype = base_dtype

    quant_kwargs = {}
    device_is_cuda = device.startswith("cuda")
    if use_4bit:
        if not device_is_cuda:
            print("[WARN] 4-bit loading requires a CUDA device; disabling --use-4bit.")
            use_4bit = False
        else:
            try:
                from transformers import BitsAndBytesConfig
                quant_kwargs["load_in_4bit"] = True
                quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16 if torch_dtype is None else torch_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                torch_dtype = None
            except Exception as e:
                print(f"[WARN] bitsandbytes not available ({e}); falling back to {dtype}.")
                use_4bit = False
                torch_dtype = base_dtype
                quant_kwargs.clear()

    if use_4bit and device_is_cuda:
        device_map = {"": device}
    else:
        device_map = None if device_is_cuda else {"": device}

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name, torch_dtype=torch_dtype, trust_remote_code=True, device_map=device_map, **quant_kwargs
    )
    if not use_4bit:
        model = model.to(device)
    model.eval()
    model.generation_config.use_cache = True

    if maybe_enable_flash_attn(model):
        print("[info] FlashAttention-2 enabled")
    else:
        print("[info] FlashAttention-2 not available; using default attn.")

    if compile_model:
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            print("[info] torch.compile enabled (reduce-overhead)")
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}")

    return processor, model

# ============================
# Prompt (System)
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

# ============================
# CSV helpers & parsing
# ============================
LABELS = ["A", "B", "C", "D"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CHOICE_LINE_RE = re.compile(r"\b([ABCD])\s*\.\s*([^\n\r]+)")

def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]

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
# Images (bind names directly)
# ============================
class ImgWithName(NamedTuple):
    image: Image.Image
    name: str

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
# Prompt builders
# ============================
def build_user_text(question: str, choices: List[str], query_type: str, cross_frame, perspective_changing, object_num) -> str:
    choice_lines = "\n".join(choices)
    tail = "Follow the Reasoning Path Template strictly. Your last line must match ^[ABCD]\\.[^\\n\\r]+$ and contain only one choice."
    return f"Question: {question}\nChoices:\n{choice_lines}\n\n{tail}"

def build_messages(pil_images_with_names: List[ImgWithName], user_text: str, csv_path: str):
    """Bind each image by inserting its raw filename immediately before the image."""
    prompt_key = [k for k in SYSTEM_PROMPT.keys() if k in csv_path][0]
    print(prompt_key)
    system_prompt = SYSTEM_PROMPT[prompt_key]

    content = []
    for it in pil_images_with_names:
        # Direct binding: filename text -> image
        content.append({"type": "text", "text": it.name})
        content.append({"type": "image", "image": it.image})
    content.append({"type": "text", "text": user_text})

    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
    ]

def parse_prediction(generated: str) -> Tuple[str, str]:
    if not generated: return "INVALID", ""
    matches = list(CHOICE_LINE_RE.finditer(generated))
    if matches:
        m = matches[-1]; return m.group(1), m.group(2).strip()
    single_labels = re.findall(r"\b([ABCD])\b", generated)
    if single_labels: return single_labels[-1], "unknown"
    return "INVALID", ""

# ============================
# Token filtering
# ============================
UNICODE_PUNCT = "，。！？；：、（）《》【】“”‘’—…·「」『』《》–—"
PUNCT_CHARS = set(string.punctuation + UNICODE_PUNCT)

def is_punct_or_newline(tok_str: str) -> bool:
    if tok_str == "": return True
    if tok_str.strip() == "": return True
    t = tok_str.strip()
    t_no_space = t.replace(" ", "")
    if t_no_space and all(ch in PUNCT_CHARS for ch in t_no_space): return True
    if "\n" in tok_str or "\r" in tok_str: return True
    return False

# ============================
# Entropy helper (in nats) — per vector
# ============================
def logits_entropy_nats(logits: List[float]) -> float:
    with torch.no_grad():
        t = torch.tensor(logits, dtype=torch.float32)
        t = t - t.max()
        p = torch.softmax(t, dim=-1)
        eps = 1e-12
        return float(-(p * (p.add(eps).log())).sum().item())

# ============================
# Marker detection helpers
# ============================
def decode_token(tokenizer, tid: int) -> str:
    return tokenizer.decode([tid])

def find_marker_start(tokenizer, new_tokens: torch.Tensor, end_step: int, marker_text: str, max_span_tokens: int = 6) -> Optional[int]:
    for start in range(max(0, end_step - max_span_tokens + 1), end_step + 1):
        cand = tokenizer.decode(new_tokens[start:end_step+1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if cand.endswith(marker_text):
            return start
    return None

# ============================
# Generation with stage regions (first appearance)
# ============================
STAGE_TEXTS = {1: "(1)", 2: "(2)", 3: "(3)", 4: "(4)"}

@torch.inference_mode()
def generate_once_stage_end_logits(
    processor, model, device, messages, max_new_tokens=80
) -> Tuple[
    str,
    Dict[str, List[float]],
    Dict[str, Dict[str, Optional[str]]],
    Dict[str, Dict[str, Optional[float]]],
]:
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True,
    )
    prompt_len = inputs["input_ids"].shape[-1]

    try:
        inputs = inputs.to(device)
    except AttributeError:
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device, non_blocking=True)

    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        use_cache=True,
        output_scores=True,
        return_dict_in_generate=True
    )

    gen_ids = gen.sequences
    new_tokens = gen_ids[0, prompt_len:]
    tokenizer = processor.tokenizer

    stage_start_step: Dict[int, Optional[int]] = {1: None, 2: None, 3: None, 4: None}
    stage_end_step:   Dict[int, Optional[int]] = {1: None, 2: None, 3: None, 4: None}

    cumulative_text = ""
    seen = {1: False, 2: False, 3: False, 4: False}

    for t in range(new_tokens.shape[0]):
        tok_id = new_tokens[t].item()
        tok_str = decode_token(tokenizer, tok_id)
        cumulative_text += tok_str

        for k in (1,2,3,4):
            if not seen[k] and STAGE_TEXTS[k] in cumulative_text:
                start_idx = find_marker_start(tokenizer, new_tokens, t, STAGE_TEXTS[k])
                if start_idx is None:
                    start_idx = t
                stage_start_step[k] = start_idx
                seen[k] = True

    for k in (1,2,3):
        if stage_start_step[k] is not None and stage_start_step[k+1] is not None:
            stage_end_step[k] = max(stage_start_step[k+1]-1, stage_start_step[k])
    if stage_start_step[4] is not None:
        stage_end_step[4] = new_tokens.shape[0]-1

    def last_valid_within(start_i: Optional[int], end_i: Optional[int]) -> Optional[int]:
        if start_i is None or end_i is None or start_i > end_i:
            return None
        t = end_i
        while t >= start_i:
            s = decode_token(tokenizer, new_tokens[t].item())
            if not is_punct_or_newline(s):
                return t
            t -= 1
        return None

    key_to_logits: Dict[str, List[float]] = {}
    stage_debug: Dict[str, Dict[str, Optional[str]]] = {}

    def tail_text(start_i: Optional[int], end_step: Optional[int], width: int = 140) -> str:
        if start_i is None or end_step is None:
            return ""
        txt = tokenizer.decode(
            new_tokens[start_i:end_step+1],
            skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return txt[-width:]

    def stage_token_indices(start_i: Optional[int], end_i: Optional[int]) -> List[int]:
        if start_i is None or end_i is None or start_i > end_i:
            return []
        idxs = []
        for t in range(start_i, end_i + 1):
            s = decode_token(tokenizer, new_tokens[t].item())
            if not is_punct_or_newline(s):
                idxs.append(t)
        return idxs

    def entropy_from_score_tensor(score_tensor: torch.Tensor) -> float:
        t = score_tensor.detach().float()
        t = t - t.max()
        p = torch.softmax(t, dim=-1)
        eps = 1e-12
        return float(-(p * (p.add(eps).log())).sum().item())

    def mean_var(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
        n = len(vals)
        if n == 0:
            return None, None
        m = sum(vals) / n
        var = sum((x - m) ** 2 for x in vals) / n
        return m, var

    stage_stats: Dict[str, Dict[str, Optional[float]]] = {}

    for k in (1,2,3,4):
        start_i = stage_start_step[k]
        end_i = stage_end_step[k]
        end_step_tok = last_valid_within(start_i, end_i)
        stage_debug[str(k)] = {
            "start_step": None if start_i is None else str(start_i),
            "end_step":   None if end_i is None else str(end_i),
            "end_token_step": None if end_step_tok is None else str(end_step_tok),
            "end_token_text": None if end_step_tok is None else decode_token(tokenizer, new_tokens[end_step_tok].item()).replace("\n","\\n"),
            "region_tail": tail_text(start_i, end_step_tok),
        }
        if end_step_tok is not None:
            logits_vec = gen.scores[end_step_tok][0].detach().float().cpu().tolist()
            key_to_logits[str(k)] = logits_vec

        idxs = stage_token_indices(start_i, end_i)
        ents = [entropy_from_score_tensor(gen.scores[t][0]) for t in idxs]
        m, v = mean_var(ents)
        stage_stats[str(k)] = {
            "mean_entropy_nats": None if m is None else float(m),
            "var_entropy_nats":  None if v is None else float(v),
        }

    full_text = tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    pred_label, _ = parse_prediction(full_text)
    if pred_label in LABELS:
        choice_step = None
        for t in range(new_tokens.shape[0]-1, -1, -1):
            s = tokenizer.decode([new_tokens[t].item()]).strip()
            if s.startswith(pred_label):
                choice_step = t
                break
        if choice_step is not None:
            logits_vec = gen.scores[choice_step][0].detach().float().cpu().tolist()
            key_to_logits[pred_label] = logits_vec
            stage_debug[pred_label] = {
                "start_step": None,
                "end_step": None,
                "end_token_step": str(choice_step),
                "end_token_text": tokenizer.decode([new_tokens[choice_step].item()]).replace("\n","\\n"),
                "region_tail": full_text[-140:],
            }

    return full_text, key_to_logits, stage_debug, stage_stats

# ============================
# Main
# ============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV with columns: folder_path,query_type,query,ground_truth,choices,cross_frame,perspective_changing,object_num")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto", choices=["auto","bf16","fp16","fp32"])
    parser.add_argument("--use-4bit", action="store_true", help="Load model in 4-bit (bitsandbytes)")
    parser.add_argument("--compile", action="store_true", help="torch.compile the model forward for speed")
    parser.add_argument("--max-new-tokens", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows for quick test")
    parser.add_argument("--out", default=None, help="Output CSV (default: <input>_pred.csv)")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--stage-logits-out", default=None,
                        help="Sidecar JSONL for stage(1..4)+decision logits (default: <input>_stage_end_logits.jsonl)")
    parser.add_argument("--entropy-stats-out", default=None,
                        help="Per-sample JSONL of stage entropies in nats (default: <input>_entropy_stats.jsonl)")
    parser.add_argument("--print-stage-debug", dest="print_stage_debug", action="store_true", default=False,
                        help="Print stage boundary debug info (start/end steps, end token, tail)")
    parser.add_argument("--no-print-stage-debug", dest="print_stage_debug", action="store_false")
    args = parser.parse_args()

    print(f"[info] device={args.device} dtype={args.dtype} 4bit={args.use_4bit} compile={args.compile}", flush=True)

    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit).copy()
    validate_csv(df)

    processor, model = load_model(args.model, args.device, args.dtype, args.use_4bit, args.compile)

    img_cache: Dict[str, List[ImgWithName]] = {}
    preds_label, preds_text, raw_out, latency, correct = [], [], [], [], []

    base = os.path.splitext(args.csv)[0]
    stage_logits_path = args.stage_logits_out or (args.model_name + "_" + base + "_stage_end_logits.jsonl")
    entropy_path = args.entropy_stats_out or (args.model_name + "_" + base + "_entropy_stats.jsonl")
    stage_f = open(stage_logits_path, "w", encoding="utf-8")
    entropy_f = open(entropy_path, "w", encoding="utf-8")

    for i, row in df.iterrows():
        folder_path = str(row["folder_path"])
        scope, kind, qt_clean = parse_query_type(row["query_type"])
        question = str(row["query"])
        choices  = split_choices(row["choices"])
        gt_label, gt_text = normalize_ground_truth(row.get("ground_truth",""), choices)

        try:
            imgs_named = list_images_cached(folder_path, img_cache)
        except Exception as e:
            print(f"[WARN] Row {i}: {e}", flush=True)
            imgs_named = []

        user_text = build_user_text(
            question, choices, qt_clean,
            row.get("cross_frame",""), row.get("perspective_changing",""), row.get("object_num","")
        )
        messages = build_messages(imgs_named, user_text, args.csv)

        t0 = time.time()
        out_text, stage_end_logits, stage_debug, stage_stats = generate_once_stage_end_logits(
            processor, model, args.device, messages, max_new_tokens=args.max_new_tokens
        )
        dt = time.time() - t0

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

        if args.print_stage_debug:
            print(f"\n===== [Row {i}] Reasoning stage separation =====", flush=True)
            for k in ["1","2","3","4"]:
                dbg = stage_debug.get(k, {})
                print(
                    f"(Stage {k}) start={dbg.get('start_step')}  "
                    f"end={dbg.get('end_step')}  "
                    f"end_token_step={dbg.get('end_token_step')}  "
                    f"end_token={repr(dbg.get('end_token_text'))}\n"
                    f"  tail≈{dbg.get('region_tail')}",
                    flush=True
                )
            if label in LABELS and stage_debug.get(label):
                dbg = stage_debug[label]
                print(
                    f"(Decision {label}) token_step={dbg.get('end_token_step')} "
                    f"token={repr(dbg.get('end_token_text'))}\n"
                    f"  tail≈{dbg.get('region_tail')}",
                    flush=True
                )
            print("==============================================\n", flush=True)

        ordered_logits = {k: stage_end_logits[k] for k in sorted(stage_end_logits.keys())}
        stage_f.write(json.dumps({
            "row_index": int(i),
            "stage_end_token_logits": ordered_logits
        }, ensure_ascii=False) + "\n")

        entropy_f.write(json.dumps({
            "row_index": int(i),
            "stage_entropy_stats": {
                "1": stage_stats.get("1", {}),
                "2": stage_stats.get("2", {}),
                "3": stage_stats.get("3", {}),
                "4": stage_stats.get("4", {}),
            }
        }, ensure_ascii=False) + "\n")

        print(f"[{i}] {folder_path} -> pred={label}.{text} | gt={gt_label}.{gt_text} | ok={is_ok} | {dt:.2f}s", flush=True)

    stage_f.close()
    entropy_f.close()

    print(f"[done] stage logits -> {stage_logits_path}", flush=True)
    print(f"[done] per-sample stage entropies (JSONL) -> {entropy_path}", flush=True)

    out_path = args.out or (args.model_name + "_" + base + "_pred.csv")
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
