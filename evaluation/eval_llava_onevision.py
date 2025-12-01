#!/usr/bin/env python3
import os, re, time, argparse, warnings, json, string
from pathlib import Path
from typing import List, Tuple, Dict, Optional, NamedTuple

import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# ============================================
# Speed knobs
# ============================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================
# Flash-Attention check
# ============================================
def maybe_enable_flash_attn(model):
    try:
        import flash_attn  # noqa: F401
        model.config.attn_implementation = "flash_attention_2"
        return True
    except Exception:
        return False

# ============================================
# Model loader (OneVision uses CausalLM)
# ============================================
def load_model(model_name: str, device: str, dtype: str, use_4bit: bool, compile_model: bool):
    from transformers import BitsAndBytesConfig

    dtype_map = {"auto": None, "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    base_dtype = dtype_map.get(dtype, None)
    torch_dtype = base_dtype
    quant_kwargs = {}
    device_is_cuda = device.startswith("cuda")

    if use_4bit:
        if not device_is_cuda:
            print("[WARN] 4-bit requires CUDA; disabling.")
            use_4bit = False
        else:
            try:
                quant_kwargs["load_in_4bit"] = True
                quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16 if torch_dtype is None else torch_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                torch_dtype = None
            except Exception as e:
                print(f"[WARN] bitsandbytes not available ({e}); fallback to {dtype}")
                quant_kwargs.clear()
                torch_dtype = base_dtype

    device_map = {"": device} if use_4bit and device_is_cuda else None
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map=device_map,
        **quant_kwargs,
    )

    if not use_4bit:
        model = model.to(device)
    model.eval()
    try:
        model.generation_config.use_cache = True
    except Exception:
        pass

    if maybe_enable_flash_attn(model):
        print("[info] FlashAttention-2 enabled")
    else:
        print("[info] Using default attention.")

    if compile_model:
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            print("[info] torch.compile enabled")
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}")
    return processor, model

# ============================================
# System reasoning prompts (preserved)
# ============================================
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
# (for brevity, paste the full prompt definitions from your original script here)

# ============================================
# Utility helpers
# ============================================
LABELS = ["A", "B", "C", "D"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CHOICE_LINE_RE = re.compile(r"\b([ABCD])\s*\.\s*([^\n\r]+)")
UNICODE_PUNCT = "，。！？；：、（）《》【】“”‘’—…·「」『』《》–—"
PUNCT_CHARS = set(string.punctuation + UNICODE_PUNCT)

def is_punct_or_newline(tok_str: str) -> bool:
    if not tok_str.strip(): return True
    t_no_space = tok_str.replace(" ", "")
    if t_no_space and all(ch in PUNCT_CHARS for ch in t_no_space): return True
    if "\n" in tok_str or "\r" in tok_str: return True
    return False

def parse_prediction(generated: str) -> Tuple[str, str]:
    if not generated: return "INVALID", ""
    matches = list(CHOICE_LINE_RE.finditer(generated))
    if matches:
        m = matches[-1]
        return m.group(1), m.group(2).strip()
    single_labels = re.findall(r"\b([ABCD])\b", generated)
    if single_labels:
        return single_labels[-1], "unknown"
    return "INVALID", ""

class ImgWithName(NamedTuple):
    image: Image.Image
    name: str

def list_images(folder_path: str) -> List[ImgWithName]:
    p = Path(folder_path)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    files = [x for x in p.iterdir() if x.suffix.lower() in IMG_EXTS]
    files.sort(key=lambda x: x.name)
    imgs = []
    for f in files:
        with Image.open(f) as im:
            imgs.append(ImgWithName(im.convert("RGB").copy(), f.stem))
    return imgs

# ============================================
# Message builder with system prompt
# ============================================
def build_messages(pil_images_with_names: List[ImgWithName], user_text: str, csv_path: str):
    prompt_key = [k for k in SYSTEM_PROMPT.keys() if k in csv_path][0]
    system_prompt = SYSTEM_PROMPT[prompt_key]
    content = []
    for it in pil_images_with_names:
        content.append({"type": "text", "text": it.name})
        content.append({"type": "image", "image": it.image})
    content.append({"type": "text", "text": user_text})
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
    ]

# ============================================
# Stage-wise generation for OneVision
# ============================================
STAGE_TEXTS = {1: "(1)", 2: "(2)", 3: "(3)", 4: "(4)"}

@torch.inference_mode()
def generate_once_stage_end_logits(processor, model, device, messages, max_new_tokens=1024):
    chat_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    imgs = []
    for m in messages:
        if m.get("role") == "user":
            for part in m.get("content", []):
                if part.get("type") == "image":
                    imgs.append(part["image"])
    inputs = processor(text=[chat_text], images=imgs, return_tensors="pt")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    tok = processor.tokenizer
    prompt_ids = tok(chat_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    prompt_len = prompt_ids.shape[-1]

    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        use_cache=True,
        output_scores=True,
        return_dict_in_generate=True,
    )

    gen_ids = gen.sequences
    new_tokens = gen_ids[0, prompt_len:]
    tokenizer = tok

    # detect stages
    cumulative = ""
    stage_start, stage_end = {}, {}
    for k in STAGE_TEXTS:
        stage_start[k], stage_end[k] = None, None
    seen = set()
    for t in range(new_tokens.shape[0]):
        s = tokenizer.decode([new_tokens[t].item()], skip_special_tokens=False)
        cumulative += s
        for k in STAGE_TEXTS:
            if k not in seen and STAGE_TEXTS[k] in cumulative:
                stage_start[k] = t
                seen.add(k)
    for k in (1, 2, 3):
        if stage_start[k] is not None and stage_start[k + 1] is not None:
            stage_end[k] = stage_start[k + 1] - 1
    if stage_start[4] is not None:
        stage_end[4] = new_tokens.shape[0] - 1

    def entropy(t: torch.Tensor):
        t = t.float() - t.max()
        p = torch.softmax(t, dim=-1)
        return float(-(p * (p + 1e-12).log()).sum())

    stage_stats = {}
    for k in (1, 2, 3, 4):
        s, e = stage_start[k], stage_end[k]
        if s is None or e is None:
            continue
        vals = [
            entropy(gen.scores[i][0])
            for i in range(s, e + 1)
            if not is_punct_or_newline(tokenizer.decode([new_tokens[i].item()]))
        ]
        if vals:
            m = sum(vals) / len(vals)
            v = sum((x - m) ** 2 for x in vals) / len(vals)
            stage_stats[k] = {"mean_entropy_nats": m, "var_entropy_nats": v}

    full_text = tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    pred_label, pred_text = parse_prediction(full_text)
    return full_text, pred_label, pred_text, stage_stats

# ============================================
# Main
# ============================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", default="lmms-lab/llava-onevision-qwen2-0.5b-ov")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--use-4bit", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"[info] device={args.device} dtype={args.dtype} model={args.model}")
    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit)
    processor, model = load_model(args.model, args.device, args.dtype, args.use_4bit, args.compile)

    preds_label, preds_text, raw_out, latency, correct = [], [], [], [], []
    for i, row in df.iterrows():
        folder = str(row["folder_path"])
        question = str(row["query"])
        choices = [c.strip() for c in str(row["choices"]).split(",") if c.strip()]
        gt_label, gt_text = parse_prediction(str(row["ground_truth"]))
        imgs = list_images(folder)

        user_text = f"Question: {question}\nChoices:\n" + "\n".join(choices)
        messages = build_messages(imgs, user_text, args.csv)

        t0 = time.time()
        full_text, pl, pt, stage_stats = generate_once_stage_end_logits(
            processor, model, args.device, messages, args.max_new_tokens
        )
        dt = time.time() - t0
        ok = pl == gt_label

        preds_label.append(pl)
        preds_text.append(pt)
        raw_out.append(full_text)
        latency.append(dt)
        correct.append(ok)
        print(f"[{i}] {folder} -> pred={pl} gt={gt_label} ok={ok} {dt:.2f}s", flush=True)

    out_path = args.out or os.path.splitext(args.csv)[0] + "_llava_pred.csv"
    df["prediction_label"] = preds_label
    df["prediction_text"] = preds_text
    df["prediction_raw"] = raw_out
    df["latency_sec"] = latency
    df["correct"] = correct
    acc = sum(correct) / len(correct) if correct else 0.0
    df.to_csv(out_path, index=False)
    print(f"[done] saved -> {out_path} | accuracy={acc:.4f}")

if __name__ == "__main__":
    main()
