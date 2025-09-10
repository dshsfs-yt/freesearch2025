# infer_interactive_ke_t5.py
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


DEF_MODEL_DIR = Path("ckpt/ke-t5-sent-correction")  # 학습 저장 경로 기본값


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_cfg(model_dir: Path):
    """훈련 시 저장한 training_config.json(있으면)에서 prefix/길이 로드."""
    cfg = {"prefix": "fix: ", "max_src_len": 256, "max_tgt_len": 128}
    cfg_path = model_dir / "training_config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            cfg["prefix"] = saved.get("prefix", cfg["prefix"])
            cfg["max_src_len"] = int(saved.get("max_src_len", cfg["max_src_len"]))
            cfg["max_tgt_len"] = int(saved.get("max_tgt_len", cfg["max_tgt_len"]))
        except Exception:
            pass
    return cfg


def load_model(model_dir: Path, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    return tokenizer, model


@torch.inference_mode()
def generate_one(
    text: str,
    tokenizer,
    model,
    device: torch.device,
    prefix: str,
    max_src_len: int,
    max_tgt_len: int,
    num_beams: int = 1,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> str:
    inp = prefix + text.strip()
    enc = tokenizer(
        inp,
        max_length=max_src_len,
        truncation=True,
        padding=False,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    gen = model.generate(
        **enc,
        max_new_tokens=max_tgt_len,
        num_beams=num_beams,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        length_penalty=1.0,
        no_repeat_ngram_size=0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    out = tokenizer.decode(gen[0], skip_special_tokens=True)
    return out.strip()


def main():
    ap = argparse.ArgumentParser(description="KE-T5 sentence correction (interactive)")
    ap.add_argument("--model_dir", type=Path, default=DEF_MODEL_DIR, help="학습 모델 디렉토리")
    ap.add_argument("--num_beams", type=int, default=1, help="빔 서치 개수(메모리 아끼려면 1 권장)")
    ap.add_argument("--do_sample", action="store_true", help="샘플링 사용(탐색 강화)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_src_len", type=int, default=0, help="0이면 config 사용")
    ap.add_argument("--max_tgt_len", type=int, default=0, help="0이면 config 사용")
    args = ap.parse_args()

    device = get_device()
    cfg = load_cfg(args.model_dir)
    tokenizer, model = load_model(args.model_dir, device)

    max_src_len = args.max_src_len or cfg["max_src_len"]
    max_tgt_len = args.max_tgt_len or cfg["max_tgt_len"]

    print(f"[device] {device} | model_dir={args.model_dir}")
    print("입력 문장을 치면 교정 결과를 출력합니다. 종료: /q 또는 /quit")
    while True:
        try:
            src = input("> ").strip()
        except EOFError:
            break
        if not src:
            continue
        if src in ("/q", "/quit"):
            break
        pred = generate_one(
            src,
            tokenizer,
            model,
            device,
            prefix=cfg["prefix"],
            max_src_len=max_src_len,
            max_tgt_len=max_tgt_len,
            num_beams=args.num_beams,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(pred, flush=True)


if __name__ == "__main__":
    main()
