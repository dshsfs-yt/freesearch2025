from pathlib import Path
from typing import Dict, Any, List, Tuple

import json
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,          # ✅ Trainer → Seq2SeqTrainer
    Seq2SeqTrainingArguments,  # ✅ TrainingArguments → Seq2SeqTrainingArguments
    set_seed,
)
import torch

# =============================================================
# 0) 고정 설정 (환경변수 사용 안 함)
#    - 입력: stt_text (교정 전)
#    - 출력: correct_text (교정 후)
# =============================================================

CSV_PATH = "ouput.csv"  # ← 실제 파일명이 output.csv라면 수정하세요.
SAVE_DIR = Path("ckpt/ke-t5-sent-correction")
MODEL_NAME = "KETI-AIR/ke-t5-small-ko"

RANDOM_SEED = 42
MAX_SAMPLES = 0            # 0이면 전체 사용
MAX_SRC_LEN = 256
MAX_TGT_LEN = 128
BATCH_TRAIN = 32  # 배치 사이즈로 메모리 조절
BATCH_EVAL = 32
EPOCHS = 3
LR = 3e-4
SAVE_STEPS = 2000
EVAL_STEPS = 2000
LOG_STEPS = 50
PREFIX = "fix: "           # 입력 문장 앞에 붙여 교정 태스크를 명시

# 컬럼 고정
SRC_COL = "stt_text"
TGT_COL = "correct_text"

set_seed(RANDOM_SEED)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================
# 1) 디바이스/정밀도 설정
# =============================================================

def get_device_and_precision() -> Tuple[torch.device, Dict[str, bool], bool, bool]:
    use_cuda = torch.cuda.is_available()
    use_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if use_cuda:
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        bf16_ok = torch.cuda.is_bf16_supported()
        precision = {"bf16": bf16_ok, "fp16": (not bf16_ok)}
    elif use_mps:
        device = torch.device("mps")
        precision = {"bf16": False, "fp16": False}
    else:
        device = torch.device("cpu")
        precision = {"bf16": False, "fp16": False}

    return device, precision, use_cuda, use_mps

device, precision_kwargs, use_cuda, use_mps = get_device_and_precision()
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

print(f"[Device] {device} | CUDA: {use_cuda} | MPS: {use_mps}")
if use_cuda:
    try:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        bf16_sup = torch.cuda.is_bf16_supported()
        print(f"[CUDA] device_name={name} | capability={cap} | bf16_supported={bf16_sup}")
    except Exception as e:
        print(f"[CUDA] info fetch error: {e}")

# =============================================================
# 2) 데이터 로드/정제
# =============================================================

print(f"[Load] {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
k=int(len(df)*0.8)
df = df[:k]

missing = [c for c in [SRC_COL, TGT_COL] if c not in df.columns]
if missing:
    raise ValueError(
        f"CSV에 필요한 컬럼이 없습니다: {missing}. "
        f"CSV는 반드시 ['{SRC_COL}','{TGT_COL}'] 두 컬럼을 포함해야 합니다."
    )

if MAX_SAMPLES > 0:
    df = df.sample(n=min(MAX_SAMPLES, len(df)), random_state=RANDOM_SEED).reset_index(drop=True)

# 형변환/결측/공백/중복 처리
for c in [SRC_COL, TGT_COL]:
    df[c] = df[c].astype(str)

df = df.dropna(subset=[SRC_COL, TGT_COL])
df = df[(df[SRC_COL].str.strip() != "") & (df[TGT_COL].str.strip() != "")].reset_index(drop=True)
df = df.drop_duplicates(subset=[SRC_COL, TGT_COL]).reset_index(drop=True)

# 학습/검증 분할 (기본 95/5, 데이터가 매우 작으면 마지막 한 샘플을 검증)
n_total = len(df)
val_size = max(1, int(n_total * 0.05)) if n_total > 20 else max(1, n_total - 1)
train_df = df.iloc[: n_total - val_size].reset_index(drop=True)
val_df = df.iloc[n_total - val_size :].reset_index(drop=True)

raw_ds = DatasetDict(
    {
        "train": Dataset.from_pandas(train_df[[SRC_COL, TGT_COL]], preserve_index=False),
        "validation": Dataset.from_pandas(val_df[[SRC_COL, TGT_COL]], preserve_index=False),
    }
)
print(raw_ds)

# =============================================================
# 3) 토크나이저/모델 로드
# =============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
try:
    model.to(device)
except Exception:
    pass

# =============================================================
# 4) 토크나이즈 함수 (입력에 PREFIX 부착)
# =============================================================

def make_tokenize_fn(max_src_len=256, max_tgt_len=128):
    def tokenize_function(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        src_texts = [PREFIX + s for s in batch[SRC_COL]]
        tgt_texts = [t for t in batch[TGT_COL]]
        enc = tokenizer(src_texts, max_length=max_src_len, truncation=True)
        lab = tokenizer(text_target=tgt_texts, max_length=max_tgt_len, truncation=True)
        enc["labels"] = lab["input_ids"]
        return enc
    return tokenize_function

tokenized = raw_ds.map(
    make_tokenize_fn(MAX_SRC_LEN, MAX_TGT_LEN),
    batched=True,
    remove_columns=raw_ds["train"].column_names,
)

# =============================================================
# 5) Collator, Metrics, Trainer 설정
# =============================================================

collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding="longest",
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)

def _normalize(s: str) -> str:
    return " ".join(s.strip().split())

def _levenshtein(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,          # deletion
                dp[i][j - 1] + 1,          # insertion
                dp[i - 1][j - 1] + cost,   # substitution
            )
    return dp[n][m]

def compute_metrics(eval_pred):
    # ✅ Seq2SeqTrainer + predict_with_generate=True 이면 predictions는 생성된 토큰 ID
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # ✅ 정수형으로 강제 캐스팅 후 음수 마스킹값(-100) 치환
    labels = np.asarray(labels, dtype=np.int64)
    labels[labels == -100] = pad_id

    # ✅ 혹시 모를 음수 예측값도 방지
    preds = np.asarray(preds, dtype=np.int64)
    preds[preds < 0] = pad_id
    
    decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds  = [_normalize(p) for p in decoded_preds]
    decoded_labels = [_normalize(l) for l in decoded_labels]

    total = len(decoded_preds)
    exact = sum(1 for p, l in zip(decoded_preds, decoded_labels) if p == l)
    exact_match = exact / total if total > 0 else 0.0

    # CER
    cer_sum = 0.0
    for p, l in zip(decoded_preds, decoded_labels):
        dist = _levenshtein(list(p), list(l))
        cer = dist / max(1, len(l))
        cer_sum += cer
    cer_avg = cer_sum / total if total > 0 else 0.0

    # WER
    wer_sum = 0.0
    for p, l in zip(decoded_preds, decoded_labels):
        p_tok, l_tok = p.split(), l.split()
        dist = _levenshtein(p_tok, l_tok)
        wer = dist / max(1, len(l_tok))
        wer_sum += wer
    wer_avg = wer_sum / total if total > 0 else 0.0

    return {"exact_match": exact_match, "cer": cer_avg, "wer": wer_avg}

optim_choice = "adamw_torch_fused" if use_cuda else "adamw_torch"

args = Seq2SeqTrainingArguments(
    output_dir=str(SAVE_DIR),
    per_device_train_batch_size=BATCH_TRAIN,
    per_device_eval_batch_size=BATCH_EVAL,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    eval_strategy="steps",     
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    logging_strategy="steps",
    logging_steps=LOG_STEPS,
    report_to="none",
    dataloader_pin_memory=use_cuda,
    optim=optim_choice,
    predict_with_generate=True,       # ✅ 생성 기반 평가 활성화
    generation_max_length=MAX_TGT_LEN,
    generation_num_beams=1,           # ✅ 메모리 절약
    eval_accumulation_steps=4,        # ✅ CPU 메모리 완화
    **precision_kwargs,
)

trainer = Seq2SeqTrainer(             # ✅ Seq2SeqTrainer 사용
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# =============================================================
# 6) 학습
# =============================================================

trainer.train()

# =============================================================
# 7) 저장: 모델/토크나이저와 설정
# =============================================================

trainer.save_model(str(SAVE_DIR))
tokenizer.save_pretrained(str(SAVE_DIR))

cfg = {
    "model_name": MODEL_NAME,
    "src_col": SRC_COL,
    "tgt_col": TGT_COL,
    "prefix": PREFIX,
    "max_src_len": MAX_SRC_LEN,
    "max_tgt_len": MAX_TGT_LEN,
}
with open(SAVE_DIR / "training_config.json", "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)

print(f"[OK] Saved to: {SAVE_DIR.resolve()}")
