from __future__ import annotations
from training import ID2LABEL
import math

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from text_preprocessing.tokenization import BertTokenizer, hf_token
from model import IntentClassifier, BANKING77_CONFIG
from text_preprocessing.embeddings import BertEmbedder 
from training import TRAIN_CONFIG

# ── Config ────────────────────────────────────────────────────────────────────
BERT_MODEL_NAME = 'bert-base-uncased'
MODEL_PATH      = r"D:/@FYP-IntentClassifier/best_model.pt"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH      = 64
THRESHOLD       = 0.5
TOP_K           = 3

# ── Load artefacts at startup ─────────────────────────────────────────────────
print(f"[startup] device={DEVICE}  bert={BERT_MODEL_NAME}")

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME, token=hf_token)

embedder = BertEmbedder(device=DEVICE)  

model = IntentClassifier(
    cfg=BANKING77_CONFIG,
    num_classes=TRAIN_CONFIG["num_classes"],
    pool=TRAIN_CONFIG["pool"],
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

NUM_LABELS = len(ID2LABEL)  # defined once at module level
print(f"[startup] model loaded — {NUM_LABELS} labels")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="K&P Intent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str


class IntentResult(BaseModel):
    intent: str
    label: str
    confidence: float
    escalate: bool
    top_k: list[dict]

def is_low_quality_input(text: str, probs: torch.Tensor) -> bool:
    # Too short
    if len(text.split()) < 2:
        return True
    
    # Model is "spread out" / confused — high entropy means uncertain
    p = probs[0].cpu().numpy()
    entropy = -sum(pi * math.log(pi + 1e-9) for pi in p)
    max_entropy = math.log(len(p))
    normalized_entropy = entropy / max_entropy  # 0=certain, 1=random
    if normalized_entropy > 0.6:
        return True
    
    return False

@app.post("/predict", response_model=IntentResult)
def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be empty")

    tokens = tokenizer(
        text, max_length=MAX_LENGTH,
        padding="max_length", truncation=True, return_tensors="pt",
    )

    emb  = embedder.get_embeddings(tokens["input_ids"], tokens["attention_mask"])
    mask = tokens["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(emb, mask)
        probs  = F.softmax(logits, dim=1)

    # ── Reject low-quality input before even checking threshold ──────────────
    if is_low_quality_input(text, probs):
        return IntentResult(
            intent="support",
            label="Support",
            confidence=0.0,
            escalate=True,
            top_k=[],
        )

    top_probs, top_ids = probs.topk(min(TOP_K, NUM_LABELS), dim=1)

    top_k_results = [
        {
            "intent":     ID2LABEL[idx.item()],
            "confidence": round(prob.item(), 4),
        }
        for prob, idx in zip(top_probs[0], top_ids[0])
    ]

    best_intent = ID2LABEL[top_ids[0][0].item()]
    best_conf   = top_probs[0][0].item()
    escalate    = best_conf < THRESHOLD

    if escalate:
        best_intent = "support"

    return IntentResult(
        intent=best_intent,
        label=best_intent.replace("_", " ").title(),
        confidence=round(best_conf, 4),
        escalate=escalate,
        top_k=top_k_results,
    )


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}

@app.get("/")
def welcome():
    return {'Message':'Welcome Aboard!'}