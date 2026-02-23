"""SRL 모델 로딩 + 추론."""
import torch


def _load_pipeline(model_id):
    """token-classification pipeline 로딩."""
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    return pipeline("token-classification", model=model, tokenizer=tokenizer,
                    aggregation_strategy="simple")


def _load_manual(model_id):
    """수동 추론용 모델 로딩."""
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModelForTokenClassification.from_pretrained(model_id)
    except Exception:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_id)
    return {"tokenizer": tokenizer, "model": model}


# ── [V] 마커 기반 추론 ──

def _infer_marker(pipe, tokens, pred_idx, outside_label):
    """서술어 앞에 [V] 마커를 삽입하고 추론."""
    marked = tokens[:pred_idx] + ["[V]"] + tokens[pred_idx:]
    text = " ".join(marked)
    results = pipe(text)
    return [
        {"label": r["entity_group"], "text": r["word"], "score": round(float(r["score"]), 4)}
        for r in results if r["entity_group"] != outside_label
    ]


def infer_dannashao(pipe, tokens, pred_idx):
    return _infer_marker(pipe, tokens, pred_idx, "_")


def infer_electra(pipe, tokens, pred_idx):
    return _infer_marker(pipe, tokens, pred_idx, "O")


def infer_mbert(ctx, tokens, pred_idx):
    """mbert 기반 추론 (token_type_ids로 서술어 마킹)."""
    tokenizer, model = ctx["tokenizer"], ctx["model"]
    predicate = tokens[pred_idx]
    text = " ".join(tokens)
    inputs = tokenizer(text, predicate, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    if not hasattr(outputs, "logits"):
        return []

    predictions = torch.argmax(outputs.logits, dim=2)[0]
    id2label = model.config.id2label
    input_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    results = []
    for i, (tok, pred) in enumerate(zip(input_tokens, predictions)):
        label = id2label.get(pred.item(), "O")
        if label != "O" and tok not in ("[CLS]", "[SEP]", "[PAD]"):
            score = torch.softmax(outputs.logits[0][i], dim=0)[pred.item()].item()
            results.append({"label": label, "text": tok.replace("##", ""), "score": round(score, 4)})

    return results


# ── 모델 레지스트리 ──

MODELS = {
    "dannashao": {
        "loader": _load_pipeline,
        "infer": infer_dannashao,
    },
    "electra-srl": {
        "loader": _load_pipeline,
        "infer": infer_electra,
    },
    "mbert-srl": {
        "loader": _load_manual,
        "infer": infer_mbert,
    },
}
