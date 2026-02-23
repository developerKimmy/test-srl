"""SRL 모델 추론 + 결과 저장."""
import random

from config import MODELS, SENTENCES, SENTENCES_EN, OUTPUT_DIR
from config import detect_predicates_ko, detect_predicates_en
from shared import save_json
from inference import MODELS as INFER_REGISTRY


def _tokenize(sentence):
    return sentence.split()


def _print_results(sentence, pred_token, results):
    print(f"원문: {sentence}")
    print(f"서술어: {pred_token}\n")
    if not results:
        print("  감지된 역할 없음")
    for r in results:
        print(f"  [{r['label']}] {r['text']} (score: {r['score']:.4f})")
    print()


def _run_model(model_name, sentences, pred_detector, suffix=""):
    config = MODELS[model_name]
    registry = INFER_REGISTRY[model_name]

    print(f"=== {model_name} ({config['id']}) | {len(sentences)}문장 ===\n")

    ctx = registry["loader"](config["id"])
    infer = registry["infer"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{model_name}{suffix}.json"

    records = []
    for sentence in sentences:
        tokens = _tokenize(sentence)
        pred_indices = pred_detector(tokens)

        if not pred_indices:
            records.append({"sentence": sentence, "predicates": []})
            continue

        pred_results = []
        for pred_idx in pred_indices:
            try:
                results = infer(ctx, tokens, pred_idx)
            except Exception as e:
                print(f"  오류 ({tokens[pred_idx]}): {e}")
                results = []

            _print_results(sentence, tokens[pred_idx], results)

            pred_results.append({
                "predicate": tokens[pred_idx],
                "predicate_index": pred_idx,
                "arguments": results,
            })

        records.append({
            "sentence": sentence,
            "predicates": pred_results,
        })

    save_json(output_path, records)
    print(f"저장: {output_path}\n")


def run(models, limit=0, lang="ko"):
    """모델 리스트에 대해 SRL 추론 실행."""
    if lang == "en":
        pool = SENTENCES_EN
        pred_detector = detect_predicates_en
        suffix = "_en"
    else:
        pool = SENTENCES
        pred_detector = detect_predicates_ko
        suffix = ""

    sentences = random.sample(pool, min(limit, len(pool))) if limit > 0 else pool
    for model_name in models:
        _run_model(model_name, sentences, pred_detector, suffix)
