"""AllenNLP End-to-End SRL 추론 스크립트.

별도 conda 환경(srl-e2e)에서 실행:
    conda activate srl-e2e
    python run_allennlp.py --lang ko
    python run_allennlp.py --lang en
"""
import argparse
import json
from pathlib import Path

from allennlp.predictors import Predictor

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "data" / "input"
OUTPUT_DIR = BASE_DIR / "data" / "output"

MODEL_URL = (
    "https://storage.googleapis.com/allennlp-public-models/"
    "structured-prediction-srl-bert.2020.12.15.tar.gz"
)


def load_sentences(lang):
    """입력 문장 로드."""
    filename = "sentences_en.json" if lang == "en" else "sentences.json"
    path = INPUT_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("sentences", data) if isinstance(data, dict) else data


def convert_result(sentence, result):
    """AllenNLP 출력을 기존 포맷으로 변환.

    AllenNLP 출력:
        {"verbs": [{"verb": "surged", "tags": ["O", "B-ARG1", ...]}], "words": [...]}

    변환 대상 포맷:
        {"sentence": "...", "predicates": [{"predicate": "...", "predicate_index": N,
         "arguments": [{"label": "ARG1", "text": "...", "score": 1.0}]}]}
    """
    words = result.get("words", [])
    predicates = []

    for verb_entry in result.get("verbs", []):
        verb = verb_entry["verb"]
        tags = verb_entry["tags"]

        # 서술어 인덱스 찾기: "B-V" 태그 위치
        pred_idx = None
        for i, tag in enumerate(tags):
            if tag == "B-V":
                pred_idx = i
                break

        # BIO 태그에서 인자 추출
        arguments = []
        current_label = None
        current_tokens = []

        for i, tag in enumerate(tags):
            if tag.startswith("B-") and tag != "B-V":
                # 이전 인자 저장
                if current_label:
                    arguments.append({
                        "label": current_label,
                        "text": " ".join(current_tokens),
                        "score": 1.0,
                    })
                current_label = tag[2:]
                current_tokens = [words[i]]
            elif tag.startswith("I-") and tag != "I-V":
                current_tokens.append(words[i])
            else:
                if current_label:
                    arguments.append({
                        "label": current_label,
                        "text": " ".join(current_tokens),
                        "score": 1.0,
                    })
                    current_label = None
                    current_tokens = []

        # 마지막 인자
        if current_label:
            arguments.append({
                "label": current_label,
                "text": " ".join(current_tokens),
                "score": 1.0,
            })

        predicates.append({
            "predicate": verb,
            "predicate_index": pred_idx,
            "arguments": arguments,
        })

    return {"sentence": sentence, "predicates": predicates}


def run(lang):
    """AllenNLP SRL 추론 실행."""
    sentences = load_sentences(lang)
    suffix = "_en" if lang == "en" else ""
    output_path = OUTPUT_DIR / f"allennlp-srl{suffix}.json"

    print(f"=== AllenNLP SRL (structured-prediction-srl-bert) | {len(sentences)}문장 ===\n")
    print("모델 로딩 중...")
    predictor = Predictor.from_path(MODEL_URL)
    print("모델 로딩 완료\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for i, sentence in enumerate(sentences, 1):
        print(f"[{i}/{len(sentences)}] {sentence[:60]}...")
        try:
            result = predictor.predict(sentence=sentence)
        except Exception as e:
            print(f"  오류: {e}\n")
            records.append({"sentence": sentence, "predicates": []})
            continue

        record = convert_result(sentence, result)
        records.append(record)

        # 결과 출력
        if not record["predicates"]:
            print("  서술어 탐지 없음")
        for pred in record["predicates"]:
            print(f"  서술어: {pred['predicate']} (idx={pred['predicate_index']})")
            for arg in pred["arguments"]:
                print(f"    [{arg['label']}] {arg['text']}")
        print()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"저장: {output_path}")

    # 요약
    total_preds = sum(len(r["predicates"]) for r in records)
    total_args = sum(
        len(p["arguments"]) for r in records for p in r["predicates"]
    )
    print(f"\n요약: {len(sentences)}문장 → 서술어 {total_preds}개, 인자 {total_args}개")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AllenNLP End-to-End SRL")
    parser.add_argument("--lang", choices=["ko", "en"], default="ko")
    args = parser.parse_args()
    run(args.lang)
