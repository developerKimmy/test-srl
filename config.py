"""SRL 평가 설정."""
import re
from pathlib import Path

from shared import load_json

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "data" / "input"
OUTPUT_DIR = BASE_DIR / "data" / "output"

_RAW = load_json(INPUT_DIR / "sentences.json") or {}
SENTENCES = _RAW.get("sentences", []) if isinstance(_RAW, dict) else _RAW

_RAW_EN = load_json(INPUT_DIR / "sentences_en.json") or {}
SENTENCES_EN = _RAW_EN.get("sentences", []) if isinstance(_RAW_EN, dict) else _RAW_EN

MODELS = {
    "dannashao": {
        "id": "dannashao/bert-base-uncased-finetuned-advanced-srl_arg",
    },
    "electra-srl": {
        "id": "JorgeTC/electra-SRL",
    },
    "mbert-srl": {
        "id": "liaad/srl-en_mbert-base",
    },
}

# ── 한국어 서술어 탐지 ──

_VERB_ENDINGS = (
    "했다", "됐다", "된다", "한다", "하다",
    "었다", "겠다", "였다", "냈다", "갔다", "왔다",
    "줬다", "봤다", "셨다", "렸다", "졌다", "혔다",
    "났다", "있다", "없다", "이다", "인다",
    "섰다", "뤘다",
    "습니다", "됩니다", "입니다", "랍니다",
)

_NOT_PREDICATE = re.compile(
    r"^(그보다|이보다|저보다|보다|위하다|대하다|관하다)$"
    r"|보다$"
)


def detect_predicates_ko(tokens):
    """한국어 동사 어미 패턴으로 predicate 후보 인덱스를 반환."""
    indices = []
    for i, token in enumerate(tokens):
        clean = token.rstrip(".!?。,;")
        if _NOT_PREDICATE.search(clean):
            continue
        for ending in _VERB_ENDINGS:
            if clean.endswith(ending):
                indices.append(i)
                break
    return indices


# ── 영어 서술어 탐지 ──

_EN_COMMON_VERBS = {
    "is", "are", "was", "were", "has", "have", "had",
    "said", "made", "did", "got", "went", "came", "took",
    "cut",
}

_EN_VERB_SUFFIXES = ("ed", "es")

_EN_NOT_VERB = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "about", "after", "before", "between",
    "amid", "following", "compared", "based", "powered", "concerned",
}


def detect_predicates_en(tokens):
    """영어 동사 패턴으로 predicate 후보 인덱스를 반환."""
    indices = []
    for i, token in enumerate(tokens):
        clean = token.rstrip(".!?,;:\"'").lower()
        if clean in _EN_NOT_VERB:
            continue
        if clean in _EN_COMMON_VERBS:
            indices.append(i)
            continue
        for suffix in _EN_VERB_SUFFIXES:
            if clean.endswith(suffix) and len(clean) > 3:
                indices.append(i)
                break
    return indices
