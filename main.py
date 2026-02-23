"""SRL 모델 평가 진입점."""
import argparse

from config import MODELS
from run_srl import run


def build_parser():
    parser = argparse.ArgumentParser(description="SRL 모델 테스트")
    parser.add_argument(
        "--model", choices=[*MODELS.keys(), "all"],
        required=True, help="테스트할 모델 (all=전체)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="처리할 문장 수 (0=전부)",
    )
    parser.add_argument(
        "--lang", choices=["ko", "en"], default="ko",
        help="입력 언어 (ko=한국어, en=영어)",
    )
    return parser


def main():
    args = build_parser().parse_args()
    models = list(MODELS.keys()) if args.model == "all" else [args.model]
    run(models, args.limit, lang=args.lang)


if __name__ == "__main__":
    main()
