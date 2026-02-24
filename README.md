# test-srl

HuggingFace SRL(Semantic Role Labeling) 모델 비교 평가.

## 구조

```
test-srl/
├── main.py              # 진입점
├── config.py            # 모델 설정, 서술어 탐지
├── inference.py         # 모델 로딩 + 추론
├── run_srl.py           # 추론 루프 + 결과 저장
├── shared/              # 공유 유틸리티 (I/O, 알림)
├── data/
│   ├── input/           # 입력 문장
│   │   ├── sentences.json      # 한국어 10문장
│   │   └── sentences_en.json   # 영어 10문장 (한국어 번역)
│   └── output/          # 모델별 추론 결과
└── srl_analysis.md      # 분석 보고서
```

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

```bash
# 전체 모델, 한국어
python main.py --model all --lang ko

# 특정 모델, 영어
python main.py --model dannashao --lang en

# 문장 수 제한
python main.py --model all --lang ko --limit 5
```

## 테스트 대상 모델

| 모델 | 베이스 | 파라미터 |
|------|--------|---------|
| dannashao | bert-base-uncased | 110M |
| electra-srl | ELECTRA-small | 13.5M |
| mbert-srl | bert-base-multilingual-cased | 178M |

3개 모델 모두 서술어 위치를 외부에서 지정해야 동작하는 argument labeling 모델.

## 분석 결과

`srl_analysis.md` 참조.
