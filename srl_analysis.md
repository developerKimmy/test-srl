# HuggingFace SRL 모델 비교 분석

영어 기반 SRL(Semantic Role Labeling) 모델 3종을 한국어/영어 문장에 각각 적용하여 비교 평가한 결과.

## 테스트 환경

- **한국어 입력**: 비즈니스 뉴스 10문장 (전처리 파이프라인 출력)
- **영어 입력**: 비즈니스 뉴스 스타일 10문장
- **추론**: HuggingFace Transformers `token-classification` 파이프라인

### 모델 입력 구조에 대한 주의사항

테스트한 3개 모델 모두 **end-to-end SRL 모델이 아니다**. 문장을 그대로 넣으면 서술어를 스스로 찾지 못하며, 서술어 위치를 외부에서 지정해줘야 동작한다.

| 모델 | 서술어 입력 방식 | 마커 없이 실행 시 |
|------|-----------------|-----------------|
| dannashao | `[V]` 토큰을 서술어 앞에 삽입 | 전 토큰 `_`(outside) — 인자 0건 |
| electra-srl | `[V]` 토큰을 서술어 앞에 삽입 | 전 토큰 `O`(outside) — 인자 0건 |
| mbert-srl | `token_type_ids`로 서술어 위치 마킹 | LABEL_0/LABEL_1만 출력 (classification head 미학습) |

즉, 이 모델들의 역할은 **"서술어가 주어졌을 때, 해당 서술어의 인자(arguments)를 찾는 것"**이며, 서술어 탐지(predicate identification)는 모델 범위 밖이다. 본 테스트에서는 한국어 동사 어미 패턴과 영어 동사형 휴리스틱으로 서술어를 탐지한 뒤 `[V]` 마커를 삽입하여 테스트했다. 서술어 탐지 정확도가 모델 성능 평가에 영향을 줄 수 있다는 점을 감안해야 한다.

---

## 1. 모델 상세 스펙

### 1-1. dannashao/bert-base-uncased-finetuned-advanced-srl_arg

| 항목 | 내용 |
|------|------|
| **베이스 모델** | `bert-base-uncased` |
| **아키텍처** | BertForTokenClassification |
| **파라미터 수** | ~110M |
| **학습 데이터** | Universal PropBank English v1.0 |
| **학습 설정** | LR 2e-05, batch 16, 3 epochs, Adam |
| **보고 성능** | F1 86.69% (epoch 3) |
| **라벨 수** | 60종 |
| **라벨 형식** | BIO 접두사 없음 (ARG0, ARGM-TMP, _ 등) |
| **서술어 입력 방식** | `[V]` 마커를 서술어 앞에 삽입 |
| **문서화** | 양호 (모델 카드 + 노트북) |

**라벨 체계 상세**:

| 분류 | 라벨 |
|------|------|
| 핵심 인자 | ARG0, ARG1, ARG1-DSP, ARG2, ARG3, ARG4, ARG5, ARGA |
| 수식 인자 (ARGM-*) | ADJ, ADV, CAU, COM, CXN, DIR, DIS, EXT, GOL, LOC, LVB, MNR, MOD, NEG, PRD, PRP, PRR, REC, TMP |
| 계속 (C-*) | C-ARG0, C-ARG1, C-ARG2, C-ARG3, C-ARG4, C-ARGM-* 등 |
| 참조 (R-*) | R-ARG0, R-ARG1, R-ARG2, R-ARGM-* 등 |
| Outside | `_` (역할 없음) |

**입력 예시**:
```
원문:   "Amazon acquired a robotics startup"
마킹:   "Amazon [V] acquired a robotics startup"
         ↑ [V] 마커를 서술어 바로 앞에 삽입
```

---

### 1-2. JorgeTC/electra-SRL

| 항목 | 내용 |
|------|------|
| **베이스 모델** | ELECTRA-small (커스텀, uncased) |
| **아키텍처** | ElectraForTokenClassification |
| **파라미터 수** | ~13.5M (매우 소형) |
| **학습 데이터** | 미공개 |
| **보고 성능** | 미공개 |
| **라벨 수** | 18종 |
| **라벨 형식** | BIO 접두사 없음 (A0, AM-TMP, O 등) |
| **서술어 입력 방식** | `[V]` 토큰 삽입 (vocab에 추가, id=30522) |
| **문서화** | 부실 (자동 생성 스텁) |

**라벨 체계 상세**:

| ID | 라벨 | 의미 |
|----|------|------|
| 0 | A0 | 행위자 (Agent) |
| 1 | A1 | 대상 (Patient) |
| 2 | A2 | 도구/수혜자/속성 |
| 3 | A3 | 출발점/수혜자 |
| 4 | A4 | 도착점 |
| 5 | AM-ADV | 부사어 |
| 8 | AM-DIR | 방향 |
| 11 | AM-LOC | 장소 |
| 12 | AM-MNR | 방식 |
| 13 | AM-MOD | 양태 |
| 14 | AM-NEG | 부정 |
| 15 | AM-TMP | 시간 |
| 16 | O | Outside (역할 없음) |
| 17 | SU | 보조/종속 |

**특이점**: dannashao 대비 라벨 수가 1/3 수준 (60 → 18). ARG 대신 A 접두사, ARGM 대신 AM 접두사 사용. ELECTRA-small 기반이라 파라미터 수가 매우 적음.

---

### 1-3. liaad/srl-en_mbert-base

| 항목 | 내용 |
|------|------|
| **베이스 모델** | `bert-base-multilingual-cased` (mBERT) |
| **아키텍처** | BertModel + 외부 Linear + Viterbi 디코딩 |
| **파라미터 수** | ~178M |
| **학습 데이터** | CoNLL-2012 (OntoNotes v5.0 영어) |
| **보고 성능** | F1 63.07% (PropBank.Br 도메인 내) |
| **라벨 수** | ~50+ (BIO 태깅, 동적) |
| **라벨 형식** | BIO 접두사 (B-A0, I-A0, O 등) |
| **서술어 입력 방식** | `token_type_ids`로 서술어 위치를 1로 마킹 |
| **문서화** | 보통 (논문 + GitHub) |

**원래 의도된 사용법**:
```python
# HuggingFace pipeline으로는 사용 불가
# GitHub 저장소(asofiaoliveira/srl_bert_pt)의 전체 파이프라인 필요:
#   1. BertModel로 토큰 임베딩 추출
#   2. 별도 저장된 Linear layer로 라벨 분류
#   3. Viterbi 디코딩으로 BIO 시퀀스 보정
```

**근본적 문제**: HuggingFace에 올라간 가중치는 `BertModel`(feature extractor)만 포함. `AutoModelForTokenClassification`으로 로드하면 classification head가 랜덤 초기화되어 의미 없는 LABEL_0/LABEL_1만 출력. 원래는 포르투갈어 SRL 연구용으로, 영어 CoNLL-2012 데이터를 포르투갈어 PropBank.Br 형식에 맞춰 변환한 것.

**출처**: Oliveira, Loureiro & Jorge (2021), *Transformers and Transfer Learning for Improving Portuguese Semantic Role Labeling*

---

## 2. 모델 비교 요약

| 항목 | dannashao | electra-srl | mbert-srl |
|------|-----------|-------------|-----------|
| **베이스** | BERT-base (영어) | ELECTRA-small (영어) | mBERT (다국어) |
| **파라미터** | 110M | 13.5M | 178M |
| **학습 데이터** | Universal PropBank EN | 미공개 | CoNLL-2012 |
| **라벨 수** | 60 | 18 | ~50+ (BIO) |
| **라벨 형식** | ARG0, ARGM-TMP, _ | A0, AM-TMP, O | B-A0, I-A0, O |
| **서술어 마킹** | `[V]` 토큰 삽입 | `[V]` 토큰 삽입 | token_type_ids |
| **보고 F1** | 86.69% | 미공개 | 63.07% |
| **HF Pipeline 호환** | O | O | X (별도 파이프라인 필요) |
| **토크나이저 언어** | 영어 전용 | 영어 전용 | 104개 언어 |

---

## 3. 테스트 결과

### 3-1. 한국어 입력 (비즈니스 뉴스 10문장)

| 모델 | 감지된 서술어 | 감지된 인자 | 유효 결과 | 실패 원인 |
|------|-------------|-----------|----------|----------|
| dannashao | 9 | **0** | 없음 | 영어 토크나이저가 한국어를 `[UNK]`로 처리 |
| electra-srl | 9 | **1** | 없음 | `수위를 높일` 1건만 A1 태깅, 나머지 전부 O |
| mbert-srl | 9 | **전 토큰** | 없음 | 전 토큰 LABEL_1 (classifier head 미학습) |

**dannashao 한국어 예시**:
```
원문: 17일 투자은행(IB) 업계에 따르면 산은은... 밝혔다.
서술어: 밝혔다.
→ 감지된 역할 없음
  (한국어 토큰이 전부 [UNK]로 변환되어 분류 불가)
```

**electra-srl 한국어 예시**:
```
원문: 매매 차익뿐만 아니라 보유 단계에서도 압박 수위를 높일 수 있다.
서술어: 있다.
→ [A1] 수위를 높일 (score: 0.6614)
  (10문장 통틀어 유일하게 감지된 인자)
```

**mbert-srl 한국어 예시**:
```
원문: 의료기기 내수 성장세 둔화와... 분석됐다.
서술어: 분석됐다.
→ [LABEL_1] 의 (0.5747)
→ [LABEL_1] 료 (0.5603)
→ [LABEL_1] 기 (0.5563)
→ ... (모든 토큰이 동일 라벨 — 분류 기능 없음)
```

### 3-2. 영어 입력 (비즈니스 뉴스 10문장)

| 모델 | 감지된 서술어 | 감지된 인자 | 유효 결과 | 평가 |
|------|-------------|-----------|----------|------|
| dannashao | 19 | **38** | 정상 | ARG0/ARG1/ARGM 정확하게 분류, 평균 score 0.95+ |
| electra-srl | 19 | **34** | 정상 | A0/A1/AM 분류, dannashao와 유사 수준 |
| mbert-srl | 19 | **전 토큰** | 없음 | 여전히 전 토큰 LABEL_1 (언어 무관 결함) |

**dannashao 영어 예시**:
```
원문: Amazon acquired a robotics startup for $1.7 billion.
서술어: acquired
→ [ARG0] amazon (score: 0.9973)     ← 행위자
→ [ARG1] startup (score: 0.9994)    ← 대상
→ [ARG3] $ (score: 0.6936)          ← 가격
```

```
원문: The CEO resigned following allegations of financial misconduct.
서술어: resigned
→ [ARG0] ceo (score: 0.8546)        ← 행위자
→ [CAU] allegations (score: 0.8105)  ← 원인
```

```
원문: The Federal Reserve raised interest rates for the third consecutive time.
서술어: raised
→ [ARG0] reserve (score: 0.9984)    ← 행위자
→ [ARG1] rates (score: 0.9990)      ← 대상
→ [TMP] time (score: 0.9950)        ← 시간
```

**electra-srl 영어 예시**:
```
원문: Oil prices surged after OPEC decided to cut production.
서술어: cut
→ [A0] opec (score: 0.9977)         ← 행위자
→ [A1] production (score: 0.9963)   ← 대상

서술어: surged
→ [A1] prices (score: 0.9966)       ← 대상
→ [TMP] after (score: 0.9840)       ← 시간
```

**mbert-srl 영어 예시**:
```
원문: The company reported a 20% increase in quarterly revenue.
서술어: reported
→ [LABEL_1] The (0.6134)
→ [LABEL_1] company (0.5685)
→ [LABEL_1] reported (0.5899)
→ ... (영어에서도 전 토큰 동일 라벨 — 모델 자체 결함)
```

---

## 4. dannashao vs electra-srl 상세 비교

영어에서 정상 동작한 두 모델의 품질 비교.

### 동일 문장 결과 대조

| 문장 | 서술어 | dannashao | electra-srl |
|------|--------|-----------|-------------|
| The company reported... | reported | ARG0: company, ARG1: increase | A0: company, A1: increase |
| Apple announced... | announced | ARG0: apple, ARG1: chip, LOC: conference | A0: apple, A1: chip, LOC: at |
| ...was approved by regulators. | approved | ARG1: merger, ARG0: regulators | A0: merger, A0: regulators |
| OPEC decided to cut... | decided | ARG0: opec, ARG1: cut | A0: opec |
| The CEO resigned... | resigned | ARG0: ceo, CAU: allegations | A0: ceo, TMP: following |
| Amazon acquired... | acquired | ARG0: amazon, ARG1: startup, ARG3: $ | A0: amazon, A1: startup |
| ...raised interest rates | raised | ARG0: reserve, ARG1: rates, TMP: time | A0: reserve, A1: rates, TMP: for |
| The stock market crashed... | crashed | ARG1: market, CAU: fears | A0: market, LOC: amid |

### 비교 분석

| 관점 | dannashao | electra-srl |
|------|-----------|-------------|
| **핵심 인자 (ARG0/ARG1)** | 거의 동일 | 거의 동일 |
| **수식 인자 (ARGM)** | CAU(원인), TMP(시간), LOC(장소) 구분 정확 | LOC/TMP는 정확하나 CAU를 TMP/LOC로 태깅하는 경향 |
| **세분화** | 60종 라벨로 세밀한 분류 (ARG3=가격 등) | 18종이라 거친 분류 |
| **score 분포** | 핵심 인자 0.99+, 수식 인자 0.67~0.99 | 핵심 인자 0.99+, 수식 인자 0.40~0.99 |
| **오분류** | prices, Sales 등 명사를 서술어로 처리 시 부정확 | 동일 |
| **모델 크기** | 110M (8배 큼) | 13.5M |

---

## 5. 결론

### 한국어 SRL 현황

- HuggingFace에 공개된 **한국어 SRL 모델은 0건** (API 검색: `srl`, `korean+srl`, `한국어+의미역`)
- 영어 SRL 모델 3종 모두 한국어에서 **사용 불가** 확인
- mBERT 기반 모델도 다국어 토크나이저만 공유할 뿐, SRL classification head가 영어로만 학습되어 한국어에서 무의미
- GitHub에서 발견된 유일한 한국어 SRL 프로젝트: [machinereading/BERT_for_Korean_SRL](https://github.com/machinereading/BERT_for_Korean_SRL) (HuggingFace 미등록)

### 모델 평가 요약

| 모델 | 영어 | 한국어 | HF 사용성 | 총평 |
|------|------|--------|----------|------|
| **dannashao** | 우수 (F1 86.69%) | 사용 불가 | 간편 | 영어 SRL 최선 선택지. 라벨 60종으로 세밀한 분류 가능 |
| **electra-srl** | 양호 | 사용 불가 | 간편 | 소형 모델(13.5M)치고 양호하나 문서 부재 |
| **mbert-srl** | 사용 불가 | 사용 불가 | 불가 | classification head 누락. 언어와 무관하게 동작하지 않음 |

### 시사점

- 영어에서 dannashao와 electra-srl 모두 **높은 정확도로 SRL 수행** (핵심 인자 score 0.99+)
- 한국어 SRL은 **직접 파인튜닝이 유일한 방법** — 동일 아키텍처(BERT/ELECTRA)에 한국어 SRL 데이터로 학습하면 영어 수준의 결과를 기대할 수 있음
- 베이스 모델로는 한국어에 특화된 **KoELECTRA** (`monologg/koelectra-base-v3-discriminator`) 등이 적합

---

## 모델 조사 방법

1. **HuggingFace Hub API**: `huggingface.co/api/models?search=srl&pipeline_tag=token-classification` — 30건 중 한국어 0건
2. **HuggingFace Hub API**: `search=korean+srl`, `search=한국어+의미역` — 결과 0건
3. **GitHub**: `machinereading/BERT_for_Korean_SRL` — 유일한 한국어 SRL 프로젝트 (HuggingFace 미등록)
4. **기타 플랫폼**: PyTorch Hub, TensorFlow Hub, ModelScope — SRL 모델 거의 없음, 한국어 0건

## 결과 파일

- 한국어: `data/output/dannashao.json`, `electra-srl.json`, `mbert-srl.json`
- 영어: `data/output/dannashao_en.json`, `electra-srl_en.json`, `mbert-srl_en.json`
- 영어 입력: `data/input/sentences_en.json`
