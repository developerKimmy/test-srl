# HuggingFace SRL 모델 비교 분석

영어 기반 SRL(Semantic Role Labeling) 모델 3종을 한국어/영어 문장에 각각 적용하여 비교 평가한 결과.

## 테스트 환경

- **한국어 입력**: 비즈니스 뉴스 10문장 (전처리 파이프라인 출력)
- **영어 입력**: 한국어 10문장의 영어 번역본 (동일 의미, 교차 비교용)
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
원문:   "Pressure can be increased not only through trading profits"
마킹:   "Pressure can be [V] increased not only through trading profits"
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

### 3-2. 영어 입력 (한국어 문장 번역본 10문장)

한국어 입력과 동일한 의미의 영어 번역본으로 테스트. 동일 의미에 대한 교차 비교가 가능하다.

| 모델 | 감지된 서술어 | 감지된 인자 | 빈 서술어 | 유효 결과 | 평가 |
|------|-------------|-----------|----------|----------|------|
| dannashao | 32 | **49** | 10 (31%) | 정상 | ARG0/ARG1/ARGM 분류, 라벨 16종 사용, 핵심 인자 0.95+ |
| electra-srl | 32 | **62** | 2 (6%) | 정상 | A0/A1/AM 분류, 라벨 8종, 보조동사에서도 인자 추출 |
| mbert-srl | 32 | **1,151** | 0 | 없음 | 전 토큰 LABEL_0/LABEL_1 (언어 무관 결함) |

> **공통 실패**: 10번째 문장("SK Group Chairman Chey Tae-won drew attention...")에서 3개 모델 모두 서술어 0건 감지. 고유명사가 많은 복잡한 문장에서 서술어 탐지 휴리스틱 한계.

**dannashao 영어 예시**:
```
원문: ...Korea Development Bank stated in a response...
서술어: stated
→ [ADV] sources (score: 0.98)       ← 출처
→ [TMP] 17th (score: 0.996)         ← 시간
→ [ARG0] bank (score: 0.999)        ← 행위자
→ [LOC] response (score: 0.753)     ← 장소
→ [ARG1] considering (score: 0.983) ← 내용
```

```
원문: Pressure can be increased not only through trading profits...
서술어: increased
→ [ARG1] pressure (score: 0.997)    ← 대상
→ [MOD] can (score: 0.999)          ← 양태
→ [MNR] profits (score: 0.838)      ← 수단
```

**electra-srl 영어 예시**:
```
원문: ...domestic medical device market growth and delayed earnings recovery...had an impact.
서술어: had
→ [A0] recovery (score: 0.993)      ← 행위자
→ [A1] impact (score: 0.997)        ← 대상

서술어: delayed
→ [A1] earnings (score: 0.998)      ← 대상
→ [A2] recovery (score: 0.994)      ← 결과
```

```
원문: This trend is also leading to increased convenience store sales.
서술어: is
→ [A1] trend (score: 0.997)         ← 대상
→ [DIS] also (score: 0.884)         ← 담화표지
→ [A2] leading (score: 0.993)       ← 보어
```

**mbert-srl 영어 예시**:
```
원문: Due to the Dubai chewy cookie craze, demand...surged...
서술어: dried
→ [LABEL_1] Due (0.503) → [LABEL_1] to (0.585) → [LABEL_1] the (0.580) → ...
  (48개 토큰 전부 LABEL_0/LABEL_1 — 영어에서도 분류 기능 없음)
```

---

## 4. dannashao vs electra-srl 상세 비교

영어에서 정상 동작한 두 모델의 품질 비교. 한국어 번역 문장으로 동일 의미에 대한 인자 추출 차이를 확인했다.

### 동일 서술어 결과 대조

| 문장 (한국어 원문 → 영어) | 서술어 | dannashao | electra-srl | 우세 |
|------|--------|-----------|-------------|------|
| 산은은...밝혔다 → ...stated... | stated | ADV:sources, TMP:17th, ARG0:bank, LOC:response, ARG1:considering | A0:bank, LOC:in | dannashao |
| ...수요가 급증 → ...surged... | surged | CAU:craze, ARG1:demand, EXT:much | A0:kadaif, MNR:much | dannashao |
| ...영향을 미친 → ...had an impact | had | PRR:impact | A0:recovery, A1:impact | electra |
| 포트폴리오도 갖췄습니다 → ...has...portfolio | has | DIS:addition, ARG0:it, ARG1:portfolio | LOC:in, A0:it, A1:portfolio | dannashao |
| AI 인프라 관련 ETF → ETFs related to AI... | related | ARG1:etfs, ARG2:infrastructure | A0:etfs, A1:to | dannashao |

### 비교 분석

| 관점 | dannashao | electra-srl |
|------|-----------|-------------|
| **핵심 인자 (ARG0/ARG1)** | 거의 동일 | 거의 동일 |
| **수식 인자 (ARGM)** | CAU(원인), TMP(시간), MOD(양태), DIS(담화) 등 세밀하게 구분 | LOC/TMP는 정확하나 CAU를 TMP/LOC로 태깅하는 경향 |
| **인자 텍스트** | 내용어(명사) 포착 경향 (infrastructure, platforms, profits) | 기능어(전치사) 포착 경향 (in, to, on, through) |
| **보조동사 처리** | 빈 결과 다수 (31% 빈 서술어) | 보조동사에서도 인자 추출 (6% 빈 서술어) |
| **세분화** | 60종 라벨, 16종 실사용 (ARG3=가격, EXT=정도 등) | 18종 라벨, 8종 실사용 |
| **score 분포** | 핵심 인자 0.95+, 수식 인자 0.60~0.99 | 핵심 인자 0.95+, 수식 인자 0.40~0.99 |
| **서브워드 누출** | 간헐적 (##maker 등) | 다소 빈번 (##f, ##ptively 등) |
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
| **dannashao** | 우수 (49 인자/32 서술어, 라벨 16종) | 사용 불가 | 간편 | 영어 SRL 최선 선택지. 세밀한 수식 인자 분류 (CAU, MOD, DIS 등) |
| **electra-srl** | 양호 (62 인자/32 서술어, 라벨 8종) | 사용 불가 | 간편 | 13.5M 소형 모델치고 양호. 보조동사 처리에서 우위 |
| **mbert-srl** | 사용 불가 (LABEL_0/1만 출력) | 사용 불가 | 불가 | classification head 누락. 언어와 무관하게 동작하지 않음 |

### 시사점

- 영어에서 dannashao와 electra-srl 모두 **높은 정확도로 SRL 수행** (핵심 인자 score 0.95+)
- dannashao는 **내용어(명사) 중심**으로 인자를 포착하고, electra는 **기능어(전치사) 중심** 경향 — 용도에 따라 선택
- 동일 의미의 한국어/영어 비교를 통해 영어 SRL의 세밀함 vs 한국어 SRL의 부재가 명확히 드러남
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
- 영어 입력: `data/input/sentences_en.json` (한국어 문장 영어 번역본)
