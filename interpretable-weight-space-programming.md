# Interpretable weight-space programming builds transparent transformers and brain encoding models

> **해석 가능한 가중치 공간 프로그래밍(iWSP): LLM 에이전트가 가중치를 직접 코드로 써서 투명한 트랜스포머와 뇌 인코딩 모델을 만든다**

| 항목 | 내용 |
|---|---|
| 저자 | Yash Mehta, Chandan Singh, Weijia Xu, Richard J. Antonello, Adeen Flinker, Nima Mesgarani, Michael F. Bonner, Nebojsa Jojic, Jianfeng Gao |
| 소속 | Microsoft Research, Johns Hopkins University, Columbia University, New York University |
| arXiv | [2611.18114](https://arxiv.org/abs/2611.18114) (2026-11-13, cs.CE) |
| 코드 | [github.com/csinva/weight-space-programming-autoresearch](https://github.com/csinva/weight-space-programming-autoresearch) |

---

## TL;DR

- **아이디어**: 모델을 역전파로 학습시키지 않고, **LLM 에이전트가 모델의 가중치와 특징(feature)을 전부 코드로 직접 작성**하게 한다. 에이전트는 고정된 평가기가 주는 검증 점수 하나만 보고 반복적으로 모델을 고친다.
- **목표**: 예측 성능이 좋은 것만이 아니라 **사람이 읽을 수 있는 모델**을 만드는 것. 모든 부분이 의도적으로 작성되었으므로 각 구성요소가 무엇을 하는지 알 수 있다.
- **결과**
  - 알고리즘 과제(덧셈, 정렬, GCD 등): Opus로 돌린 26개 실행 모두 hidden set 정확도 **100%**
  - 시각 피질 fMRI: 처음 보는 데이터셋(NSD)에서 해석 가능 베이스라인 7종을 모두 능가 (초기 시각피질 0.443 vs 0.400, 복측 0.317 vs 0.256). 단, 복측에서는 AlexNet(0.391)에 뒤짐
  - 언어 fMRI: 읽을 수 있는 모델 덕분에 **데이터셋의 교란 변수(스캔 시작·끝 부분 효과)를 발견**
  - 청각 ECoG: 해석 가능한 음향 특징을 찾았고, Whisper와 결합하면 Whisper 단독보다 예측이 좋아짐
- **해석 가능성 검증**: 모델을 200단어 이하로 설명하고, 새 에이전트가 그 설명만으로 모델을 다시 만들어도 성능이 거의 유지됨

---

## 1. 배경: 왜 이 연구가 필요한가

신경망은 예측은 잘하지만 내부가 불투명하다. 과학에서는 예측보다 **설명**이 목적인 경우가 많아서 이 불투명성이 큰 문제가 된다.

신경과학의 **인코딩 모델**이 대표적인 예다. 딥러닝 특징(CNN, LLM, 음성 모델 임베딩)에 선형 회귀를 붙이면 자연 자극(이미지, 이야기, 소리)에 대한 뇌 반응을 잘 예측한다. 그러나 **자극의 어떤 특징이 그 예측을 만들어내는지**는 알려주지 못한다. 반대로 Gabor 필터 같은 전통적 수작업 특징은 해석은 쉽지만 예측력이 떨어진다.

이 논문은 둘 사이의 간극을 LLM 에이전트로 메우려 한다. 에이전트가 사전 지식(문헌, 계산신경과학 지식)을 활용해 **해석 가능한 계산을 직접 설계·구현·개선**하게 하는 방식이다.

---

## 2. 방법: Interpretable Weight-Space Programming (iWSP)

![Figure 1. iWSP 루프](https://raw.githubusercontent.com/csinva/weight-space-programming-autoresearch/4f8eb61580fe9aa307c868d0bdd86914e673ef1f/.github/readme/fig1_loop.png)

*Figure 1 (논문 Fig. 1, 저자 저장소 원본). **(a)** LLM 에이전트가 과제를 받아 모델의 모든 가중치를 코드로 작성한다(`optimizer.step()` 금지). 고정된 평가기가 출력을 정답과 비교하고, 그 점수가 다음 수정을 이끈다. 알고리즘·NLP 과제에서는 직접 답을 출력하는 수작업 트랜스포머를 만들어 exact-match 정확도로 평가하고, 뇌 인코딩에서는 특징 임베딩을 출력하는 신경망을 만들어 리지 회귀 후 Pearson 상관으로 평가한다. **(b)** 해석 가능성 감사: LLM 심사관이 규칙 준수(학습 없음, 해석 가능한 편집)를 확인하고, "병목 테스트"로 모델을 200단어 이하로 설명한 뒤 새 에이전트가 설명만으로 모델을 재구성한다. **(c)** 원본 대비 재구성 모델의 hidden accuracy.*

### 2.1 핵심 규칙

| 요소 | 설명 |
|---|---|
| **학습 금지** | 경사하강법(backprop)을 쓰지 않는다. 에이전트가 가중치 값을 하나하나 코드로 대입한다 (예: `model.layer1[0, 3] = 1.0`). |
| **고정된 평가기** | 에이전트가 수정할 수 없다. 알고리즘/NLP: 토큰 단위로 답을 생성해 정확도 측정. 뇌 인코딩: 모델이 낸 특징 위에 **선형 리지 회귀만** 적합한 뒤 Pearson 상관으로 채점 (적합되는 유일한 부분). |
| **정보 제한** | 매 반복마다 에이전트는 **검증 점수 하나**만 본다. 그래서 데이터를 외우기보다 이미 아는 지식(예: 시각 피질의 Gabor 유사 반응, 음향 특징)을 활용해야 한다. |
| **데이터 분할** | visible(검증) 세트로 탐색하고, 최종 성능은 에이전트가 본 적 없는 hidden 세트로 보고한다. 시각 실험은 피험자와 이미지를 모두 분리했다. |

### 2.2 해석 가능성을 어떻게 검증하나

1. **LLM 감사(audit)**: 저장된 각 모델을 LLM 심사관이 검사한다. 학습을 하지 않았는지, 규칙을 지켰는지, 편집이 해석 가능한지 확인한다. 저장소에는 코드에서 학습 흔적을 찾는 스캔, 데이터를 바꿔 재구성한 뒤 가중치를 비교하는 검사, 세션 로그 검사가 사후 감사로 포함되어 있다. 하네스가 에이전트 자신의 Python 코드 안에서 학습하는 것까지 막을 수는 없기 때문이다.
2. **텍스트 병목 테스트(description-and-rebuild)**: LLM이 모델을 **최대 200단어**로 설명하고, 새로운 세션의 에이전트가 **그 설명만으로** 모델을 다시 만든다. 재구성 모델이 원본 성능을 유지하면 모델이 짧은 자연어로 담길 만큼 이해 가능하다는 근거가 된다. 설명 없이 같은 과제만 받은 대조군과도 비교한다.

---

## 3. 실험 결과

### 3.1 수작업 트랜스포머: 알고리즘 및 NLP 과제

**과제 구성**
- **알고리즘 13개**: 5자리 덧셈·곱셈, 5자리 정렬, 숫자 세기, 패리티(최대 10비트), 5비트 불리언 회로, 두 점 선형 보간, 3×3 단어 뒤집기, 3자리 GCD, 8비트 10진→2진 변환, 편집 거리, 최소 소인수, 날짜 정규화
- **NLP 4개**: SST-2(감성), MRPC(패러프레이즈), SNLI(자연어 추론), TREC(질문 분류)

**결과**
- 알고리즘: 감사를 통과한 **Opus 실행 26개(13과제 × 2시드) 모두 hidden 예제에서 100% 정확도**. 단, 가능한 모든 입력에 대해 정답을 보장한다는 뜻은 아니다.
- NLP: TREC 87.3%. 4개 중 3개 과제에서 다수 클래스 기준선을 크게 넘었지만, **패러프레이즈 판별(MRPC)에서는 개선이 없었다.**
- 텍스트 병목: 알고리즘 13과제의 재구성 모델은 **모두 100%** hidden accuracy. NLP 재구성 모델은 평균 **약 1.4%p**만 하락했다.
- 예시: Figure 1(b)의 덧셈 모델은 파라미터 27개짜리로, "cross-attention 한 번으로 덧셈을 하고 합을 10칸 원형 고리 위의 각도로 다룬다"는 식으로 설명된다.

![NLP 과제 텍스트 병목 결과](https://raw.githubusercontent.com/csinva/weight-space-programming-autoresearch/4f8eb61580fe9aa307c868d0bdd86914e673ef1f/analyze/nl_bottleneck/fig1_hidden_acc.png)

*저자 저장소 `analyze/nl_bottleneck` 그림. NLP 4개 과제의 hidden accuracy: 원본 모델(검정), 설명으로부터 재구성한 모델 2개(파랑 계열), 설명 없이 처음부터 만든 대조군(주황), 다수 클래스 기준선(회색). MRPC는 모든 모델이 다수 클래스 수준이다.*

### 3.2 시각 피질 인코딩 모델 (fMRI)

**설정**
- 7T fMRI 데이터셋 **LAION-fMRI**에서 에이전트 검색(후보 600개 채점)을 하고, 최종 모델(`erjpu`)을 고정한 뒤 검색에 쓰지 않은 **NSD(Natural Scenes Dataset)** 에서 제로샷으로 평가했다.
- NSD 평가: 피험자 01, 02, 05, 07. 리드아웃은 각 피험자 고유 이미지로 적합하고, LAION 자극과 겹치지 않는 공유 이미지 998장으로 테스트했다.
- 모델은 Gabor 필터링, 표면 기하(surface geometry), 윤곽 곡률(contour curvature), 채널 간 상관 같은 **명시적 계산**으로 구성된다. 특징 수는 9,996개다.

**NSD 성능 (encoding r)**

| 영역 | iWSP (erjpu) | 최고 해석 가능 베이스라인 | AlexNet |
|---|---|---|---|
| 초기 시각피질 (early) | **0.443** | 0.400 | iWSP와 비슷 |
| 복측 시각피질 (ventral) | **0.317** | 0.256 | **0.391** |

→ Gabor, GIST, HMAX, scattering 등 **해석 가능한 베이스라인 7종을 모두 능가**했다. 다만 복측 피질에서는 사전학습된 AlexNet에 뒤처진다.

**시각 영역별로 다른 계산** (LAION-fMRI 5명 피험자, Shapley 기여도)
- 국소 평균 표면 기하의 비중은 **V1 37.9% → PPA 14.8%** 로 줄어든다.
- 윤곽 곡률의 비중은 **V1 3.4% → FFA 14.9%** 로 늘어난다.
- 즉 초기 영역은 모서리·윤곽 같은 국소 구조, 고위 영역(FFA, PPA)은 텍스처 통계 쪽 계산으로 더 잘 설명된다.
- ⚠️ 저자들은 이 값이 **리드아웃 가중치 배분의 비율**일 뿐이고, 해당 계산이 고유하게 필요하다는 인과적 증거는 아니라고 명시한다.

![시각 피질 최종 모델 erjpu 구조](https://raw.githubusercontent.com/csinva/weight-space-programming-autoresearch/4f8eb61580fe9aa307c868d0bdd86914e673ef1f/analyze/erjpu_diagram/erjpu_architecture.png)

*저자 저장소 `analyze/erjpu_diagram` 그림. 최종 시각 피질 모델(erjpu)의 모든 단계, 채널, 출력 특징. 학습된 가중치는 없다. 모든 필터는 해석적으로 정하거나 고정 시드로 생성하고, 적합되는 단계는 리지 회귀 하나뿐이다. 입력(픽셀 압축, 반대색 채널) → V1 유사 단계(3옥타브 30채널: 휘도 Gabor 에너지 16, 휘도 blob 2, 색 Gabor 에너지 8, 색 blob 4, 국소 에너지로 나누는 정규화) → 중간 모듈(텍스처 경계, 방향장과 윤곽 곡률, 표면 기하, 동시발생 통계, 전역 배치 등) → 9,996개 특징 → 리지 회귀.*

> 참고: 논문 본문의 "시각 영역별 계산 비중" 그림은 저장소에 이미지 파일이 없어서 위의 수치로 대신했다.

### 3.3 이야기 청취 언어 fMRI: 데이터 교란 변수 발견

**설정**: 팟캐스트 이야기를 듣는 동안 기록한 fMRI (피험자 UTS03, OpenNeuro ds003020). 비교 기준선은 GPT-2 XL 특징(전체 voxel 평균 상관 0.079).

**무슨 일이 있었나**
1. 검색 도중 평균 voxel 상관이 **한 번의 수정으로 0.067 → 0.104** 로 크게 뛰었다.
2. 모델이 읽을 수 있었으므로 원인을 바로 확인할 수 있었다. 새 특징은 단어 의미가 아니라 **이야기 안에서의 위치**(story position, 푸리에·체비쇼프·르장드르 기저, RBF 등)를 인코딩하고 있었다.
3. **위치 특징만으로** visible 세트 약 0.090. GPT-2 XL(0.079)보다 높다.
4. 위치 특징은 스캔 가장자리의 평균 반응을 split-half 데이터에서 **r = 0.97** 로 추적했다. 이는 각성·주의, 시작 과도 반응처럼 내용과 무관한 신호다.
5. 각 스캔 끝에서 30 TR씩 잘라내자 hidden 점수가 **0.043 → 0.023** 으로 떨어졌다.

**권고**: 이후 연구에서는 스캔 양끝에서 **약 40 TR**을 잘라낼 것.

**가장자리를 자른 뒤 다시 검색한 결과**: 0.0851 vs GPT-2 XL 0.0826으로 **비슷한 수준**이었지만, 모든 영역에서 이기지는 못했다.

![스토리 위치 기저와 평균 fMRI 반응](https://raw.githubusercontent.com/csinva/weight-space-programming-autoresearch/4f8eb61580fe9aa307c868d0bdd86914e673ef1f/experiments/runs/neuro/fmri-may27-run1/analysis/storyarc_vs_response.png)

*저자 저장소 `fmri-may27-run1/analysis` 그림. 이야기 6개 각각에 대해 z-점수화한 평균 fMRI 반응(회색: 원신호, 검정: 평활)과 위치 기저 두 개(빨강: 기본 푸리에 sine, 파랑: 선형 p−½)를 겹쳐 그렸다. 평활 반응은 이야기 시작의 급상승과 끝부분의 변동이 지배하며, 이 모양은 학습·테스트 이야기 모두에서 비슷하다. 그래서 위치 특징만으로도 점수가 크게 오른다.*

> **의의**: 블랙박스 모델이었다면 "점수가 올랐다"로 끝났을 결과다. 해석 가능한 모델이었기에 **점수 상승이 데이터 아티팩트 때문**이라는 사실이 드러났다.

### 3.4 청각 ECoG: 원시 파형에서 해석 가능한 음향 특징 찾기

**설정**: 오디오를 듣는 환자의 ECoG(피질 전극) 반응. 한 환자(약 10시간 분량)의 데이터로 검색했고, 에이전트가 **원시 파형(raw waveform)** 에서 해석 가능한 특징을 설계했다(저장소에서는 이 모델을 LEAN이라 부른다). 비교 기준은 Whisper와 교과서적 음향 특징이다.

**결과**

| 비교 (hidden set, 전극별 평균 상관) | 값 |
|---|---|
| 교과서적 음향 특징 | 0.059 |
| **iWSP 단독** | **0.074** |
| Whisper 단독 | 0.082 |
| **iWSP + Whisper 결합 리지** | **0.087** |

- iWSP 단독은 Whisper를 넘지 못했지만, 교과서적 특징보다는 훨씬 좋다.
- 두 특징을 하나의 리지 모델에 넣으면 Whisper 단독보다 좋아진다. 즉 **iWSP 특징이 Whisper에 없는 정보를 담고 있다.**
- 전체 166개 전극 기준으로 보면 결합 모델이 평균 상관을 **0.070 → 0.075** 로 올렸다. 향상폭은 0.0052 (95% bootstrap CI 0.0039–0.0064). **166개 중 126개 전극은 좋아졌고 40개는 나빠졌다.** (위 표는 저자 README의 수치이고, 166전극 수치는 alphaXiv 요약 기준이다. 저장소 보고서 표를 보면 두 수치는 사용한 전극 집합이 다른 것으로 보인다.)
- 텍스트 병목: 모델을 **197단어**로 설명한 뒤 재구성했을 때, 환자 6명에 걸쳐 원본 성능과 거의 차이가 없었다(alphaXiv 요약 기준 **0.003 이내**).

![ECoG 6명 환자 검색 이력](https://raw.githubusercontent.com/csinva/weight-space-programming-autoresearch/4f8eb61580fe9aa307c868d0bdd86914e673ef1f/experiments/ecog/LEAN_newtest_package/multisubject/summary/ms_fig1_progression.png)

*저자 저장소 `ecog/LEAN_newtest_package/multisubject` 그림. LEAN 검색 이력을 환자 6명에게 다시 적용하고, 각자 held-out 세션으로 채점했다(점: 각 시도, 검정선: patient1에서 검색이 채택한 모델, 점선: Whisper). patient1에서 설계한 모델이 다른 환자에게도 비슷한 궤적으로 일반화되지만, 모든 환자에서 Whisper 단독보다 약간 낮다.*

![LEAN vs Whisper 분산 분할](https://raw.githubusercontent.com/csinva/weight-space-programming-autoresearch/4f8eb61580fe9aa307c868d0bdd86914e673ef1f/experiments/ecog/LEAN_newtest_package/multisubject/summary/ms_fig3_lean_whisper.png)

*저자 저장소 그림. **(a)** 환자별 LEAN-v20, Whisper, 둘을 결합한 리지의 held-out 상관. **(b)** LEAN과 Whisper 사이의 설명 분산 분할. 대부분은 공유 분산이고, LEAN 고유 분산은 환자에 따라 −0.1%~17%다.*

![교과서 특징 vs 검색 결과](https://raw.githubusercontent.com/csinva/weight-space-programming-autoresearch/4f8eb61580fe9aa307c868d0bdd86914e673ef1f/experiments/ecog/LEAN_followups/3_textbook_baseline/fig_textbook_vs_search.png)

*저자 저장소 그림. 환자별 비교: 교과서적 특징 공간을 한 번에 만든 에이전트 5개(회색 점, 막대: 평균), 검색 중간 시점(trial 52), 최종 LEAN-v20(파랑 사각형), Whisper(주황 마름모). 검색으로 얻은 모델이 교과서적 특징보다 일관되게 높다.*

![ECoG 텍스트 병목 테스트](https://raw.githubusercontent.com/csinva/weight-space-programming-autoresearch/4f8eb61580fe9aa307c868d0bdd86914e673ef1f/experiments/ecog/LEAN_followups/2_text_bottleneck/fig_bottleneck.png)

*저자 저장소 그림. **(a)** 197단어 설명으로 재구성한 모델(파랑), 설명 없이 만든 대조군(회색), 원본 LEAN-v20(가로선)의 환자별 held-out 성능. **(b)** 원본 모델 예측과의 상관(patient1). 재구성 모델의 예측이 대조군보다 원본에 훨씬 가깝다.*

### 3.5 저장소에 포함된 부가 실험

- **실제 모델 가중치 읽기**: Claude Opus 5.5가 실제 모델의 가중치를 읽고 편집 효과를 예측할 수 있는지 420건 이상의 편집으로 시험했는데, 대체로 예측하지 못했다. 학습으로 *얻어진* 가중치는 LLM이 읽어도 해석하기 어렵다는 결과로, 처음부터 읽을 수 있게 *작성*하는 iWSP의 동기를 뒷받침한다.
- **Mona Lisa 가중치**: 가중치를 층 순서대로 타일처럼 펼치면 모나리자 그림이 되면서도, 학습 없이 MNIST를 96.17% 정확도로 읽는 16층 CNN이다(가중치 4.0M개 기준, 64M개에서는 97.48%, 학습된 선형 분류기는 92.55%).

---

## 4. 한계 (저자 언급 및 결과에서 드러난 점)

- **피험자 수가 적다**: 언어 fMRI는 1명(UTS03) 중심이고, ECoG 검색도 1명의 세션에서 이루어졌다.
- **관찰 데이터**: 자연 자극에 대한 관찰 데이터이므로 인과적 결론을 내릴 수 없다.
- **기여도 ≠ 인과**: 시각 영역별 계산 비중은 리드아웃 배분 비율일 뿐, 그 계산이 필수라는 증거가 아니다.
- **성능 상한**: 복측 시각피질에서는 AlexNet, ECoG에서는 Whisper에 아직 못 미친다.
- **100% 정확도의 의미**: hidden 예제에서의 100%이지, 모든 입력에 대한 증명이 아니다.
- **규칙 준수는 사후 감사에 의존**: 하네스가 에이전트 코드 안의 학습을 원천 차단하지 못한다.
- **NLP의 한계**: 패러프레이즈 판별처럼 의미를 깊이 이해해야 하는 과제에서는 실패했다.

---

## 5. 정리 및 시사점

1. **"학습" 대신 "작성"**: LLM 에이전트가 사전 지식과 이전 시도의 피드백만으로, 학습 없이도 실제로 작동하는 모델(트랜스포머, 뇌 인코딩 모델)을 만들 수 있다.
2. **해석 가능성을 측정 가능한 형태로**: "200단어 설명으로 재구성 가능한가"라는 운영적 기준이 해석 가능성을 정량화하는 실용적 방법이 된다.
3. **과학적 발견 도구**: 언어 fMRI 사례처럼, 읽을 수 있는 모델은 성능 향상의 *이유*를 보여준다. 덕분에 데이터 아티팩트를 잡아내고 실험 설계 권고(약 40 TR 트리밍)로 이어졌다.
4. **해석 가능성과 성능 사이의 간극을 좁힘**: 시각 피질에서는 기존 해석 가능 모델을 모두 넘었고, ECoG에서는 딥 모델이 놓치는 정보를 보완했다.
5. 저자들은 iWSP를 **예측력 높은 과학 모델과 사람이 이해할 수 있는 과학 이론을 정렬하는 방법**으로 제시한다.

---

## 참고 링크

- 논문: https://arxiv.org/abs/2611.18114
- alphaXiv 페이지: https://www.alphaxiv.org/abs/2611.18114
- 코드 및 실행 로그: https://github.com/csinva/weight-space-programming-autoresearch

<sub>작성 메모: arXiv HTML 본문은 요청 한도 제한으로 직접 열람하지 못했다. 그래서 alphaXiv의 논문 요약과 저자 공식 GitHub 저장소(README, 실험 보고서, 그림 파일)를 바탕으로 정리했다. Figure 1은 논문 Figure 1과 같은 저자 원본이고, 나머지 그림은 저장소의 분석 그림이라 논문에 실린 최종 버전과 세부가 다를 수 있다. 이미지는 저자 공개 저장소의 특정 커밋(`4f8eb61`)에 고정된 원본 파일 링크로 표시된다.</sub>
