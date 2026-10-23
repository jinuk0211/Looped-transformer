# LLM·VLM 해석과 제어: 표현 읽기, 회로 추적, Activation Steering
===========
<img width="680" height="357" alt="image" src="https://github.com/user-attachments/assets/839478b1-a272-497f-b4a2-606e713063bd" />

이 문서는 관련 논문과 공개 구현을 **표현 읽기 → 계산 경로 추적 → 내부 상태 개입 → 행동 평가**라는 공통 문제로 다시 엮은 긴 학습·연구 노트다. 논문별 소개에 그치지 않고, 수식이 가리키는 대상, 비슷한 방법들의 차이, 공개 코드와 설명 사이의 간극, 직접 실험할 때의 설계까지 연결했다.

LatentLens·J-lens·Circuit Tracing과 여러 activation steering 방법을 공통 질문으로 연결한다. 핵심 주장은 논문·모델 카드·공개 코드와 대조했고, 각 절에 출처를 연결했다. 코드 점검은 정적 검토이며 GPU에서 모델을 실행해 논문 성능을 재현한 결과는 아니다.

문서에서 **원문 요약**은 자료가 제시한 내용, **종합 해석**은 자료 사이의 연결, **실험 제안**은 이 문서에서 제안하는 검증 방법을 뜻한다. 예시 수치와 작은 벡터 계산은 이해를 위한 가상 사례다.

**그림 안내:** 논문과 공식 연구 페이지의 원본 그림 44개를 수록했다. 다중 패널을 포함한 이미지 파일은 48개이며, 같은 원문 figure의 패널은 하나의 그림으로 센다. 이미지의 선·색·수치는 수정하지 않았고, 한국어 제목과 읽는 포인트를 새로 붙였다. 각 캡션에서 원문의 figure 또는 도판 위치로 이동할 수 있다. 그림의 저작권과 이용 조건은 각 원출처에 따른다. 외부 원본 링크를 사용하므로 원출처 주소가 바뀌면 표시가 영향을 받을 수 있다.

## 목차

- [1. 이 자료들은 어떤 큰 질문으로 연결되는가](#sec-01)
- [2. 주요 자료와 각각의 역할](#sec-02)
- [3. 공통 기초: 우리가 바꾸는 ‘활성화’는 정확히 무엇인가](#sec-03)
- [4. 해석의 증거에는 단계가 있다](#sec-04)
- [5. LatentLens: 벡터를 문맥이 있는 텍스트로 읽기](#sec-05)
- [6. J-lens: 표현이 말로 이어지는 평균적인 방향 읽기](#sec-06)
- [7. LatentLens와 J-lens를 함께 쓰면 무엇이 좋은가](#sec-07)
- [8. 뉴런, SAE, transcoder, CLT를 구분하기](#sec-08)
- [9. Circuit Tracing: 의미 목록에서 계산 그래프로](#sec-09)
- [10. VLM Circuit Tracing: 이미지가 답변으로 이어지는 길](#sec-10)
- [11. Steering 방법을 비교하는 공통 좌표](#sec-11)
- [12. CAA와 ITI: 평균 방향과 선택한 head](#sec-12)
- [13. SADI: 고정된 마스크, 상태에 따라 달라지는 변화량](#sec-13)
- [14. MERA: 예측 오류를 줄이는 최소 변화와 보정](#sec-14)
- [15. AUSteer: 큰 상태 전체보다 작은 activation unit](#sec-15)
- [16. VLM의 hallucination을 하나의 원인으로 묶지 않기](#sec-16)
- [17. VTI: 시각 경로와 언어 경로를 함께 조정하기](#sec-17)
- [18. REVIS: 고정 방향에서 prior 성분을 줄이고 필요할 때 개입](#sec-18)
- [19. RUDDER: 현재 입력에서 anchor를 만들고 상태에 맞춰 주입](#sec-19)
- [20. GrAInS: 중요한 토큰을 찾은 뒤 방향 만들기](#sec-20)
- [21. 아홉 steering 계열을 한 표에서 비교하기](#sec-21)
- [22. TraitBasis: 제어를 평가 환경을 만드는 데 사용하기](#sec-22)
- [23. 모든 방법을 연결하는 하나의 설계식](#sec-23)
- [24. 수식을 실제로 이해하기 위한 여섯 가지 작은 계산](#sec-24)
- [25. 공개 코드를 대조하면서 확인한 점](#sec-25)
- [26. 구현할 때 특히 자주 섞이는 개념](#sec-26)
- [27. 개입 연산을 분리해 생각하는 작은 코드 예시](#sec-27)
- [28. 제대로 비교하려면 무엇을 고정해야 하는가](#sec-28)
- [29. 이 자료들을 연결해 볼 만한 연구 질문](#sec-29)
- [30. 직접 공부하거나 재현할 때의 진행 순서](#sec-30)
- [31. 용어를 다시 찾기 위한 사전](#sec-31)
- [32. 자료를 다시 읽을 때 남겨 둘 질문](#sec-32)

<a id="sec-01"></a>

## 1. 이 자료들은 어떤 큰 질문으로 연결되는가

자료 전체를 관통하는 질문은 다음과 같다.

> 모델 내부에는 어떤 의미가 표현되어 있고, 그중 무엇이 실제 답변을 만들며, 어느 부분을 얼마나 바꾸면 원하는 행동을 얻을 수 있는가?

이 질문에는 서로 다른 네 문제가 들어 있다. 첫째는 **읽기**다. 이미지 패치나 토큰의 벡터가 어떤 의미와 연결되는지 사람이 이해할 수 있는 말로 바꾸는 작업이다. LatentLens와 J-lens가 여기에 가깝다. 둘째는 **추적**이다. 특정 출력에 도달하는 계산에서 어떤 내부 요소가 어떤 요소에 영향을 주었는지 연결한다. Circuit Tracing과 VLM 확장이 여기에 해당한다. 셋째는 **제어**다. 표현을 읽는 데 그치지 않고 벡터나 특정 좌표를 바꾸어 행동을 움직인다. CAA부터 GrAInS까지의 steering 방법들이 여기에 들어간다. 넷째는 **평가**다. 바뀐 행동이 정말 개선인지, 어떤 부작용이 생기는지, 조건이 달라도 유지되는지 살핀다. TraitBasis는 제어를 평가 환경을 만드는 데 활용한다.

이 네 문제는 연결되어 있지만 자동으로 해결되지는 않는다. 벡터에 그럴듯한 이름을 붙였다고 그 벡터가 출력의 원인이라는 뜻은 아니다. 어떤 방향을 더해서 정답률이 올랐다고 그 방향이 순수한 ‘진실성’ 개념이라는 뜻도 아니다. 아주 좋은 회로 설명을 얻어도 그 회로에 큰 개입을 하면 설명을 만들 때의 근사가 무너질 수 있다.

따라서 이 자료들을 함께 읽을 때 가장 생산적인 태도는 방법마다 **무엇을 관측했고, 무엇을 조작했고, 무엇을 평가했는지**를 분리하는 것이다. 이 구분만 유지해도 여러 논문이 같은 말을 하는 것처럼 보이는 혼란이 크게 줄어든다.

<a id="sec-02"></a>

## 2. 주요 자료와 각각의 역할

| 번호 | 자료 | 이 정리에서의 역할 |
|---|---|---|
| 1 | [jlens-qwen3.5-27b 모델 카드](https://huggingface.co/kieraisverybored/jlens-qwen3.5-27b) | 이미지 토큰 표현을 어휘로 읽는 실제 배포 사례 |
| 2 | [Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) | CLT, replacement model, attribution graph의 방법론 |
| 3 | [LatentLens · v5](https://arxiv.org/html/2602.00462v5) | 시각 표현을 문맥 속 텍스트 표현과 대조해 해석 |
| 4 | [Circuit Tracing in Vision–Language Models · v1](https://arxiv.org/html/2602.20330v1) | 멀티모달 계산 경로 추적과 개입 |
| 5 | [CAA](https://aclanthology.org/2024.acl-long.828/) · [ITI](https://proceedings.neurips.cc/paper_files/paper/2023/hash/81b8390039b7302c909cb769f8b6cd93-Abstract-Conference.html) · [MERA](https://proceedings.mlr.press/v267/hedstrom25a.html) · SADI · AUSteer | 일반적인 activation steering의 비교 축 |
| 6 | [VTI](https://github.com/shengliu66/VTI) · [REVIS](https://github.com/antgroup/Revis) · RUDDER · GrAInS 공개 구현 | 수식에서 구현으로 넘어갈 때의 의문 |
| 7 | [VTI](https://arxiv.org/html/2410.15778v2) · [REVIS](https://arxiv.org/html/2602.11824) · [RUDDER](https://arxiv.org/html/2511.10292v2) · [GrAInS](https://arxiv.org/html/2507.18043v2) | VLM hallucination과 steering의 연결 |
| 8 | [Impatient Users Confuse AI Agents · TraitBasis · v3](https://arxiv.org/html/2510.04491v3) | 사용자 시뮬레이터를 조절하는 평가 응용 |
| 9 | [Goodfire: Already Clean](https://already-clean.goodfire.com/) | 원래 뉴런 자체의 해석 가능성을 다시 묻는 관점 |

위 표는 공개 출처만 연결한다. 개별 방법의 세부 논문과 코드 링크는 해당 절에도 배치했다. 아래의 비교는 이 자료 묶음에서 드러난 연구 흐름을 정리한 것이며, 전체 분야를 빠짐없이 조사한 문헌 리뷰나 최신 성능 순위표는 아니다.

<a id="sec-03"></a>

## 3. 공통 기초: 우리가 바꾸는 ‘활성화’는 정확히 무엇인가

모델의 파라미터는 학습을 통해 저장된 가중치다. 활성화는 그 가중치에 특정 입력을 넣었을 때 생기는 중간 계산값이다. 같은 모델도 입력이 달라지면 활성화가 달라진다. Activation steering은 보통 가중치를 다시 학습시키기보다 실행 중간의 활성화에 개입한다.

언어 모델의 중간 상태를 `h[l,t]`라고 쓰자. `l`은 레이어, `t`는 토큰 위치다. 실제 배치 텐서는 흔히 `[B,T,D]` 모양이다. `B`는 한 번에 처리하는 입력 수, `T`는 시퀀스 길이, `D`는 숨은 차원이다. ‘벡터를 더한다’는 짧은 설명에도 어느 레이어, 어느 토큰, 어느 연산 전후인지가 모두 포함되어야 한다.

개념적으로 transformer 블록을 다음처럼 생각할 수 있다. 실제 모델의 normalization 위치나 세부 구조는 다를 수 있다.

```text
u = h + Attention(Norm(h))
h_next = u + MLP(Norm(u))
```

여기서 attention의 출력, MLP의 출력, 블록 전체의 출력은 다르다. MLP 출력에 `v`를 더하면 residual에 합쳐지는 업데이트를 수정한다. 블록 출력에 더하면 attention과 MLP 계산을 거친 전체 상태를 수정한다. 수식이 둘 다 `h + αv`처럼 보이더라도, 코드에서의 의미와 부작용은 달라질 수 있다.

VLM에서는 이미지가 먼저 시각 표현으로 변환되고, 모델 구조에 맞게 언어 모델과 연결된다. 이후에는 이미지에서 유래한 위치, 질문 텍스트 위치, 생성된 답변 위치가 서로 다른 역할을 한다. ‘이미지의 의미가 들어 있는 벡터’와 ‘답변을 생성하는 위치의 벡터’도 같은 대상이 아니다. 이미지 토큰을 해석하는 lens와 답변 토큰을 조절하는 steering을 연결하려면, 그 사이에 정보가 어떻게 전달되는지를 추가로 확인해야 한다.

또 하나의 구분은 **prefill과 decode**다. Prefill은 입력 문맥을 처리하는 단계이고, decode는 생성된 토큰을 추가하면서 다음 토큰을 예측하는 단계다. 첫 생성 토큰은 보통 prefill의 마지막 유효 위치에서 나온 logits로 선택된다. 따라서 ‘답변 토큰에서만 개입한다’는 마스크를 무심코 구현하면 첫 생성 토큰에는 개입이 적용되지 않을 수 있다.

<a id="sec-04"></a>

## 4. 해석의 증거에는 단계가 있다

이 절은 여러 자료를 읽기 위한 종합적인 판단 틀이다. 아래 단계는 논문의 공식 등급 체계가 아니라, 주장과 증거를 혼동하지 않기 위한 기준이다.

| 관측 | 말할 수 있는 것 | 아직 말할 수 없는 것 |
|---|---|---|
| 가까운 단어·문장이 검색된다 | 표현이 해당 텍스트 표현과 유사하다 | 그 단어가 모델 내부의 유일한 의미다 |
| 특정 사례에서 활성화가 높다 | 해당 특징과 통계적으로 연결된다 | 출력에 반드시 필요한 요소다 |
| 출력에 대한 미분값이 크다 | 현재 상태 근처에서 출력이 민감하다 | 큰 폭의 조작에서도 같은 효과가 난다 |
| ablation·patching으로 출력이 바뀐다 | 시험한 개입이 출력에 인과적 영향을 준다 | 다른 프롬프트에서도 같은 회로가 작동한다 |
| 독립 데이터에서 반복 개선된다 | 정해진 조건에서 유용성이 재현된다 | 모든 모델·분포·과제에 보편적으로 통한다 |

예를 들어 고양이 사진에서 어떤 뉴런이 크게 켜지고, lens가 ‘cat’을 보여주었다고 하자. 이것은 좋은 해석 후보지만, 그 뉴런을 끄고도 정답이 그대로 나올 수 있다. 다른 경로가 같은 정보를 전달하기 때문일 수도 있고, 그 뉴런은 배경 정보만 나타냈기 때문일 수도 있다. 반대로 뉴런을 껐더니 출력이 망가졌다고 해서 곧바로 ‘고양이 뉴런’이 증명되는 것도 아니다. 개입이 일반적인 문장 생성 능력까지 손상했을 수 있다.

따라서 설명을 강화하려면 **특이성**이 필요하다. 고양이 관련 판단에는 영향을 주되 무관한 산술·문법·다른 객체 판단에는 영향을 덜 주는지 살펴야 한다. 이 관점은 lens, neuron 분석, 회로 추적, steering 모두에 적용된다.

<a id="sec-05"></a>

## 5. LatentLens: 벡터를 문맥이 있는 텍스트로 읽기

**원문 요약.** LatentLens는 시각 hidden state를 텍스트 코퍼스에서 얻은 문맥화된 토큰 표현과 비교한다. 가까운 표현을 찾은 뒤 그 토큰이 포함된 문장을 보여준다. 입력 임베딩 테이블이나 최종 어휘 출력 행렬만 이용하는 방식과 달리, 문맥과 레이어를 가진 텍스트 표현을 해석의 기준으로 삼는다. 시각 표현과 비교하는 텍스트 표현의 레이어가 반드시 같을 필요도 없다. 논문은 초기 시각 표현이 중간 이후의 텍스트 표현과 의미적으로 잘 연결되는 현상을 Mid-Layer Leap으로 분석한다. 별도의 lens 학습이 필요 없다는 것은 코퍼스 표현의 계산·저장 비용까지 없다는 뜻은 아니다. 논문의 interpretability 평가는 생성 정답률이나 인과적 충실도를 직접 측정한 값으로 읽어서는 안 된다. [LatentLens v5](https://arxiv.org/html/2602.00462v5)

논문은 15개 VLM에 대한 평균 interpretability를 LatentLens 68%, EmbeddingLens 32%, LogitLens 24%로 보고한다. 이는 정해진 검색 결과와 주변 토큰을 함께 본 평가 절차의 수치이며, ‘시각 문제의 68%를 맞혔다’는 뜻이 아니다. 평가자와 참조 코퍼스가 달라지면 결과의 해석도 달라진다.

이를 개념적으로 쓰면 다음과 같다.

```text
q = 해석하려는 시각 hidden state
H = 코퍼스의 문맥화된 텍스트 토큰 표현 모음
검색 결과 = cosine(q, H[j])가 큰 토큰과 그 원문 문맥
```

**종합 해석.** 이 접근의 강점은 ‘bank’처럼 문맥에 따라 의미가 달라지는 단어를 고립된 어휘 항목으로 다루지 않는다는 데 있다. 하지만 검색 결과의 질은 참조 코퍼스에도 좌우된다. 코퍼스에 바닷가 장면 설명이 많고 실험실 장면 설명이 드물다면, 유사한 벡터를 설명하는 문장 후보 자체가 한쪽으로 치우칠 수 있다. 결과를 읽을 때는 한 문장만 고르기보다 여러 이웃이 공유하는 의미와 서로 충돌하는 의미를 함께 보는 편이 낫다.


### 원본 그림으로 확인하기

**그림 1. 시각 토큰을 문맥 속 단어와 비교하는 이유**

![LatentLens — 시각 토큰을 문맥 속 단어와 비교하는 이유](https://arxiv.org/html/2602.00462v5/figure1.svg)

*원본: [LatentLens, Figure 1](https://arxiv.org/html/2602.00462v5#S1.F1).*

낱말 하나의 입력 임베딩 대신 문장 안에서 계산한 표현을 검색한다. 이미지 패치의 의미와 검색된 단어의 문맥을 함께 읽는다.

**그림 2. LatentLens의 전체 분석 절차**

![LatentLens — LatentLens의 전체 분석 절차](https://arxiv.org/html/2602.00462v5/figure2_v2.svg)

*원본: [LatentLens, Figure 2](https://arxiv.org/html/2602.00462v5#S3.F2).*

텍스트 표현의 사전 계산, 시각 표현 추출, 최근접 이웃 검색, VLM 판정 단계가 연결된다. 마지막 판정 점수는 검색 결과의 해석 가능성에 대한 평가다.

**그림 3. 세 가지 lens의 층별 해석 가능성**

![LatentLens — 세 가지 lens의 층별 해석 가능성](https://arxiv.org/html/2602.00462v5/fig1_unified_interpretability.svg)

*원본: [LatentLens, Figure 3](https://arxiv.org/html/2602.00462v5#S4.F3).*

각 곡선은 논문이 판정한 해석 가능한 시각 토큰의 비율이다. 후반층의 logit lens 개선과 LatentLens의 전 층 경향을 구분해서 본다.

**그림 4. 초기 시각 표현이 어느 텍스트 층과 정렬되는가**

![LatentLens — 초기 시각 표현이 어느 텍스트 층과 정렬되는가](https://arxiv.org/html/2602.00462v5/layers_heatmap_combined_3x3.png)

*원본: [LatentLens, Figure 4](https://arxiv.org/html/2602.00462v5#S4.F4).*

행과 열은 서로 다른 층의 비교다. 입력 단계의 시각 표현도 중간 텍스트 층과 가까울 수 있다는 관찰을 보여 주며, 인과 경로 자체를 나타내지는 않는다.

**그림 5. 이미 학습된 여섯 VLM에서의 비교**

![LatentLens — 이미 학습된 여섯 VLM에서의 비교](https://arxiv.org/html/2602.00462v5/fig_offtheshelf.svg)

*원본: [LatentLens, Figure 5](https://arxiv.org/html/2602.00462v5#S4.F5).*

정규화된 층 깊이를 사용한 실제 모델 비교다. 서로 다른 구조에서 나타나는 공통 경향과 모델별 차이를 함께 확인한다.

---

<a id="sec-06"></a>

## 6. J-lens: 표현이 말로 이어지는 평균적인 방향 읽기

**원문 요약.** Jacobian lens는 중간 표현이 뒤쪽 표현에 미치는 미분 관계를 여러 문맥·위치에 걸쳐 평균낸 변환을 사용한다. 이를 통해 원래 어휘 출력 장치와 연결해서 읽을 수 있는 성분을 살핀다. 원 논문은 이런 성분과 verbalizable representation, global workspace의 관계를 연구한다. 이것은 곧바로 의식이나 모든 내부 생각을 읽는다는 주장은 아니다. [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/)

위 Hugging Face 저장소는 이 아이디어를 Qwen3.5-27B의 이미지 토큰에 적용한 커뮤니티 배포물이다. 기본 모델 전체를 새로 학습한 체크포인트가 아니라 fitted lens와 적용 코드가 중심이다. 모델 카드에는 MMDU-45k의 단일 이미지 사례 30개로 fitting한 설정, 중간 레이어에서 마지막 레이어로의 변환, 사용자가 시도할 레이어 등이 제시되어 있다. 제시된 권장 레이어와 결과는 이 배포물의 경험적 설정이다. 모든 입력·모델에 최적이라는 뜻은 아니다. 마지막 레이어의 이미지 토큰에서 같은 레이어의 뒤쪽 텍스트 토큰으로 이어지는 Jacobian은 더 이상의 토큰 간 계산이 없어 퇴화할 수 있다. [jlens-qwen3.5-27b](https://huggingface.co/kieraisverybored/jlens-qwen3.5-27b)

실제로 찾아볼 파일은 약 1.2GB의 `jimage_final_lens.pt`, 적용 예제 `apply_example.py`, 멀티모달 보조 코드 `mm.py`다. 카드의 fitting 범위는 레이어 16부터 62까지 2 간격이고 목표 레이어는 62다. 기본 읽기 레이어 46과 일부 좋은 결과가 보고된 50은 시작점으로 볼 수 있다. 모델 가중치와 lens 배포물의 이용 조건은 각각의 카드를 확인해야 한다.

```text
J[l] ≈ E[ ∂h[final, target_position] / ∂h[l, source_position] ]
읽기 ≈ vocabulary_head( final_norm( J[l] @ h[l, source_position] ) )
```

**종합 해석.** 평균 Jacobian은 특정 입력 하나의 전체 인과 구조를 그대로 보존하지 않는다. 평균을 내면서 널리 공유되는 변환은 강조되고 문맥에 특수한 효과는 약해질 수 있다. 따라서 J-lens가 보여주는 어휘는 ‘이 표현을 모델의 언어 출력 공간에서 보면 무엇과 연결되는가’의 단서다. 특정 정답이 왜 나왔는지를 입증하려면 해당 입력의 개입 실험을 더해야 한다.

이미지에 읽기 결과를 겹쳐 표시할 때는 원본 픽셀, resize된 이미지, patch grid, merge된 토큰의 좌표를 구분해야 한다. 잘못 정렬된 heatmap은 언어 해석 자체가 맞더라도 엉뚱한 물체를 가리킬 수 있다.


### 원본 그림으로 확인하기

**그림 6. 평균 Jacobian으로 읽고 좌표를 바꾸는 J-lens**

![J-lens · Verbalizable Representations Form a Global Workspace in Language Models — 평균 Jacobian으로 읽고 좌표를 바꾸는 J-lens](https://transformer-circuits.pub/2026/workspace/png/img_1b62b10ab235e6e7.png)

*원본: [J-lens · Verbalizable Representations Form a Global Workspace in Language Models, Figure 4](https://transformer-circuits.pub/2026/workspace/#fig-jlens-schematic).*

원문 Figure 4의 읽기와 개입 절차다. 평균 Jacobian을 통과한 어휘 읽기와 lens 좌표의 교환은 서로 다른 실험 단계다.

**그림 7. 중간 개념 복원 능력의 비교**

![J-lens · Verbalizable Representations Form a Global Workspace in Language Models — 중간 개념 복원 능력의 비교](https://transformer-circuits.pub/2026/workspace/png/img_c983850908bf60d9.png)

*원본: [J-lens · Verbalizable Representations Form a Global Workspace in Language Models, Figure 52](https://transformer-circuits.pub/2026/workspace/#fig-method-compare-passk).*

여섯 프롬프트 분포에서 측정한 정규화 pass@k AUC다. 개념을 잘 읽는다는 결과와 개입이 출력을 바꾼다는 결과를 분리해 평가한다.

---

<a id="sec-07"></a>

## 7. LatentLens와 J-lens를 함께 쓰면 무엇이 좋은가

두 방법을 동일한 점수로 줄 세우기보다 서로 다른 질문을 던지는 도구로 보면 유용하다.

| 질문 | LatentLens 쪽 관점 | J-lens 쪽 관점 |
|---|---|---|
| 무엇과 닮았는가 | 코퍼스의 문맥화된 텍스트 표현 | 뒤쪽 표현·어휘 출력과 연결되는 성분 |
| 설명의 형태 | 이웃 토큰과 문장 | 어휘 공간의 읽기 결과 |
| 주요 의존성 | 참조 코퍼스, 표현 레이어, 유사도 | fitting 분포, 평균 Jacobian, 출력 head |
| 검증할 지점 | 이웃의 일관성과 코퍼스 편향 | 문맥 변화에 따른 읽기의 안정성 |

**실험 제안.** 같은 이미지 토큰에 대해 두 결과를 나란히 기록한다. 둘 다 특정 객체를 가리킨다면 좋은 후속 조사 후보가 된다. 결과가 다르다면 실패라고 즉시 판단하지 않는다. 하나는 장면의 문맥적 유사성을, 다른 하나는 언어로 출력되기 쉬운 성분을 보여주었을 수 있다. 그 차이를 토큰 위치, 레이어, 질문 변경, 이미지 교체로 분해한다.

예를 들어 같은 자전거 사진에 ‘무슨 물체가 있나’와 ‘안전장비가 보이나’를 각각 물었을 때 읽기가 바뀌는지 본다. 이미지 내용이 같아도 언어 문맥에 의해 관련 의미의 강조가 달라질 수 있다. 이때 중요한 결과는 ‘정답 단어를 맞혔다’ 하나가 아니라, **어느 표현이 이미지에 안정적으로 연결되고 어느 표현이 질문에 따라 달라지는가**다.


### 원본 그림으로 확인하기

**그림 8. 좌표 교환 뒤 실제 출력이 뒤집히는 비율**

![J-lens · Verbalizable Representations Form a Global Workspace in Language Models — 좌표 교환 뒤 실제 출력이 뒤집히는 비율](https://transformer-circuits.pub/2026/workspace/png/img_151fa39f2f7df490.png)

*원본: [J-lens · Verbalizable Representations Form a Global Workspace in Language Models, Figure 54](https://transformer-circuits.pub/2026/workspace/#fig-method-compare-swap).*

세 모델 규모에서 lens 좌표를 바꾼 뒤의 top-1 답변 전환을 비교한다. 앞의 복원 지표보다 인과 개입에 가까운 증거로 읽을 수 있다.

---

<a id="sec-08"></a>

## 8. 뉴런, SAE, transcoder, CLT를 구분하기

| 대상 | 분석 단위 | 핵심 목적 |
|---|---|---|
| 원래 뉴런·좌표 | 모델이 실제로 가진 활성화 축 | 추가 표현 학습 없이 구조 관찰 |
| SAE | 활성화를 희소 코드로 바꾸는 학습된 특징 | 표현을 해석하기 쉬운 요소로 분해 |
| Transcoder | MLP 입력으로부터 출력 기여를 예측하는 특징 | MLP 계산을 추적 가능한 형태로 근사 |
| CLT | 레이어를 가로질러 기여할 수 있는 특징 | 여러 층의 계산을 연결하는 근사 |

**원문 요약.** Goodfire의 Already Clean은 원래 뉴런 자체에 이해 가능한 구조가 상당히 많이 있을 수 있다는 작업 가설을 제시한다. 과거의 다의적인 뉴런 사례 일부가 실제로는 관련된 일반화이거나 분석 방법·아키텍처와 관련된 현상일 수 있다는 관점이다. 탐색적 관찰과 사례를 통해 기존 관행에 문제를 제기하는 글로 읽는 것이 적절하다. 모든 모델에서 superposition이 사라졌다는 확정된 결론이나 SAE가 항상 불필요하다는 증명으로 확대하면 안 된다. [Already Clean](https://already-clean.goodfire.com/)

**종합 해석.** 여기서 얻을 실용적인 교훈은 양쪽 접근을 공정하게 비교하라는 것이다. 원래 뉴런의 설명이 좋다면 그것을 기준선으로 포함한다. 학습된 특징이 더 좋은 설명을 준다면 재구성 오차와 학습 비용도 함께 기록한다. 사람이 붙인 이름이 예쁜지보다, 그 이름이 새로운 예제의 활성화를 예측하는지, 개입 결과를 설명하는지, 여러 문맥에서 유지되는지가 중요하다.

‘희소하다’라는 말에도 주의해야 한다. 적은 좌표만 수정하는 것, 적은 토큰에서만 개입하는 것, 적은 레이어만 선택하는 것, 학습된 sparse feature를 사용하는 것은 서로 다르다. REVIS의 sparse intervention을 SAE 기반 방법이라고 읽거나, AUSteer의 AU를 SAE feature와 동일시하면 설계 자체를 오해하게 된다.


### 원본 그림으로 확인하기

**그림 9. SAE와 transcoder가 복원하는 대상의 차이**

![Circuit Tracing in Vision–Language Models — SAE와 transcoder가 복원하는 대상의 차이](https://arxiv.org/html/2602.20330v1/figures/sae_vs_transcoder.png)

*원본: [Circuit Tracing in Vision–Language Models, Figure 3](https://arxiv.org/html/2602.20330v1#S3.F3).*

SAE는 활성화 자체를, transcoder는 MLP의 입력–출력 동작을 근사한다. 두 도구를 같은 종류의 복원기로 취급하지 않도록 원문 그림을 나란히 본다.

**그림 10. 낯선 이미지에 동원되는 기존 뉴런**

![Goodfire · Neural Networks Are Already Clean — 낯선 이미지에 동원되는 기존 뉴런](https://already-clean.goodfire.com/assets/september-revision/rare-image-mini/rare-images-neurons@3x.webp)

*원본: [Goodfire · Neural Networks Are Already Clean, rare-image-examples 원본 도판](https://already-clean.goodfire.com/#rare-image-examples).*

위쪽 활성화 지도와 아래쪽 지지 뉴런의 자연 이미지 예시를 연결한다. 개별 사례가 낯설다는 사실만으로 새로운 전용 특징이 필요하다고 결론내릴 수는 없다.

**그림 11. ConvNeXt의 층과 뉴런군을 잇는 전체 지도**

![Goodfire · Neural Networks Are Already Clean — ConvNeXt의 층과 뉴런군을 잇는 전체 지도 · 패널 1](https://already-clean.goodfire.com/assets/convnext-map-still/whole.webp)

![Goodfire · Neural Networks Are Already Clean — ConvNeXt의 층과 뉴런군을 잇는 전체 지도 · 패널 2](https://already-clean.goodfire.com/assets/convnext-map-still/zoom.webp)

*원본: [Goodfire · Neural Networks Are Already Clean, convnext-map-still · 전체와 확대](https://already-clean.goodfire.com/#convnext-map-still).*

원본 인터페이스의 전체 지도와 확대 화면이다. 군집의 이름은 탐색을 돕는 해석이며, 그 자체가 모든 뉴런의 단일 의미를 보증하지는 않는다.

**그림 12. SwiGLU 출력의 부호를 분해해서 보기**

![Goodfire · Neural Networks Are Already Clean — SwiGLU 출력의 부호를 분해해서 보기](https://already-clean.goodfire.com/assets/swiglu-audit.png)

*원본: [Goodfire · Neural Networks Are Already Clean, swiglu-audit 원본 도판](https://already-clean.goodfire.com/#swiglu-audit).*

게이트와 값 성분의 부호 조합을 활성화 구간별로 조사한 원본 분석이다. 최종 출력값만 보고 두 내부 성분의 의미를 단정하지 않도록 돕는다.

---

<a id="sec-09"></a>

## 9. Circuit Tracing: 의미 목록에서 계산 그래프로

**원문 요약.** Anthropic의 방법은 cross-layer transcoder로 MLP 계산을 근사한다. 한 레이어에서 읽은 특징이 그 레이어와 이후 레이어의 MLP 출력에 기여할 수 있다. 특정 입력을 설명하는 local replacement model에서는 attention pattern과 normalization의 일부 입력 의존성을 고정하고, 원래 모델과 근사 모델의 차이를 error term으로 보정한다. 이 위에서 특징·입력·출력 사이의 attribution graph를 만든다. 원래 입력에서 출력을 맞추는 것과, 모든 개입에서 원래 모델의 행동을 그대로 재현하는 것은 다르다. 그래프는 고정된 attention 가정, 근사 오차, pruning, 특징 분해의 한계를 가진다. [Circuit Tracing 방법론](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)

```text
원래 MLP 출력 = transcoder가 설명한 부분 + 남은 error
그래프의 연결 = 활성화 크기 × 현재 근사 안에서의 전달 효과
```

**종합 해석.** 회로 그래프를 읽을 때는 노드 이름보다 다음 질문이 먼저다. 어떤 출력 토큰을 설명하는 그래프인가? 어떤 경로가 생략되었는가? 그래프에서 큰 역할을 하는 error node가 있는가? 개입할 때 attention routing도 바뀔 수 있는가? 특징을 묶어서 붙인 이름이 서로 다른 계산을 숨기지는 않는가?

그래프가 복잡하다는 이유만으로 설명이 깊은 것도 아니고, 간단하다는 이유만으로 충실한 것도 아니다. 좋은 설명은 관찰한 출력을 설명하면서, 아직 보지 않은 개입 결과에 대한 예측을 제공해야 한다. 그래프에서 중요한 특징을 줄이면 특정 logit 차이가 감소할 것이라는 예측을 세우고 원래 모델에서 확인하는 식이다.


### 원본 그림으로 확인하기

**그림 13. 원래 모델에서 해석 가능한 특징과 attribution graph로**

![Anthropic · Circuit Tracing: Revealing Computational Graphs in Language Models — 원래 모델에서 해석 가능한 특징과 attribution graph로](https://transformer-circuits.pub/2025/attribution-graphs/png/methods.png)

*원본: [Anthropic · Circuit Tracing: Revealing Computational Graphs in Language Models, 논문 도입부의 방법 개요](https://transformer-circuits.pub/2025/attribution-graphs/methods.html#methods-diagram-svg).*

저자가 공식 페이지에서 제공하는 대표 개요 이미지다. 특징을 통한 대체 모델과 특징 사이의 영향 그래프를 구분하며, 그래프의 가설은 별도의 개입으로 검증해야 한다.

---

<a id="sec-10"></a>

## 10. VLM Circuit Tracing: 이미지가 답변으로 이어지는 길

**원문 요약.** 위 VLM 논문은 Gemma3-4B-IT에서 layer-wise sparse transcoder를 학습해 멀티모달 계산을 추적한다. Anthropic의 cross-layer CLT와 같은 구조라고 보아서는 안 된다. 시각 입력, sparse feature, 출력 사이를 연결하고, 활성 사례와 SigLIP attention rollout 등을 활용해 시각 특징의 위치와 의미를 해석한다. 시각적 단서와 의미적 연상이 어떻게 결합하는지를 사례로 분석하고 일부 경로를 개입으로 검증한다. 관찰된 visual·semantic stream의 패턴은 해당 모델과 실험 사례에서 얻은 결과다. 하나의 보편적 VLM 구조로 단정하기에는 모델 범위, 수작업 해석, 위치 대응, 근사 계산 등의 제한이 있다. [Circuit Tracing in Vision–Language Models](https://arxiv.org/html/2602.20330v1)

**종합 해석.** 이 자료는 lens와 steering 사이의 빈칸을 메우는 방향을 보여준다. 이미지 표현에서 ‘무엇처럼 보이는가’를 읽고, 그 표현이 실제 정답 후보까지 전달되는 경로를 찾고, 그중 일부를 바꾸어 출력이 예상대로 달라지는지 확인한다. 그러나 attention rollout의 위치 표시만으로 특정 픽셀이 정답의 원인임을 증명할 수는 없다.

실험에서는 객체를 가린 이미지, 객체 위치만 바꾼 이미지, 질문만 바꾼 입력을 함께 사용하면 좋다. 시각적으로 연결된 특징이라면 이미지 변화에 반응해야 하고, 질문에 따라 선택되는 의미 경로라면 질문 변화에도 체계적으로 반응해야 한다. 단일 사례에서 발견한 흥미로운 경로를 반복 가능한 기제로 바꾸는 단계다.


### 원본 그림으로 확인하기

**그림 14. 해달 이미지에서 내부 회로까지**

![Circuit Tracing in Vision–Language Models — 해달 이미지에서 내부 회로까지](https://arxiv.org/html/2602.20330v1/figures/main_figure_seaotter.jpg)

*원본: [Circuit Tracing in Vision–Language Models, Figure 1](https://arxiv.org/html/2602.20330v1#S1.F1).*

이미지와 프롬프트에서 특징 및 출력으로 이어지는 연구 개요다. 아래의 구체적인 실험 그림들과 함께 읽되 모든 VLM에 공통인 완성된 회로도로 일반화하지 않는다.

**그림 15. VLM용 transcoder 학습과 회로 추출**

![Circuit Tracing in Vision–Language Models — VLM용 transcoder 학습과 회로 추출](https://arxiv.org/html/2602.20330v1/figures/method.jpg)

*원본: [Circuit Tracing in Vision–Language Models, Figure 2](https://arxiv.org/html/2602.20330v1#S3.F2).*

멀티모달 자료로 학습한 transcoder, 특징 분석, attribution graph와 연구자의 해석이 연결된다. 자동 계산과 사람이 붙인 의미의 경계를 확인한다.

**그림 16. 화성 특징을 지구 특징으로 바꾸는 개입**

![Circuit Tracing in Vision–Language Models — 화성 특징을 지구 특징으로 바꾸는 개입](https://arxiv.org/html/2602.20330v1/figures/mars_intervention.jpg)

*원본: [Circuit Tracing in Vision–Language Models, Figure 6](https://arxiv.org/html/2602.20330v1#S4.F6).*

같은 프롬프트에서 발견한 두 회로의 특징을 사용해 activation patching을 수행한다. 시각적 유사성만 보는 분석보다 개입에 따른 변화가 핵심이다.

**그림 17. 여섯 손가락을 다섯으로 답하는 실패 회로**

![Circuit Tracing in Vision–Language Models — 여섯 손가락을 다섯으로 답하는 실패 회로](https://arxiv.org/html/2602.20330v1/figures/circuit_hand.jpg)

*원본: [Circuit Tracing in Vision–Language Models, Figure 10](https://arxiv.org/html/2602.20330v1#S11.F10).*

실제 이미지와 모델 답변의 불일치를 추적하는 사례다. 회로 그림의 각 노드가 어떤 시각 정보 또는 언어적 기대와 연결되는지 본다.

**그림 18. 간단한 덧셈에서의 VLM 회로**

![Circuit Tracing in Vision–Language Models — 간단한 덧셈에서의 VLM 회로](https://arxiv.org/html/2602.20330v1/figures/circuit_addition.jpg)

*원본: [Circuit Tracing in Vision–Language Models, Figure 13](https://arxiv.org/html/2602.20330v1#S11.F13).*

같은 도구를 산술 입력에 적용한 원본 사례다. 물체 인식 사례와 비교해 중간 특징과 최종 답변 연결이 어떻게 달라지는지 살펴본다.

---

<a id="sec-11"></a>

## 11. Steering 방법을 비교하는 공통 좌표

이후의 방법들을 읽을 때 다음 여섯 항목을 채우면 된다.

1. **방향의 출처:** 평균 차이, PCA, probe, 현재 입력, attribution 중 무엇인가?
2. **개입 위치:** residual, attention head, MLP 내부 좌표, 시각 encoder 중 어디인가?
3. **적용 방식:** 덧셈, 곱셈, 투영 제거, norm 보존 중 무엇인가?
4. **적용 시점:** 모든 답변 위치인가, 선택한 토큰인가, 위험할 때인가?
5. **적응하는 대상:** 방향 자체인가, 강도인가, 마스크인가, on/off 결정인가?
6. **검증 대상:** 진실성, 이미지 충실도, 특정 성향, 에이전트 성공률 중 무엇인가?

‘동적’이라는 한 단어로는 충분하지 않다. 고정된 방향에 현재 상태로 계산한 강도를 곱하는 방법과, 입력마다 새 방향을 만드는 방법은 다르다. 방향은 고정되어 있어도 현재 활성화에 좌표별로 곱하면 실제 변화량은 매번 달라진다. 이 차이가 SADI, REVIS, RUDDER를 이해하는 핵심이다.

<a id="sec-12"></a>

## 12. CAA와 ITI: 평균 방향과 선택한 head

**CAA 원문 요약.** Contrastive Activation Addition은 행동이 다른 대조 응답에서 residual activation의 차이를 모아 방향을 만들고, 생성 중 선택한 위치에 더한다. 기본 형태는 `v = mean(h_positive − h_negative)`, `h' = h + αv`다. 대조 자료를 어떻게 구성했는지가 방향의 의미를 결정한다. 이름은 행동 개념을 가리키더라도 길이·어조·형식 차이까지 함께 담길 수 있다. [CAA, ACL 2024](https://aclanthology.org/2024.acl-long.828/)

**ITI 원문 요약.** Inference-Time Intervention은 truthfulness와 관련된 정보를 가진 attention head를 probe로 찾아 일부 head에 개입한다. head를 고르는 데 쓰는 분류기의 가중치와 실제 개입 방향을 무조건 같은 것으로 보면 안 된다. 논문은 mean-shift 등의 선택을 다루며, 선택한 head와 개입 강도에 따라 효과와 부작용이 달라진다. 보고된 benchmark 개선을 모든 질문에서 사실이 보장된다는 의미로 일반화할 수 없다. [ITI, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/81b8390039b7302c909cb769f8b6cd93-Abstract-Conference.html)

**종합 해석.** 두 접근을 연결하는 생각은 ‘많은 계산을 다시 학습시키지 않고도 내부의 일부 방향을 바꾸면 행동이 움직일 수 있다’는 것이다. 다른 점은 어디를 선택하느냐다. 큰 상태 공간의 방향을 다루는 것과, 특정 head를 골라 개입하는 것은 자유도와 부작용의 구조가 다르다. 어느 쪽이 더 좋을지는 같은 데이터·같은 출력 길이·같은 개입 에너지에서 비교해야 한다.


### 원본 그림으로 확인하기

**그림 19. CAA 벡터 만들기와 추론 중 적용**

![Contrastive Activation Addition (CAA) — CAA 벡터 만들기와 추론 중 적용 · 패널 1](https://arxiv.org/html/2312.06681v4/generation_of_vector.svg)

![Contrastive Activation Addition (CAA) — CAA 벡터 만들기와 추론 중 적용 · 패널 2](https://arxiv.org/html/2312.06681v4/inference.svg)

*원본: [Contrastive Activation Addition (CAA), Figure 1](https://arxiv.org/html/2312.06681v4#S0.F1).*

원문의 두 패널을 함께 제시한다. 대조 응답의 평균 차이로 방향을 구성하는 단계와 새 입력에서 그 방향을 더하는 단계를 분리한다.

**그림 20. 행동별 CAA의 실제 효과**

![Contrastive Activation Addition (CAA) — 행동별 CAA의 실제 효과](https://arxiv.org/html/2312.06681v4/ab_results.svg)

*원본: [Contrastive Activation Addition (CAA), Figure 4](https://arxiv.org/html/2312.06681v4#S4.F4).*

7B와 13B 모델의 객관식 행동 평가 결과다. 한 행동에서의 제어 효과를 다른 행동이나 자유 응답의 품질까지 확대 해석하지 않는다.

**그림 21. 어떤 attention head가 진실성 정보를 담는가**

![Inference-Time Intervention (ITI) — 어떤 attention head가 진실성 정보를 담는가](https://arxiv.org/html/2306.03341v6/figs/headacc.png)

*원본: [Inference-Time Intervention (ITI), Figure 2](https://arxiv.org/html/2306.03341v6#S3.F2).*

head별 선형 probe 성능과 선택한 head의 활성화 분포다. 정보가 구분된다는 관찰은 유용하지만 성공적인 제어 여부는 후속 개입 결과로 확인한다.

**그림 22. ITI의 개입 강도와 head 수에 따른 변화**

![Inference-Time Intervention (ITI) — ITI의 개입 강도와 head 수에 따른 변화](https://arxiv.org/html/2306.03341v6/figs/sweeping.png)

*원본: [Inference-Time Intervention (ITI), Figure 4](https://arxiv.org/html/2306.03341v6#S4.F4).*

원문의 강도 및 head 수 탐색 결과다. 한 지표만 최대화하지 말고 진실성과 정보성 등 여러 지표가 같이 어떻게 움직이는지 본다.

---

<a id="sec-13"></a>

## 13. SADI: 고정된 마스크, 상태에 따라 달라지는 변화량

**원문 요약.** SADI는 대조 사례에서 의미 관련 요소를 선택하고, 그 요소의 현재 활성화에 따라 개입량을 만든다. 대표적인 이해 방식은 `Δh = δ(M ⊙ h)`다. `M`이 offline에서 정해진 선택 마스크라면, 새 입력마다 마스크를 다시 찾는다는 뜻은 아니다. 현재의 `h`가 달라지므로 변화량이 달라진다. hidden state, head, neuron 등 여러 수준의 변형을 다룬다. [Semantics-Adaptive Activation Intervention, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c4d26a95fd83f8e590f81c54ae670b5d-Abstract-Conference.html)

**종합 해석.** 덧셈과 스케일링은 다음 가상 예에서 분명히 다르다. 활성화가 `h=[2,−3,0]`이고 두 번째 좌표만 선택했다고 하자. 그 좌표를 1.5배 하면 `[2,−4.5,0]`이 된다. 수치는 더 작아졌지만 절댓값은 커졌다. 그러므로 ‘선택한 특징을 증가시킨다’를 모든 좌표의 숫자가 양의 방향으로 이동한다는 뜻으로 읽으면 안 된다.

또한 현재 활성화가 거의 0인 좌표는 곱셈으로 크게 살아나지 않는다. 덧셈은 꺼져 있던 방향을 새로 밀어 넣을 수 있고, 곱셈은 이미 존재하는 성분을 재가중한다. 이것은 어느 방식이 우월하다는 판단이 아니라, 개입이 가진 작동상의 차이다.


### 원본 그림으로 확인하기

**그림 23. 차이 추출, 이진 마스크, 적응형 개입**

![SADI — 차이 추출, 이진 마스크, 적응형 개입](https://arxiv.org/html/2410.12299v2/framework.svg)

*원본: [SADI, Figure 1](https://arxiv.org/html/2410.12299v2#S1.F1).*

대조 자료에서 중요한 성분을 고른 뒤 현재 활성화에 마스크를 적용한다. 고정된 마스크와 입력에 따라 달라지는 실제 변화량을 구분한다.

**그림 24. 개입 강도와 head 수의 상호작용**

![SADI — 개입 강도와 head 수의 상호작용](https://arxiv.org/html/2410.12299v2/head-strength-4tasks.svg)

*원본: [SADI, Figure 2](https://arxiv.org/html/2410.12299v2#S4.F2).*

네 과제에 대한 원본 실험이다. 같은 설정이 모든 과제에 최적이라는 전제 없이 강도와 선택 범위가 성능을 어떻게 바꾸는지 본다.

---

<a id="sec-14"></a>

## 14. MERA: 예측 오류를 줄이는 최소 변화와 보정

**원문 요약.** MERA는 hidden representation에서 오류를 예측하는 회귀 모델을 만들고, 예측 오류를 기준 이하로 낮추는 작은 개입을 찾는다. 별도의 calibration으로 개입의 유용성을 판단하며, 근거가 부족하면 steering을 하지 않는 선택도 포함한다. 분포와 보정의 전제 아래 기대 성능에 관한 판단을 하는 것이지 개별 답변의 무오류를 보장하는 것은 아니다. 개입을 생략한다는 의미의 abstention과 모델이 사용자에게 답변을 거절한다는 의미는 구분해야 한다. [MERA, ICML 2025](https://proceedings.mlr.press/v267/hedstrom25a.html), [상세 수식](https://arxiv.org/html/2510.13290v1)

선형 오류 추정기를 `e_hat(h)=wᵀh+b`로 단순화해 보자. 다음 문제의 해는 직접 유도할 수 있다.

```text
최소화: ||v||²
제약:   e_hat(h+v) ≤ τ

v* = − max(0, e_hat(h)−τ) / ||w||² × w      (w ≠ 0)
```

**수식 해설.** 이미 예측 오류가 기준보다 낮으면 움직일 이유가 없으므로 `v=0`이다. 높다면 `w`의 반대 방향으로 가야 한다. 예를 들어 `w=[1,0]`, `b=0`, `h=[3,4]`, `τ=1`이면 최소 변화는 `v=[−2,0]`이다. 두 번째 좌표는 오류 추정값에 영향을 주지 않으므로 바꾸지 않는다.

이 계산은 선형 추정기 안에서는 명확하다. 실제 생성 오류는 선형 함수가 아니므로, 추정기가 보는 점수만 낮추고 실제 품질은 악화할 가능성은 별도 검증해야 한다. 이 때문에 방향 학습 데이터와 calibration 데이터, 최종 평가 데이터를 나누는 것이 중요하다. 수식을 구현할 때도 ‘오류를 낮추는 방향’이라는 제약과 실제 부호가 일치하는지 작은 수치 예제로 확인하는 편이 좋다.


### 원본 그림으로 확인하기

**그림 25. 일정한 강도와 보정한 임계값 비교**

![MERA — 일정한 강도와 보정한 임계값 비교](https://arxiv.org/html/2510.13290v1/assets/front_image.png)

*원본: [MERA, Figure 1](https://arxiv.org/html/2510.13290v1#S1.F1).*

고정 강도로 인한 부족하거나 과한 개입과 MERA의 보정 개념을 비교하는 원문 개요다. 이 도식 자체는 실험 성능 곡선이 아니다.

**그림 26. 오류 추정기를 학습하고 임계값을 보정하는 절차**

![MERA — 오류 추정기를 학습하고 임계값을 보정하는 절차](https://arxiv.org/html/2510.13290v1/assets/method.png)

*원본: [MERA, Figure 3](https://arxiv.org/html/2510.13290v1#S4.F3).*

활성화와 오류 수집, 오류 추정기 학습, steering 임계값 보정이 별도 단계다. 보정 자료와 최종 평가 자료의 역할을 나누어 읽는다.

**그림 27. 모델 및 데이터셋별 MERA 결과**

![MERA — 모델 및 데이터셋별 MERA 결과](https://arxiv.org/html/2510.13290v1/assets/main_rebuttal.png)

*원본: [MERA, Figure 4](https://arxiv.org/html/2510.13290v1#S6.F4).*

모델별·자료별 결과와 오류 분위수를 함께 보여 준다. 특정 평균 개선이 모든 입력의 오류 감소를 보장하는 것으로 읽어서는 안 된다.

---

<a id="sec-15"></a>

## 15. AUSteer: 큰 상태 전체보다 작은 activation unit

**원문 요약.** AUSteer는 선형 연산 `y=Wx=Σ x_i W[:,i]`에서 활성화 스칼라 `x_i`와 가중치 열의 기여를 activation unit으로 다룬다. 대조 사례에서 변화 방향의 일관성을 평가해 일부 AU를 고르고, 선택한 활성화를 비례 조절한다. 이 AU는 attention head 전체도 아니고 SAE가 새로 학습한 feature도 아니다. 적은 수의 요소를 수정하는 실험은 작은 개입의 가능성을 보여주지만, 고정된 개수나 위치가 모든 모델에 통한다는 뜻은 아니다. [Fine-Grained Activation Steering: Steering Less, Achieving More](https://arxiv.org/html/2602.04428)

**종합 해석.** ‘많이 다른 좌표’와 ‘차이의 방향이 일관된 좌표’는 다르다. 어떤 좌표는 일부 사례에서 아주 크게 변하고 나머지에서는 부호가 뒤집힐 수 있다. 다른 좌표는 변화량이 작아도 거의 항상 같은 방향으로 변할 수 있다. 어떤 선택 기준이 좋은지는 목표 행동과 자료의 잡음 구조에 따라 달라진다.

스케일링을 `x'_i=(1+γ_i)x_i`라고 쓰면, 원래 부호를 유지하려면 `1+γ_i≥0`이라는 조건이 필요하다. ‘곱셈이므로 언제나 부호가 보존된다’는 설명은 계수 범위를 확인하지 않으면 성립하지 않는다. 이런 작은 조건이 실제 구현의 안정성을 좌우한다.


### 원본 그림으로 확인하기

**그림 28. 블록 수준 개입과 activation unit 수준 개입**

![AUSteer — 블록 수준 개입과 activation unit 수준 개입](https://arxiv.org/html/2602.04428v1/Intro_fig.png)

*원본: [AUSteer, Figure 1](https://arxiv.org/html/2602.04428v1#S1.F1).*

어느 크기의 내부 성분을 제어하는지가 핵심 차이다. 작은 단위를 고른다는 설계와 실제 부작용 감소의 증거는 구분해서 본다.

**그림 29. 중요한 AU 탐색과 적응형 steering**

![AUSteer — 중요한 AU 탐색과 적응형 steering](https://arxiv.org/html/2602.04428v1/overview_v32.png)

*원본: [AUSteer, Figure 5](https://arxiv.org/html/2602.04428v1#S4.F5).*

activation momentum과 판별 점수로 AU를 고르는 절차 및 입력별 조절을 보여 준다. 어떤 통계로 선택했고 무엇을 조절하는지 확인한다.

**그림 30. 강도와 선택 수에 따른 과제별 성능**

![AUSteer — 강도와 선택 수에 따른 과제별 성능](https://arxiv.org/html/2602.04428v1/heper_search.png)

*원본: [AUSteer, Figure 10](https://arxiv.org/html/2602.04428v1#A3.F10).*

COPA, MAWPS, BPO의 원본 heatmap이다. 최적 영역이 과제마다 어떻게 달라지는지 보며 보정 설정의 의존성을 확인한다.

---

<a id="sec-16"></a>

## 16. VLM의 hallucination을 하나의 원인으로 묶지 않기

이미지에 없는 물체를 말하는 현상은 비슷해 보여도 여러 경로에서 생길 수 있다. 시각 encoder가 중요한 세부를 놓쳤을 수도 있다. 이미지 표현은 있지만 언어 모델이 충분히 사용하지 않았을 수도 있다. 장면과 자주 함께 등장하는 객체를 언어적 연상으로 보충했을 수도 있다. 질문의 전제를 그대로 받아들였거나, 긴 설명을 요구받으면서 근거 없는 세부를 덧붙였을 수도 있다.

다음은 원인을 구분하기 위한 가상 사례다.

| 입력과 실패 | 가능한 조사 방향 | 단순 steering만으로 충분하지 않을 수 있는 이유 |
|---|---|---|
| 작은 표지판 글자를 잘못 읽음 | 해상도, 시각 토큰, OCR 관련 표현 | 애초에 필요한 시각 정보가 손실되었을 수 있음 |
| 접시 사진에 없는 포크를 언급 | 장면의 언어적 연상, 이미지 의존성 | 전형적인 장면 지식과 관측을 혼합했을 수 있음 |
| ‘빨간 가방은 어디?’에 가방이 있다고 가정 | 질문 전제, 반박 능력 | 질문을 신뢰하는 행동이 문제일 수 있음 |
| 짧은 답은 맞지만 긴 설명에서 오류 | 길이, 생성 후반, 사실 누적 | 평균 점수만으로 오류 발생 시점을 놓칠 수 있음 |

따라서 좋은 평가에는 성공률만이 아니라 오류 유형과 생성 위치도 들어가야 한다. 개입 후 답변이 짧아졌다면 오류가 줄어도 시각적 이해가 좋아졌다고 바로 판단할 수 없다. 반대로 더 자세히 설명하면서 같은 오류율을 유지한다면 유용성은 개선되었을 수 있다. 출력 길이와 정보량을 함께 봐야 하는 이유다.

<a id="sec-17"></a>

## 17. VTI: 시각 경로와 언어 경로를 함께 조정하기

**원문 요약.** VTI는 visual intervention과 textual intervention을 결합한다. 이미지의 perturbation에서 얻는 시각 표현 차이와, 충실한 응답·환각 응답의 텍스트 표현 차이를 활용해 방향을 구성한다. 소수의 demonstration으로 미리 만든 방향을 시각 encoder와 언어 모델 쪽에 적용하는 것이 핵심이다. 새 테스트 입력마다 gradient를 구해 새 방향을 만드는 방식은 아니다. [VTI](https://arxiv.org/html/2410.15778v2)

**종합 해석.** 이 방법은 실패를 한 지점에서만 고치려 하지 않는다는 점에서 이해할 수 있다. 입력에서 얻는 시각 신호와 그 신호를 언어로 사용하는 방식이 모두 결과에 영향을 줄 수 있기 때문이다. 다만 두 경로에 개입해 성능이 좋아졌다면 각 경로의 기여를 분리해야 한다. visual only, textual only, 둘 다, 개입 없음의 네 조건이 가장 먼저 필요하다.

이미지를 가리는 대조도 조심해서 해석해야 한다. 원본과 마스킹 이미지의 차이는 객체의 의미뿐 아니라 밝기, 질감, 손상 패턴, attention 분포 차이까지 포함한다. 따라서 추출한 방향을 곧바로 ‘순수한 시각적 진실성’이라고 부르는 것은 과도하다. 이미지 perturbation의 종류를 바꾸어도 효과가 유지되는지 시험하면 이 우려를 줄일 수 있다.


### 원본 그림으로 확인하기

**그림 31. 시각 개입 뒤 이미지 교란에 대한 특징 안정성**

![Visual and Textual Intervention (VTI) — 시각 개입 뒤 이미지 교란에 대한 특징 안정성](https://arxiv.org/html/2410.15778v2/feature_variance.svg)

*원본: [Visual and Textual Intervention (VTI), Figure 5](https://arxiv.org/html/2410.15778v2#S5.F5).*

서로 다른 이미지 교란에서 시각 특징의 변동을 비교한 원본 결과다. 특징의 안정성과 최종 답변의 정확도는 별개의 측정값이므로 아래의 응답 평가와 함께 읽는다.

**그림 32. 시각 encoder와 텍스트 decoder의 두 경로 개입**

![Visual and Textual Intervention (VTI) — 시각 encoder와 텍스트 decoder의 두 경로 개입](https://arxiv.org/html/2410.15778v2/vti_overview.png)

*원본: [Visual and Textual Intervention (VTI), Figure 3](https://arxiv.org/html/2410.15778v2#S4.F3).*

예시 자료에서 방향을 미리 계산하고 새 입력에 적용하는 전체 절차다. 현재 입력에서 매번 방향을 만드는 RUDDER와 대조하기 좋은 원본 그림이다.

**그림 33. MMHAL-Bench 세부 범주별 결과**

![Visual and Textual Intervention (VTI) — MMHAL-Bench 세부 범주별 결과](https://arxiv.org/html/2410.15778v2/mmhal_radar.svg)

*원본: [Visual and Textual Intervention (VTI), Figure 4](https://arxiv.org/html/2410.15778v2#S5.F4).*

LLaVA-1.5 기반의 범주별 점수다. 전체 평균 외에 어떤 질문 유형에서 이득이 크고 작은지 살펴본다.

---

<a id="sec-18"></a>

## 18. REVIS: 고정 방향에서 prior 성분을 줄이고 필요할 때 개입

**원문 요약.** REVIS는 이미지가 있을 때와 없을 때의 표현 차이에서 시각적 grounding 관련 방향을 만들고, 이미지 없이도 나타나는 hallucination prior 관련 성분을 제거하는 구성을 사용한다. 선택한 레이어에서 현재 상태와 방향의 관계로 위험 신호를 계산해 일부 생성 시점에 개입한다. 방향의 정제와 token-level gate가 별개라는 점이 중요하다. 여기의 sparse는 개입 빈도·위치와 관련되며 SAE를 쓴다는 의미가 아니다. [REVIS](https://arxiv.org/html/2602.11824)

**수식 해설.** 단일 prior 방향 `p`에 대한 투영 제거는 다음처럼 쓸 수 있다.

```text
d_clean = d_raw − p × (pᵀd_raw)/(pᵀp)       (p ≠ 0)
```

이 식이 보장하는 것은 지정한 벡터 공간에서 `d_clean`과 `p`가 직교한다는 사실이다. 모든 언어적 편향이나 환각 원인이 사라진다는 뜻은 아니다. prior가 여러 방향에 퍼져 있거나, 제거한 방향에 유용한 시각 정보도 섞여 있으면 이 정제가 충분하지 않거나 손실을 만들 수 있다.

**종합 해석.** REVIS형 설계에서 확인할 것은 두 가지다. 정제한 방향이 정제 전보다 좋은가? 그리고 gate를 붙였을 때 같은 평균 개입량을 가진 상시 개입보다 좋은가? 둘을 동시에 바꾸고 최종 점수만 보고하면 성능 향상이 방향 덕분인지 gate 덕분인지 알기 어렵다.


### 원본 그림으로 확인하기

**그림 34. 언어 prior 성분을 줄인 시각 방향**

![REVIS — 언어 prior 성분을 줄인 시각 방향](https://arxiv.org/html/2602.11824v2/fig1_REVIS.png)

*원본: [REVIS, Figure 1](https://arxiv.org/html/2602.11824v2#S1.F1).*

직교 투영을 포함한 REVIS의 개요와 예시다. 투영으로 제거한 지정 성분과 의미적으로 완전히 순수한 시각 정보라는 주장을 동일시하지 않는다.

**그림 35. 개입 강도가 커질 때의 성능 변화**

![REVIS — 개입 강도가 커질 때의 성능 변화](https://arxiv.org/html/2602.11824v2/comparison_plot_revis.svg)

*원본: [REVIS, Figure 3](https://arxiv.org/html/2602.11824v2#S3.F3).*

원문에서 naive steering과 비교한 민감도 곡선이다. 큰 강도의 성능 하락은 방향 선택뿐 아니라 개입 크기도 평가해야 함을 보여 준다.

**그림 36. 방향 구성, 층 선택, 생성 중 조절**

![REVIS — 방향 구성, 층 선택, 생성 중 조절](https://arxiv.org/html/2602.11824v2/REVIS_design.png)

*원본: [REVIS, Figure 4](https://arxiv.org/html/2602.11824v2#S3.F4).*

보정 자료를 이용한 층 선택과 추론 시점의 동적 개입이 구분되어 있다. 고정되어 있는 부분과 매 입력·토큰에서 달라지는 부분을 짚어 본다.

---

<a id="sec-19"></a>

## 19. RUDDER: 현재 입력에서 anchor를 만들고 상태에 맞춰 주입

**원문 요약.** RUDDER는 현재 이미지·질문을 처리할 때 얻는 residual-update 관련 신호를 모아 CARD 방향을 구성하고, 생성 상태와의 유사도로 개입 강도를 조절한다. 고정된 offline 방향을 모든 입력에 쓰는 계열과 달리 입력별 anchor를 사용한다. 논문과 구현을 읽을 때 ‘single-pass’라는 설계 설명, 특정 실행 경로의 실제 forward 횟수, gate의 수학적 방향을 각각 확인해야 한다. [RUDDER](https://arxiv.org/html/2511.10292v2)

공개 구현에서 볼 수 있는 gate의 핵심 형태는 다음과 같다.

```text
s = cosine(h, v)
a = softplus(k*s + c)
b = softplus(−k*s + c)
g = a/(a+b)
h' = h + α*g*v
```

`k>0`이라면 기본 gate는 유사도 `s`가 커질수록 커진다. 즉 anchor와 더 잘 맞는 상태를 더 신뢰하는 형태로 이해해야 한다. ‘anchor에서 멀어질수록 더 세게 고친다’는 반대 방향의 gate와 혼동하면 안 된다. 실제 코드에는 clamp 등의 추가 처리가 있다. Bayesian이라는 해석이 붙어 있어도, 이 값이 실제 오류 확률로 calibration되었다고 자동으로 간주할 수는 없다. [RUDDER 공개 구현](https://github.com/Akko000/RUDDER-Residual-Update-Directed-DEcoding-Regulation-/blob/main/LLaVA/llava_methods.py)

**종합 해석.** 입력별 anchor는 적응성이 높지만 그 anchor가 신뢰할 만하다는 전제가 필요하다. 원래 입력 처리부터 잘못된 정보를 강조했다면 그 상태를 더 강하게 고정할 수도 있다. 이미지 교체, 질문 교체, 동일 이미지의 다른 crop에서 anchor가 어떻게 바뀌는지 기록하면 입력 특이성과 안정성을 함께 볼 수 있다.


### 원본 그림으로 확인하기

**그림 37. RUDDER의 환각 수정 예시와 설계 비교**

![RUDDER — RUDDER의 환각 수정 예시와 설계 비교](https://arxiv.org/html/2511.10292v2/motivation4_cropped.png)

*원본: [RUDDER, Figure 1](https://arxiv.org/html/2511.10292v2#S1.F1).*

원문이 제시한 응답 예시와 방법 특성 비교다. 오른쪽 비교표는 저자의 정리이며 구체적인 비용 판단에는 본문의 실행 조건이 필요하다.

**그림 38. prefill에서 CARD를 만들고 decoding에서 조절**

![RUDDER — prefill에서 CARD를 만들고 decoding에서 조절](https://arxiv.org/html/2511.10292v2/structure12_cropped.png)

*원본: [RUDDER, Figure 2](https://arxiv.org/html/2511.10292v2#S3.F2).*

현재 입력의 residual update를 모아 기준 방향을 얻고 Adaptive Beta Gate로 토큰별 개입을 조절한다. 방향 생성 시점과 강도 계산 시점을 나누어 본다.

**그림 39. 어느 층에 주입하는지가 성능에 미치는 영향**

![RUDDER — 어느 층에 주입하는지가 성능에 미치는 영향](https://arxiv.org/html/2511.10292v2/layer_fixedAK_chair.svg)

*원본: [RUDDER, Figure 3](https://arxiv.org/html/2511.10292v2#S4.F3).*

Idefics2에서 다른 설정을 고정한 층별 ablation이다. 이 모델에서의 유리한 층 번호를 다른 모델에 그대로 복사할 근거로 사용하지 않는다.

---

<a id="sec-20"></a>

## 20. GrAInS: 중요한 토큰을 찾은 뒤 방향 만들기

**원문 요약.** GrAInS는 gradient attribution으로 목표 응답과 관련된 토큰을 찾아 ablation하고, 그로 인한 hidden-state 차이에서 steering 방향을 구성하는 접근이다. 목적함수는 긍정·부정 응답의 로그 확률 차이처럼 대조적으로 설정할 수 있다. 중요한 점은 gradient가 주입할 벡터와 곧바로 동일한 것은 아니라는 것이다. attribution은 무엇을 제거하거나 선택할지를 정하는 데 쓰이고, 이후의 표현 차이가 방향 구성에 사용된다. 이 준비 작업과 테스트 시 생성 비용도 구분해야 한다. [GrAInS](https://arxiv.org/html/2507.18043v2)

**종합 해석.** 평균 대조 방향은 전체 응답의 차이를 담기 쉽다. attribution을 중간에 넣으면 그 차이 가운데 목표 출력과 관련된 부분을 선택하려는 시도가 된다. 그러나 attribution의 목적함수를 잘못 잡으면 선택도 어긋난다. 마지막 토큰 하나의 확률을 높이는 것과 전체 응답의 사실성을 높이는 것은 다르다. 긴 응답에서는 조사나 문장 종료 토큰이 중요한 점수에 영향을 줄 수도 있다.

Ablation도 중립적 관측이 아니다. 토큰을 PAD로 바꾸는 것, UNK로 바꾸는 것, 아예 삭제하는 것, 자연스러운 대체 표현을 넣는 것은 서로 다른 입력을 만든다. 방향이 목표 의미를 포착했는지, 단지 부자연스러운 문장을 복구하는지 확인하려면 여러 ablation 기준을 비교해야 한다.


### 원본 그림으로 확인하기

**그림 40. attribution에서 마스킹, PCA, 주입까지**

![GrAInS — attribution에서 마스킹, PCA, 주입까지](https://arxiv.org/html/2507.18043v2/vlms_steering_2.png)

*원본: [GrAInS, Figure 2](https://arxiv.org/html/2507.18043v2#S3.F2).*

부호가 있는 Integrated Gradients로 토큰을 고르고 대조 입력을 구성한 뒤 방향을 만든다. 마지막 주입뿐 아니라 방향 구성의 계산비용도 함께 읽는다.

**그림 41. 양·음 attribution 토큰 제거의 차이**

![GrAInS — 양·음 attribution 토큰 제거의 차이](https://arxiv.org/html/2507.18043v2/ablation_effect.svg)

*원본: [GrAInS, Figure 3](https://arxiv.org/html/2507.18043v2#S3.F3).*

TruthfulQA의 두 언어 모델에서 토큰 제거 뒤 선호도 차이를 측정한 실험이다. VLM 성능 자체를 그린 곡선이 아니라 attribution 선택을 뒷받침하는 분석이다.

---

<a id="sec-21"></a>

## 21. 아홉 steering 계열을 한 표에서 비교하기

아래는 앞 절의 내용을 같은 기준으로 압축한 종합 비교다. 구현 변형을 모두 표현하는 표는 아니므로, 실제 재현에서는 해당 버전의 코드와 설정을 우선한다.

| 방법 | 주된 정보원 | 적응성이 나타나는 곳 | 놓치기 쉬운 구분 |
|---|---|---|---|
| CAA | 대조 활성화의 평균 차이 | 기본형은 방향 고정 | 행동과 함께 문체·길이도 포착 가능 |
| ITI | truthfulness 관련 head·대조 정보 | head 선택과 강도 설정 | head 선택 probe와 개입 방향 구분 |
| SADI | 대조 자료로 고른 요소 | 현재 활성화에 비례한 변화량 | 고정 마스크여도 변화량은 동적 |
| MERA | 오류 예측 회귀와 calibration | 예측 오류·보정 결과에 따른 개입 | 추정 오류와 실제 오류 구분 |
| AUSteer | AU별 대조 변화의 일관성 | 선택 좌표의 현재 값 조절 | AU, head, SAE feature는 다름 |
| VTI | 이미지 perturbation·응답 대조 | 기본 방향은 미리 구성 | visual과 textual 효과 분리 |
| REVIS | grounding 차이와 prior 방향 | 생성 시점별 gate | 방향 정제와 적용 시점 분리 |
| RUDDER | 현재 입력 처리의 update 신호 | 입력별 방향과 생성별 강도 | 신뢰 gate와 위험 gate 구분 |
| GrAInS | attribution·ablation의 표현 차이 | 방향 구성 단계의 선택 | raw gradient를 바로 더하는 것이 아님 |

‘가장 발전된 방법’을 한 줄로 고르는 것보다 내가 가진 자원과 목표를 먼저 고르는 편이 정확하다. 대조 데이터가 없고 입력마다 적응해야 하는지, offline 비용을 써도 되는지, 아주 적은 좌표만 바꾸고 싶은지, 실제 오류 추정치를 얻을 수 있는지에 따라 적합한 설계가 달라진다. 이 표는 우승자를 정하는 표가 아니라 실험 조건을 설계하는 표다.

<a id="sec-22"></a>

## 22. TraitBasis: 제어를 평가 환경을 만드는 데 사용하기

**원문 요약.** TraitBasis는 impatience, confusion, skepticism, incoherence 같은 성향을 표현하는 대조 자료에서 방향을 만들고, 사용자 시뮬레이터의 행동을 조절한다. 여러 성향을 합성하고 강도를 조정하면서 별도의 목표 에이전트가 대화를 얼마나 잘 수행하는지 평가한다. τ-Trait 등의 환경에서 사용자의 성향 변화가 에이전트 성능을 크게 바꿀 수 있음을 보인다. 이 연구에서 steering 대상은 평가받는 에이전트가 아니라 사용자 역할을 하는 모델이라는 구분이 중요하다. 특정 실험의 성능 감소를 모든 실서비스에 같은 비율로 적용할 수는 없다. [Impatient Users Confuse AI Agents](https://arxiv.org/html/2510.04491v3)

**종합 해석.** 이 응용은 ‘더 좋은 답변을 만들기 위해 steering한다’는 익숙한 사용법을 넓힌다. 테스트 상대의 행동을 바꾸어 시스템의 취약성을 찾을 수도 있다. 예를 들어 사용자 시뮬레이터가 질문에 답을 늦게 주거나, 조건을 중간에 바꾸거나, 같은 요구를 다르게 표현하도록 만들면 에이전트가 상태를 안정적으로 관리하는지 볼 수 있다.

다만 시뮬레이터가 성향을 표현한다는 것과 실제 인간의 분포를 재현한다는 것은 다르다. impatience를 높였더니 단순히 답변 길이만 짧아졌다면, 복합적인 조급함을 충분히 구현했다고 보기 어렵다. 성향의 fidelity, 장시간 대화의 안정성, 다른 성향과 섞었을 때의 간섭, 과제 자체를 수행할 능력의 보존을 함께 측정해야 한다.


### 원본 그림으로 확인하기

**그림 42. 성향 벡터 추출과 사용자 시뮬레이션**

![TraitBasis — 성향 벡터 추출과 사용자 시뮬레이션 · 패널 1](https://arxiv.org/html/2510.04491v3/TraitBasis_1.drawio.png)

![TraitBasis — 성향 벡터 추출과 사용자 시뮬레이션 · 패널 2](https://arxiv.org/html/2510.04491v3/TraitBasis_2.drawio.png)

*원본: [TraitBasis, Figure 1](https://arxiv.org/html/2510.04491v3#S1.F1).*

원문 두 패널을 함께 제시한다. steering되는 사용자 시뮬레이터와 그 사용자에게 응답하는 평가 대상 에이전트를 구별한다.

**그림 43. 성향의 현실감을 비교한 Elo와 승률**

![TraitBasis — 성향의 현실감을 비교한 Elo와 승률 · 패널 1](https://arxiv.org/html/2510.04491v3/elo.svg)

![TraitBasis — 성향의 현실감을 비교한 Elo와 승률 · 패널 2](https://arxiv.org/html/2510.04491v3/wr.svg)

*원본: [TraitBasis, Figure 2](https://arxiv.org/html/2510.04491v3#S6.F2).*

방법 간 쌍대 비교의 두 지표다. 현실감 평가에서의 우세와 실제 사용자의 분포를 완벽히 재현한다는 주장은 다른 수준의 주장이다.

**그림 44. 두 성향을 함께 표현했는지 평가하기**

![TraitBasis — 두 성향을 함께 표현했는지 평가하기](https://arxiv.org/html/2510.04491v3/match_accuracy.svg)

*원본: [TraitBasis, Figure 5](https://arxiv.org/html/2510.04491v3#A1.F5).*

부분 일치와 완전 일치의 차이를 함께 본다. 한 성향만 강하게 드러나는 결과를 성공적인 성향 조합으로 계산하지 않도록 하는 평가다.

---

<a id="sec-23"></a>

## 23. 모든 방법을 연결하는 하나의 설계식

아래 식은 특정 논문의 공식식이 아니라, 서로 다른 개입을 비교하기 위해 만든 공통 표기다.

```text
h'[l,t] = h[l,t] + g(x,l,t) · λ(x,l,t) · ( M(x,l,t) ⊙ d(x,l,t) )

x : 현재 입력과 생성 문맥
d : 움직일 방향 또는 현재 상태에서 계산한 변화 후보
M : 수정할 좌표·특징을 고르는 마스크
λ : 강도
g : 적용 여부 또는 연속 gate
```

곱셈형 개입은 `d=h`로 놓고 `M`과 `λ`에 선택·스케일링을 담을 수 있다. 고정 방향 덧셈은 `d=v`, `M=1`, `g=1`인 특수한 경우다. 입력별 방향은 `d`가 입력에 따라 달라진다. 토큰 선택은 `g`에 들어간다. 실제 방법에는 projection, norm 보정, 여러 모듈의 다른 hook이 추가되므로 이 식 하나로 구현 전체를 대체하지는 못한다.

이 표기의 장점은 새로운 방법을 만들 때 변경한 요소를 분명하게 보여준다는 데 있다. 방향을 바꾸었는데 gate 개선이라고 주장하거나, 강도를 줄였는데 sparse selection의 효과라고 주장하는 혼동을 막을 수 있다.

예를 들어 ‘입력별 방향이 도움이 된다’는 가설을 시험하려면 다른 요소를 고정하고 `d_fixed`와 `d_input`을 비교한다. ‘위험할 때만 개입하면 부작용이 줄어든다’는 가설은 같은 `d`에 대해 `g=1`과 `g=risk_gate`를 비교해야 한다. 이런 요소별 실험이 여러 아이디어를 한꺼번에 넣은 복잡한 시스템보다 해석하기 쉽다.

<a id="sec-24"></a>

## 24. 수식을 실제로 이해하기 위한 여섯 가지 작은 계산

### 24.1 평균 차이와 PCA는 다른 질문을 한다

대조 차이 벡터의 평균은 여러 사례가 공통적으로 어느 방향으로 이동했는지를 본다. 중심화된 PCA는 평균 주변에서 어느 방향으로 가장 크게 변동하는지 본다. 변화량이 `[1,10]`, `[1,−10]` 두 개라면 평균은 `[1,0]`이다. 그러나 중심화한 두 점은 `[0,10]`, `[0,−10]`이므로 첫 주성분은 두 번째 축이다.

따라서 ‘PCA를 쓰면 대조 평균을 더 잘 추정한다’는 말은 일반적으로 성립하지 않는다. 평균을 다시 더하는 구현인지, 중심화를 하는지, 첫 성분의 부호를 어떻게 정하는지에 따라 결과가 달라진다. PCA의 축은 부호가 뒤집혀도 같은 축이므로, 원하는 행동 방향과 정렬하는 기준이 필요할 수 있다.

### 24.2 직교화는 좌표계 안의 성분 제거다

`d=[2,3]`, `p=[1,0]`이면 prior 투영을 제거한 방향은 `[0,3]`이다. 첫 축의 성분을 지웠다는 것은 정확하다. 하지만 첫 축이 ‘환각만’ 나타낸다는 해석은 별도로 증명해야 한다. 그 축에 사실적인 객체 정보가 함께 들어 있다면 좋은 정보도 없어진다.

하나의 방향 대신 여러 열로 이루어진 행렬 `P`가 prior 부분공간을 나타낸다면, 적절한 rank 조건에서 `d_clean = d − P(PᵀP)⁻¹Pᵀd`처럼 확장할 수 있다. 수치적으로는 역행렬을 직접 만들기보다 안정적인 분해를 사용하는 편이 낫다. 제거할 공간을 넓히면 prior가 더 잘 제거될 수도 있지만 유용한 정보 손실도 늘어날 수 있다.

### 24.3 Norm을 유지해도 의미는 달라진다

`h=[1,0]`에 `v=[0,1]`을 더하고 길이를 1로 다시 맞추면 `[1/√2,1/√2]`가 된다. 크기는 원래와 같지만 방향은 45도 변했다. 출력 head가 첫 번째 좌표에 민감하다면 점수는 감소할 수 있다. Norm 보존은 활성화 크기의 변화라는 한 요인을 통제할 뿐, 의미·성능·분포가 보존된다는 보장이 아니다.

그래서 ‘벡터를 더하고 원래 norm으로 돌렸다’는 구현은 안정화 의도를 가진 조작으로 이해해야 한다. 그것만으로 부작용이 없다고 판단하지 않는다. 개입의 상대 크기 `||Δh||/||h||`, 방향 변화, 출력 변화도 함께 기록한다.

### 24.4 Gradient는 현재 위치 근처의 기울기다

간단히 `f(z)=z²`이고 `z=1`이라면 미분은 2다. 작은 변화 `ε`에 대해서는 `f(1+ε)≈1+2ε`다. 하지만 `ε=2`이면 이 근사는 5를 예측하고 실제 값은 9다. 큰 변화에서는 곡률을 무시할 수 없다.

모델의 attribution과 steering에서도 같은 문제가 생긴다. 미분상 중요한 방향이 작은 개입에는 잘 맞아도 큰 개입에서는 attention 분포, gating, 활성 특징이 바뀔 수 있다. 그래서 개입 강도 하나만 보고하기보다 작음·중간·큼의 변화 곡선을 보는 편이 낫다.

### 24.5 희소성은 어떤 기저에서 세는지에 달려 있다

2차원 벡터 `[1,0]`은 원래 좌표에서는 한 성분만 0이 아니다. 좌표계를 45도 회전하면 두 성분 모두 0이 아닐 수 있다. 따라서 ‘두 좌표만 수정했다’는 말은 어떤 표현 기저에서 세었는지를 포함한다.

원래 뉴런 좌표의 sparse intervention과 SAE feature의 sparse intervention은 같은 개수라도 계산적으로 다르다. 비교할 때는 선택한 feature 수 외에 원래 residual에서 발생한 변화량, 영향을 받는 위치, 출력 효과를 같이 보아야 한다.

### 24.6 Gate의 이름보다 단조성을 확인한다

유사도 `s`가 커질 때 gate가 커지는지 작아지는지를 먼저 그려 본다. 커지면 해당 anchor와의 일치를 더 신뢰하는 설계일 수 있고, 작아지면 불일치 상태를 더 강하게 수정하는 설계일 수 있다. 어느 쪽이 합리적인지는 anchor의 의미에 달려 있다.

‘Bayesian’, ‘adaptive’, ‘risk’ 같은 이름은 출발점일 뿐이다. 실제 식의 입력, 출력 범위, 극한값, clamp, threshold를 확인하면 많은 오해를 피할 수 있다. `s=-1,0,1`의 세 값만 대입해도 코드와 설명의 방향이 일치하는지 빠르게 검사할 수 있다.

<a id="sec-25"></a>

## 25. 공개 코드를 대조하면서 확인한 점

이 절은 2026년 10월 23일 확인한 공개 파일에 대한 정적 검토다. 링크의 `main`은 이후 변경될 수 있다. 아래 내용은 확인한 함수와 실행 경로에 한정하며, 저장소 전체 또는 논문의 모든 실험이 같은 문제를 가진다고 단정하지 않는다. 실제 재현에는 commit, 모델 구현, 라이브러리 버전, 실행 진입점까지 고정해야 한다.

### 25.1 VTI: 어느 축에서 방향을 만들고 어디에 주입하는가

`obtain_textual_vti`는 레이어별 상태를 펼친 대조 벡터를 모아 PCA를 적용하고 다시 레이어×차원 형태로 바꾼다. ‘각 레이어에서 독립적으로 PCA를 한 뒤 첫 성분을 쓴다’는 간단한 설명과 다르다. 방향 구성에는 PCA 성분과 평균을 결합하는 코드도 있으므로 일반적인 PCA 한 줄로 치환하면 같은 재현이 아니다. [icv_utils.py, L212](https://github.com/shengliu66/VTI/blob/main/vti_utils/icv_utils.py#L212)

`add_vti_layers`는 원래 MLP 뒤에 `VTILayer`를 붙인다. 즉 이 경로의 개입 대상은 MLP 출력이다. 블록 전체 residual output을 hook으로 바꾸는 것과 같은 위치라고 가정하면 안 된다. `VTILayer`의 정규화 순서, 내부 상수, dtype 변환도 효과를 바꿀 수 있다. [llm_layers.py, L115](https://github.com/shengliu66/VTI/blob/main/vti_utils/llm_layers.py#L115)

### 25.2 REVIS: PCA의 축과 행동 방향을 구분해야 한다

확인한 helper는 첫 PCA 성분에 샘플 벡터의 평균 norm을 곱한다. 이 helper 자체에는 대조 평균과의 내적으로 부호를 명시적으로 정렬하는 단계가 보이지 않는다. PCA의 부호는 의미상 임의이므로 호출부의 부호 처리, 실제 학습된 방향, gate 정의가 의도와 맞는지 확인할 필요가 있다. prior 방향을 만들고 raw difference에서 투영 성분을 제거한 뒤 다시 방향을 구성하는 순서도 재현에 포함된다. 이는 전체 실행의 오류를 확정하는 지적이 아니라 방향의 의미를 검증해야 할 위치다. [vector_calculator.py](https://github.com/antgroup/Revis/blob/main/steering_utils/vector_calculator.py)

### 25.3 RUDDER: CARD 추출 비용과 hook 경계

`compute_card_vector_batch`는 내부에서 `model(**inputs, ...)`를 직접 호출한다. 이 함수를 정상 생성 이전에 별도로 부른다면 CARD 추출을 위한 forward 비용이 발생한다. 따라서 논문의 실행 효율 설명을 인용할 때 실제 사용한 코드 경로를 확인해야 한다. [llava_methods.py, L144](https://github.com/Akko000/RUDDER-Residual-Update-Directed-DEcoding-Regulation-/blob/main/LLaVA/llava_methods.py#L144)

또한 이 구현은 `self_attn`의 입력과 출력을 포착하고 `after−before`를 계산한다. 일반적인 pre-norm decoder에서 self-attention 입력은 정규화된 상태이고 출력은 attention의 업데이트일 수 있다. 이 둘의 차이는 residual 블록 전체의 전후 차이와 자동으로 같지 않다. 실제 모델 클래스의 `forward`에서 tensor의 의미를 확인해야 한다. 함수명에 residual update가 들어 있다고 해서 hook 경계까지 보장되는 것은 아니다.

### 25.4 GrAInS: 이미지가 gradient 계산까지 전달되는가

확인한 VLM attribution 함수는 processor에 이미지를 전달하지만, 이후 `input_ids`를 일반 token embedding으로 바꾸고 `inputs_embeds`와 attention mask만으로 모델을 호출한다. 해당 경로에서는 processor의 image tensor나 별도의 시각 특징이 forward 인자로 전달되지 않는다. 이미지 조건부 attribution을 의도한다면 실제로 시각 정보가 사용되는지 확인해야 한다. 모델이 사전 융합한 embedding을 받는 특수 경로인지도 함께 점검해야 한다. [vlm_grad.py, L29](https://github.com/duykhuongnguyen/GrAInS/blob/main/src/attribution/gradient/vlm_grad.py#L29)

같은 함수는 마지막 입력 토큰의 ID를 마지막 위치 logits로 점수화한다. 일반적인 causal LM에서 마지막 위치 logits는 **그다음** 토큰을 예측한다. 이미 입력에 포함된 마지막 토큰의 조건부 확률을 구하려던 목적이라면 한 위치 앞 logits 또는 올바르게 shift한 응답 전체 log-likelihood가 필요하다. 목적함수에 따라 선택된 중요 토큰도 바뀔 수 있는 부분이다.

### 25.5 GrAInS: norm을 보존하는 축

확인한 steering hook은 `orig.norm(dim=1)`을 사용한다. 출력이 통상적인 `[B,T,D]`라면 `dim=1`은 토큰 축이고, 토큰별 hidden-vector norm은 마지막 축 `dim=-1`이다. 따라서 ‘각 토큰의 벡터 길이를 보존한다’는 설명과 이 구현은 일치하지 않을 수 있다. 특히 `T=1`인 decode에서는 다른 의미의 재조정이 된다. 실제 출력 shape를 확인한 뒤 의도한 축을 선택해야 한다. 같은 파일의 방향 구성도 원본−ablation 차이를 모으는 구조이므로 단순한 positive−negative 평균으로 바꾸면 다른 알고리즘이 된다. [steering.py](https://github.com/duykhuongnguyen/GrAInS/blob/main/src/utils/steering.py)

### 25.6 위 점검에서 얻을 공통 원칙

수식과 코드의 대응표를 먼저 만든다. 수식의 `h`가 코드에서 어떤 텐서인지, `t`가 어떤 위치인지, ‘평균’이 배치·토큰·레이어 중 어느 축의 평균인지 적는다. 그다음 방향의 부호, normalization, hook 위치, 모델 호출 수를 확인한다. 이 작업 없이 최종 점수만 재현하려 하면 잘못된 구현이 우연히 비슷한 숫자를 내도 알아차리기 어렵다.

<a id="sec-26"></a>

## 26. 구현할 때 특히 자주 섞이는 개념

### 26.1 Hidden state의 인덱스와 레이어 번호

일부 모델의 `hidden_states`에는 embedding 출력이 0번으로 포함된다. 이때 transformer block 0의 출력은 배열 1번일 수 있다. 저장한 활성화와 주입할 레이어가 한 칸 어긋나면 shape는 맞아도 의미는 다르다. `layer=10` 같은 숫자만 기록하지 말고 실제 모듈 경로와 ‘입력인지 출력인지’를 함께 저장한다.

### 26.2 전체 응답의 확률과 마지막 토큰 확률

응답이 `y1,…,yn`이라면 조건부 로그 확률은 `Σ log P(yi | prompt,y<i)`다. teacher forcing에서는 각 target 토큰과 그 직전 위치 logits를 맞춰야 한다. padding과 prompt 부분은 loss mask로 제외한다. 길이가 다른 응답을 비교할 때 합계와 평균 중 어느 것을 쓸지도 결과에 영향을 준다. 평균을 사용해도 응답의 정보량이나 표현 난도까지 통제되는 것은 아니므로 목적에 맞춰 명시한다.

### 26.3 Token을 없애는 것과 PAD로 바꾸는 것

토큰을 삭제하면 뒤쪽 위치와 위치 임베딩이 달라진다. PAD로 바꾸어도 attention mask를 그대로 두면 PAD가 의미 있는 입력처럼 처리될 수 있다. UNK가 없는 tokenizer도 있다. ‘ablation했다’는 한 문장 대신 어떤 대체값을 썼고 attention과 위치가 어떻게 변했는지 기록해야 한다.

### 26.4 KV cache와 개입 시점

같은 텍스트를 한 번에 forward한 결과와 cache를 쓰며 생성한 결과에서 hook이 보는 텐서 길이는 다를 수 있다. 이전 토큰의 상태는 cache에 반영되어 있고, 뒤늦게 gate 정책을 바꿔도 과거 계산이 다시 수행되는 것은 아니다. Prefill에 적용한 개입이 이후 생성에 남는 효과와, decode마다 추가하는 효과를 구분해야 한다.

### 26.5 모델 병렬화와 dtype

모든 레이어가 `model.device`에 있다고 가정하면 분산된 모델에서 문제가 생길 수 있다. 방향은 hook이 실제로 받은 텐서의 device와 dtype에 맞춘다. 단, norm 계산과 매우 작은 epsilon 처리는 float32로 수행한 뒤 원래 dtype으로 돌아오는 편이 수치적으로 안정적일 수 있다. 이는 일반적인 구현 원칙이며 특정 모델에서 최적 성능을 보장하는 설정은 아니다.

### 26.6 Hook을 제거하는 책임

실험을 반복할 때 이전 hook이 남아 있으면 steering이 중복 적용된다. baseline도 이미 수정된 모델에서 측정할 수 있다. 등록한 handle을 저장하고 `try/finally`에서 제거하는 구조가 좋다. 모듈 자체를 교체하는 구현이라면 원래 모듈 복구 여부도 확인한다. ‘α=0이면 baseline과 같은가’는 이런 누적 오류를 찾는 유용한 검증이다.

<a id="sec-27"></a>

## 27. 개입 연산을 분리해 생각하는 작은 코드 예시

아래는 이 문서에서 작성한 **개념 설명용 PyTorch 코드**다. 어떤 논문의 완전한 재현 코드도 아니며, 모델의 모듈 경로·return 형식·멀티모달 입력 처리를 대신하지 않는다. 실제 모델 실행 검증은 하지 않았다. 목적은 ‘방향 덧셈’, ‘gate’, ‘토큰별 norm 보정’을 서로 분리해 보여주는 것이다.

```python
import torch

def add_direction(h, direction, strength, gate=None, keep_norm=False):
    # h: [B, T, D]
    # direction: [D], [B, 1, D], 또는 [B, T, D]
    # gate: None 또는 [B, T, 1]
    if h.ndim != 3:
        raise ValueError("h must have shape [B, T, D]")
    if direction.shape[-1] != h.shape[-1]:
        raise ValueError("hidden dimension mismatch")
    if direction.ndim not in (1, 3):
        raise ValueError("direction must be [D] or a 3-D tensor")

    base = h.float()
    v = direction.to(device=h.device, dtype=torch.float32)
    delta = float(strength) * v
    if gate is not None:
        if gate.shape != (*h.shape[:2], 1):
            raise ValueError("gate must have shape [B, T, 1]")
        delta = delta * gate.to(device=h.device, dtype=torch.float32)

    changed = base + delta
    if keep_norm:
        old_norm = torch.linalg.vector_norm(base, dim=-1, keepdim=True)
        new_norm = torch.linalg.vector_norm(changed, dim=-1, keepdim=True)
        scaled = changed * (old_norm / new_norm.clamp_min(1e-8))
        # 변경 후 벡터가 거의 0이면 방향이 정의되지 않으므로 원본 사용
        changed = torch.where(new_norm > 1e-8, scaled, base)
    return changed.to(dtype=h.dtype)
```

여기서 gate가 0인 위치는 개입하지 않는다. gate의 계산은 함수 밖에서 해야 한다. 그 계산이 위험 점수인지, anchor 신뢰도인지, 단순한 답변 마스크인지는 호출자가 정한다. `keep_norm=True`도 목표 과제에 따라 비교해야 하는 선택지다.

모델에 연결할 때는 세부 구현을 생략한 다음 순서를 따른다.

```text
1. 수정하려는 모듈의 실제 출력 shape와 반환 형식을 확인한다.
2. 원래 출력을 저장하고 intervention 함수를 별도로 적용한다.
3. 수정한 텐서 외의 cache·attention 등 반환 요소는 보존한다.
4. baseline, strength=0, 작은 양수, 작은 음수 조건을 비교한다.
5. 실험 종료나 예외 발생 시 hook을 제거한다.
```

처음부터 복잡한 위험 gate와 여러 레이어를 연결하기보다, 한 레이어·한 입력·한 토큰에서 실제 변화량이 의도와 같은지 확인하면 디버깅이 쉬워진다. 이후 batch, padding, cache, 여러 이미지로 범위를 넓힌다.

<a id="sec-28"></a>

## 28. 제대로 비교하려면 무엇을 고정해야 하는가

방법마다 다른 모델·프롬프트·데이터셋으로 보고된 숫자를 모아 순위를 매기면 공정한 비교가 되지 않는다. 연구를 시작한다면 하나의 공통 실험 틀 안에서 비교하는 편이 좋다.

| 항목 | 고정하거나 기록할 내용 | 이유 |
|---|---|---|
| 모델 | 정확한 checkpoint, revision, dtype | 같은 이름의 모델도 세부 구현이 다를 수 있음 |
| 입력 처리 | 이미지 크기·crop·chat template | 시각 정보와 토큰 위치가 달라짐 |
| 데이터 | 학습·보정·평가 분할, 중복 여부 | 방향 학습과 평가의 누수 방지 |
| 생성 | greedy/sampling, seed, 길이 제한 | 오류율과 유용성의 공통 비교 |
| 개입 위치 | 모듈 경로, 레이어, 토큰 마스크 | 같은 수식이라도 실제 연산이 달라짐 |
| 개입량 | norm, 상대 norm, 빈도, 선택 수 | 작은 개입과 큰 개입을 구분 |
| 비용 | 준비 시간, prefill, decode, 메모리 | offline 비용과 실시간 비용 분리 |
| 평가 | 사실성·정답률·정보량·부작용 | 한 지표 개선의 착시 방지 |

데이터는 최소한 세 역할로 분리한다. **방향 구성용 자료**는 벡터·마스크·probe를 만든다. **개발·보정 자료**는 레이어와 강도, threshold를 고른다. **최종 평가 자료**는 모든 선택이 끝난 뒤 한 번 정해진 설정을 측정한다. 규모가 작아도 역할을 구분하는 편이 같은 자료로 반복 최적화하는 것보다 낫다.

층과 강도를 많이 탐색할수록 우연히 좋은 설정을 고를 가능성이 커진다. 최종 점수만 보고하지 말고 탐색한 후보 수, 선택 기준, 평균과 변동성을 함께 기록한다. 모델이 여러 개라면 한 모델에서 고른 설정을 다른 모델에 그대로 적용했는지, 각 모델에 맞춰 다시 보정했는지도 분명히 한다.

### 28.1 필요한 대조군

개입 없음은 기본이다. 여기에 같은 norm의 무작위 방향, 목표와 반대 방향, 레이어만 바꾼 조건, 같은 개입 빈도의 무작위 gate를 추가하면 해석력이 커진다. 다만 모든 연구에서 모든 대조군이 필요한 것은 아니다. 자신의 핵심 주장에 직접 대응하는 것을 우선한다.

예를 들어 ‘위험한 토큰을 잘 골라서 좋아졌다’는 주장에는 무작위로 같은 수의 토큰을 고른 대조군이 유용하다. ‘희소한 좌표가 더 정밀하다’는 주장에는 같은 개수의 무작위 좌표와 같은 총 변화량의 dense 개입이 유용하다. ‘이미지별 방향이 중요하다’는 주장에는 다른 이미지에서 얻은 방향을 바꾸어 넣는 대조군이 유용하다.

### 28.2 평가 지표를 함께 해석하기

환각 감소, 객체 recall, 과제 정답률, 응답 길이, 정보량은 별도로 기록한다. ‘모르겠다’만 말하는 모델은 일부 환각 지표에서 좋아 보일 수 있지만 목표 과제를 수행하지 못한다. 반대로 정답 핵심은 맞아도 불필요한 세부가 틀릴 수 있다.

가능하다면 예제를 맞음→맞음, 맞음→틀림, 틀림→맞음, 틀림→틀림으로 나누어 본다. 전체 평균이 조금 좋아져도 원래 잘하던 영역을 크게 망가뜨렸다면 적용 범위를 좁혀야 할 수 있다. 이 분해는 새로운 gate를 설계할 때도 유용하다.

### 28.3 비용은 실제 호출 단위로 측정하기

준비 단계의 gradient·PCA·probe 학습 비용과, 새 입력마다 필요한 CARD 추출·lens 실행·gate 계산 비용을 구분한다. 같은 ‘추론 중 개입’이라도 추가 forward가 있는지에 따라 첫 응답 지연이 달라진다. 전체 tokens/sec만으로는 첫 토큰까지의 지연이나 짧은 응답의 비용을 놓칠 수 있다.

보고 항목은 준비 비용, 입력당 추가 forward 수, 첫 토큰 지연, 이후 토큰 처리량, 최대 GPU 메모리로 나눌 수 있다. 절대 수치뿐 아니라 같은 장치의 baseline 대비 비율을 같이 적으면 비교하기 쉽다.

<a id="sec-29"></a>

## 29. 이 자료들을 연결해 볼 만한 연구 질문

이 절은 **새 실험 제안**이다. 기존 논문에서 이미 성능이 입증되었다는 뜻도, 선행연구가 없는 새 아이디어라는 뜻도 아니다. 실제 연구로 발전시키기 전에는 더 넓은 문헌 조사가 필요하다.

### 29.1 원래 AU·뉴런과 학습된 feature는 언제 다른가

Goodfire의 관점, AUSteer의 작은 단위 개입, transcoder 기반 회로 분석을 한 실험에서 비교할 수 있다. 같은 목표 행동에 대해 원래 좌표 선택과 학습된 feature 선택을 비교한다. 설명의 예쁨보다 독립 자료의 활성화 예측, causal intervention 효과, 부작용, 계산비를 측정한다.

학습된 feature가 의미적으로 선명하지만 재구성 오차가 크다면 원래 모델에서의 효과가 약할 수 있다. 반대로 이름 붙이기 어려운 원래 좌표가 행동 제어에는 효율적일 수도 있다. 이 차이를 측정하면 ‘해석하기 쉬운 단위’와 ‘제어하기 좋은 단위’가 같은지 답할 수 있다.

### 29.2 방향과 gate를 교차시키면 무엇이 드러나는가

고정 대조 방향, prior를 제거한 방향, 입력별 anchor를 준비한다. 각 방향에 상시 개입, 위험 gate, 신뢰 gate를 교차 적용한다. 모든 조합이 원래 논문에 해당하는 것은 아니므로 새 조합임을 명시한다.

핵심 결과는 최고의 조합 하나보다 상호작용이다. 고정 방향은 위험 gate에서 좋지만 입력별 anchor는 신뢰 gate에서 좋을 수 있다. 이는 gate가 독립적으로 ‘좋은 것’이 아니라 방향의 의미와 함께 설계되어야 한다는 증거가 된다.

### 29.3 Lens가 제안한 의미는 개입으로 검증되는가

LatentLens와 J-lens가 비슷한 의미를 가리키는 위치를 후보로 고른다. 해당 표현을 지우거나 다른 이미지의 표현으로 바꾸었을 때 관련 답변이 선택적으로 변하는지 본다. Lens 출력만으로 후보를 확정하지 않고 원래 모델의 결과로 검증한다.

평가 예제는 후보 탐색에 쓰지 않은 것으로 고른다. 설명을 본 뒤 같은 사례에서만 성공하는 개입을 찾으면 과적합될 수 있다. 실패 사례도 기록하면 lens의 의미적 유사성과 실제 causal relevance가 어디서 갈라지는지 알 수 있다.

### 29.4 Attribution으로 CARD pooling을 바꾸면 도움이 되는가

단순 평균이나 norm 기반 가중치 대신, 목표 출력에 관련된 위치를 더 크게 가중해 입력별 anchor를 만들어 볼 수 있다. 하지만 gradient나 회로 추적을 매 입력마다 추가하면 계산비가 커진다. 정확도 증가와 지연 증가를 같이 평가해야 한다.

이 실험은 먼저 작은 offline 분석으로 진행하는 편이 합리적이다. 어떤 위치를 강조해야 좋은지 확인한 뒤, 그 선택을 가벼운 proxy로 근사할 수 있는지 본다. 비싼 분석 도구와 실시간 개입 정책을 분리하는 설계다.

### 29.5 사용자 성향을 바꾸면 grounding 개입의 효과도 달라지는가

TraitBasis식 평가 사고를 VLM 대화에 적용할 수 있다. 중립적인 질문, 성급한 단답 요구, 잘못된 전제를 가진 질문, 반복 확인을 요구하는 질문에서 같은 grounding 개입을 비교한다. 사용자의 문체 때문에 모델이 시각 근거를 덜 쓰는지, 불확실성을 말할 기회를 잃는지 살핀다.

여기서는 성향 표현과 과제 난도를 분리해야 한다. 질문을 복잡하게 바꾸어 성능이 떨어졌는데 이를 성향 때문이라고 해석하면 안 된다. 같은 의미의 과제를 여러 대화 스타일로 표현하고, 시뮬레이터의 실제 행동도 별도로 검증한다.

<a id="sec-30"></a>

## 30. 직접 공부하거나 재현할 때의 진행 순서

첫 단계는 한 모델의 계산 위치를 명확히 하는 것이다. hidden state, MLP 출력, attention 출력의 shape를 기록하고 baseline을 고정한다. 그다음 가장 단순한 고정 방향 개입을 구현해 강도에 따른 행동 변화와 부작용을 본다. 복잡한 방법을 구현하기 전에 이런 관측 장치가 있어야 결과를 해석할 수 있다.

두 번째 단계에서는 방향 추출을 바꾼다. 평균 차이, PCA, prior 제거, attribution 기반 선택 중 하나씩 비교한다. 이때 주입 위치·gate·평가 설정을 유지한다. 방향의 차이만 드러나도록 하기 위해서다.

세 번째 단계에서는 gate와 sparse selection을 추가한다. 평균 개입량과 개입 빈도를 기록한다. 점수만 좋아졌는지, 원래 맞던 예제를 덜 망가뜨리는지 확인한다. 입력별 anchor도 이 단계에서 고정 방향과 비교할 수 있다.

네 번째 단계에서는 lens와 회로 분석을 연결한다. 성공 예제뿐 아니라 steering이 실패하거나 정답을 망가뜨린 예제를 고른다. 무엇을 읽을 수 있는지, 어느 경로에서 정보가 바뀌는지, 왜 gate가 잘못 켜졌는지 분석한다. 해석 도구를 보기 좋은 그림을 만드는 데만 쓰지 않고 실패 원인을 설명하는 데 사용한다.

마지막으로 모델·과제·질문 스타일을 바꾼다. 이때 처음부터 모든 조건을 다시 조정한 결과와, 기존 설정을 고정한 채 옮긴 결과를 구분한다. 재보정하면 잘 작동한다는 결론과 그대로 일반화된다는 결론은 다르다.

<a id="sec-31"></a>

## 31. 용어를 다시 찾기 위한 사전

| 용어 | 이 문서에서의 뜻 |
|---|---|
| Activation | 특정 입력을 처리하면서 생긴 중간 계산값 |
| Residual stream | 여러 블록의 업데이트가 더해지는 상태 흐름 |
| Steering vector | 내부 상태를 움직이는 데 사용하는 방향 |
| Contrastive pair | 목표 속성이 다르도록 구성한 대조 사례 쌍 |
| Mean difference | 대조 사례의 활성화 차이를 평균한 방향 |
| PCA | 데이터 변동이 큰 축을 찾는 방법; 평균 차이와 동일하지 않음 |
| Probe | 내부 표현에서 특정 속성·오류 등을 예측하는 보조 모델 |
| Gate | 개입 여부나 개입량을 상태에 따라 조절하는 함수 |
| Calibration | 별도 자료로 threshold·신뢰성 등을 보정하는 과정 |
| Ablation | 입력·활성화의 일부를 제거하거나 대체하는 개입 |
| Activation patching | 한 실행의 내부 상태를 다른 실행에 넣는 개입 |
| Attribution | 정해진 목표에 대한 입력·내부 요소의 기여 추정 |
| Jacobian | 여러 출력의 여러 입력에 대한 미분을 모은 행렬 |
| J-lens | 평균적인 downstream 미분 관계를 이용하는 표현 읽기 |
| Logit lens | 중간 상태를 어휘 출력 장치에 연결해 읽는 계열 |
| LatentLens | 문맥화된 텍스트 표현을 참조해 시각 표현을 읽는 방법 |
| SAE feature | 희소 표현 학습으로 얻은 특징 단위 |
| Transcoder | MLP 입력에서 출력 기여를 근사하는 해석용 모델 |
| CLT | 여러 레이어에 걸친 기여를 표현하는 cross-layer transcoder |
| Replacement model | 원래 계산 일부를 해석 가능한 근사로 바꾼 모델 |
| Error node | 근사가 설명하지 못한 차이를 그래프에 반영하는 요소 |
| Grounding | 답변이 실제 입력의 근거와 연결되는 성질 |
| Hallucination | 평가 기준상 근거 없거나 사실과 다른 생성 내용 |
| Prior | 현재 관측 외에 학습된 연상·분포에서 오는 경향 |
| AU | 선형 연산에서 활성화 스칼라와 열 벡터의 기여 단위 |
| Prefill | 입력 문맥을 처리해 생성의 출발 상태를 만드는 단계 |
| Decode | 문맥에 토큰을 추가하며 다음 토큰을 생성하는 단계 |
| KV cache | 이전 토큰의 attention 계산 일부를 재사용하는 저장값 |

<a id="sec-32"></a>

## 32. 자료를 다시 읽을 때 남겨 둘 질문

새 논문이나 코드를 읽을 때 다음 항목으로 한 페이지 기록을 만들면 이번 자료 묶음에 자연스럽게 이어 붙일 수 있다.

```text
연구가 해결하려는 구체적인 실패:
사용한 모델과 데이터:
관측한 내부 텐서와 실제 모듈 경로:
방향·마스크·gate를 만드는 자료:
고정된 요소와 입력마다 달라지는 요소:
주입 위치와 prefill/decode 적용 범위:
평가에서 좋아진 것과 나빠진 것:
효과를 설명하는 가장 강한 개입 증거:
offline 비용과 입력당 추가 비용:
논문 설명과 코드에서 확인한 차이:
아직 확인하지 못한 가정:
```

이 자료 묶음의 핵심적인 연구 기회는 표현을 읽는 도구, 계산을 추적하는 도구, 행동을 바꾸는 도구를 같은 예제와 같은 평가 기준 위에 놓는 데 있다. 읽힌 의미가 실제 개입 효과를 예측하는지, 개입이 성공한 이유를 경로 분석이 설명하는지, 그 설명이 새로운 입력에서도 유지되는지를 차례로 묻는 것이다. 이 연결이 확인될수록 단순한 성능 조절을 넘어 어떤 내부 계산이 어떤 행동을 만드는지에 대한 설명이 강해진다.

---

**검토 범위 기록:** 관련 연구 자료를 종합하고, 핵심 방법의 원문과 위에 연결한 공개 코드 파일을 대조했다. 숫자가 서로 다른 데이터셋·버전·모델에서 나오는 경우 단일 성능 순위로 합치지 않았다. 코드 관련 지적은 정적 검토 범위이며 실제 모델 실행·GPU benchmark·논문 재현 결과는 포함하지 않는다.
