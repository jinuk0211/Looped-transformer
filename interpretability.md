# Neural Networks Are Already Clean - 2026/10/16
<img width="680" height="357" alt="image" src="https://github.com/user-attachments/assets/603f17d4-49a3-459b-aeb8-fcc7d536305a" />

현대의 AI 모델 interpretability에 대한 재밌는 의견이 나와서 공유합니다. 최근 alignment 이슈적으로 Interpretability가 점점 더 중요해지고 있는데 기존 anthropic의 superposition과 상반되는 의견이라 흥미롭게 읽었습니다. 개인적으로 LLM이 black-box라는 말에 전혀 동의하지 않는 사람중에 한명으로 SAE, transcoder, jacobian lens, activation patching, attribution graph 여러 Interpretability 방식들이 계속 발전하고 있고 이를 통해 LLM과 멀티모달 모델은 상당부분 해석가능할 것이다라는 의견을 제시합니다

**뉴런의 역할, 중첩(Superposition), 그리고 Qwen 분석 사례**
==========

AI 모델은 답을 만드는 동안 내부에서 무엇을 계산할까요? 이 글은 Goodfire의 Nick Cammarata가 쓴 **[Neural Networks Are Already Clean](https://already-clean.goodfire.com/)**을 바탕으로, 뉴런의 역할을 어디까지 이해할 수 있는지 정리합니다.

저자는 우리가 복잡하게 뒤섞여 있다고 생각했던 뉴런 중 상당수가, 자세히 보면 설명 가능한 역할을 한다고 주장합니다. 이를 뒷받침하는 요인으로 **관찰 도구의 발전, 모델 구조와 학습 방식의 개선, 모델 크기의 증가**를 함께 제시합니다.

원문은 저자의 연구 결과와 가설을 담은 에세이입니다. 이 글에서는 그 주장과 배경 개념을 구분해 소개합니다.

이와 함께 Anthropic의 2024~2026년 연구 그림을 통해 **개념의 표현, 추론 회로, 내부 표현에 대한 개입**의 실제 사례를 살펴봅니다. Anthropic의 그림은 Claude를 분석한 결과이며, 아래의 Goodfire·Qwen 분석과는 구분합니다.

![Anthropic이 추출한 내적 갈등 특징과 주변의 관련 특징들을 보여주는 지도](https://www-cdn.anthropic.com/images/4zrzovbb/website/fe4d42c004bf43efda0f5921adfedd2f8f42e417-2200x2140.png)

*그림 1. ‘내적 갈등’ 주변에는 감정과 이성, 상충하는 이해관계, 딜레마 등에 관련된 특징들이 모여 있습니다. 각 점은 추출한 특징을 나타내며, 개별 뉴런의 위치나 연결을 그린 것은 아닙니다. [출처: Anthropic, Mapping the mind of a large language model (2024)](https://www.anthropic.com/research/mapping-mind-language-model)*

## 목차

- [1. 뉴런을 이해한다는 것은 무슨 뜻일까?](#meaning)
- [2. 그림으로 보는 고양이와 자동차 사례](#images)
- [3. 도구가 좋아진 걸까, 모델이 좋아진 걸까?](#progress)
- [4. Qwen도 대부분 해석할 수 있을까?](#qwen)
- [5. 중첩과 SAE에 대해 무엇을 다시 생각해야 할까?](#superposition)
- [6. 어디까지 이해했고, 무엇이 남았을까?](#limits)
- [참고 자료](#references)

<a id="meaning"></a>

## 1. 뉴런을 이해한다는 것은 무슨 뜻일까?

신경망의 **뉴런**은 입력을 받아 계산하고, 그 결과를 숫자로 다음 단계에 전달합니다. 이 숫자를 **활성화 값**이라고 합니다. 뉴런들이 연결되어 특정 계산을 함께 수행하는 부분은 **Circuit**라고 부릅니다.

뉴런의 계산식을 안다고 해서 역할까지 아는 것은 아닙니다. 예를 들어 어떤 뉴런이 고양이 사진에 반응해도, 고양이라는 동물을 알아보는지, 귀의 모양이나 털의 질감에 반응하는지는 더 살펴봐야 합니다.

**Mechanistic interpretability** 연구는 이런 역할을 찾아가는 작업입니다. 어떤 입력에 반응하는지 관찰하고, 연결된 뉴런을 살펴보고, 값을 바꿨을 때 모델의 답이 어떻게 달라지는지 실험합니다.

원문에서 말하는 **clean**은 뉴런의 반응을 공통된 패턴이나 역할로 설명할 수 있다는 뜻에 가깝습니다. 모든 뉴런이 정확히 하나의 단어나 개념만 담당한다는 뜻은 아닙니다. [원문](https://already-clean.goodfire.com/)

회로를 추적하면 여러 개념이 답변으로 이어지는 과정도 살펴볼 수 있습니다. Anthropic은 Claude 3.5 Haiku에 “댈러스가 속한 주의 주도는?”이라고 물었을 때, **댈러스 → 텍사스 → 오스틴**에 해당하는 중간 단계를 관찰했습니다.

![댈러스에서 텍사스를 거쳐 오스틴이라는 답변으로 이어지는 추론 회로](https://www-cdn.anthropic.com/images/4zrzovbb/website/fd2e125879ab993949017e03e3465a12fda884bf-1650x857.png)

*그림 2. 질문에 포함된 ‘댈러스’, ‘주’, ‘주도’의 정보가 중간 특징을 거쳐 답변에 기여하는 경로를 단순화한 도식입니다. 모델 전체의 계산을 모두 그린 것은 아닙니다. [출처: Anthropic, Tracing the thoughts of a large language model (2025)](https://www.anthropic.com/research/tracing-thoughts-language-model)*

<a id="images"></a>

## 2. 그림으로 보는 고양이와 자동차 사례

### 고양이와 자동차에 함께 반응하는 이유

InceptionV1이라는 이미지 모델에는 고양이 얼굴과 자동차 앞부분에 함께 반응하는 뉴런이 있습니다. 서로 다른 대상에 반응한다는 이유로, 한 뉴런에 여러 의미가 섞인 대표 사례로 알려졌습니다.

그런데 저자는 이 뉴런을 다시 살펴보고 다른 설명을 제시합니다. **고양이와 자동차가 공유하는 뾰족한 V자 형태를 감지했을 수 있다**는 것입니다.

| 여러 입력에서 찾은 공통된 뾰족한 모양 | 뉴런의 값이 크게 음수가 될 때 나타나는 둥근 모양 |
| :---: | :---: |
| ![고양이·자동차·여우 입력이 공유하는 뾰족한 형태를 보여주는 생성 이미지](https://already-clean.goodfire.com/assets/v-figure-10x/gen-pointed-old.webp) | ![같은 뉴런의 음수 방향을 보여주는 둥근 생성 이미지](https://already-clean.goodfire.com/assets/v-figure-10x/gen-round-old.webp) |

*그림 3. 왼쪽은 `4e:55` 뉴런으로 들어오는 고양이·자동차·여우 입력의 공통점을 드러내도록 만든 이미지입니다. 오른쪽은 같은 뉴런의 값이 크게 음수가 되도록 만든 이미지입니다. 실제 사진이 아니라 뉴런의 반응을 살펴보기 위해 생성한 그림입니다. [출처: Goodfire](https://already-clean.goodfire.com/)*

고양이와 자동차를 **대상의 이름**으로 보면 서로 다릅니다. 하지만 **형태**로 보면 공통점이 있을 수 있습니다. 저자는 이처럼 우리가 뉴런을 분류하는 방식 때문에 역할을 놓쳤을 가능성을 제기합니다.

### 같은 뉴런도 그림을 만드는 방법에 따라 다르게 보인다

뉴런이 무엇에 반응하는지 알아보는 방법 중 하나는, 그 뉴런을 강하게 반응시키는 이미지를 만드는 것입니다. 이를 **특징 시각화(Feature visualization)**라고 합니다.

다음은 같은 뉴런 `4d:417`을 서로 다른 조건에서 시각화한 결과입니다.

| 기본 방법으로 만든 이미지 | 자동차 관련 입력 경로를 반영해 만든 이미지 |
| :---: | :---: |
| <img src="https://already-clean.goodfire.com/assets/superposition-circuit/dream02/4d417.png" alt="기본 방법으로 시각화했을 때 고양이처럼 보이는 4d:417 뉴런" width="200"> | <img src="https://already-clean.goodfire.com/assets/superposition-circuit/domain60_dream02/417-car.jpg" alt="자동차 관련 경로를 반영했을 때 자동차처럼 보이는 같은 뉴런" width="200"> |

*그림 4. 뉴런은 같지만, 이미지를 만드는 조건이 달라지면 보이는 대상도 달라집니다. 두 그림 모두 실제 사진이 아닌 생성 이미지입니다. [출처: Goodfire](https://already-clean.goodfire.com/)*

따라서 생성 이미지 하나만 보고 “이것은 고양이 뉴런이다”라고 단정하기는 어렵습니다. 실제 사진 중 무엇에 반응하는지, 어떤 뉴런과 연결되는지 함께 확인해야 합니다. 이미지 생성 과정에도 결과를 바꾸는 여러 설정이 있기 때문입니다. [특징 시각화 설명](https://distill.pub/2017/feature-visualization/)

<a id="progress"></a>

## 3. 도구가 좋아진 걸까, 모델이 좋아진 걸까?

원문은 **세 가지 변화가 모두 해석을 돕는다**고 주장합니다.

| 변화 | 무엇이 달라지는가? |
| --- | --- |
| **관찰 도구의 발전** | 같은 모델을 봐도, 예전에는 놓쳤던 뉴런의 역할을 더 잘 찾는다. |
| **모델 구조와 학습 방식의 개선** | 모델 자체가 더 잘 학습하면서 뉴런의 역할도 더 분명해질 수 있다. |
| **모델 크기의 증가** | 한 뉴런이 넓게 담당하던 역할이 여러 뉴런의 세부 역할로 나뉠 수 있다. |

이 차이는 **크기를 늘리지 않은 비교**에서 잘 드러납니다. 저자가 제시한 마지막 층의 대략적인 해석 가능 비율은 다음과 같습니다.

| 비교 조건 | 저자가 추정한 비율 |
| --- | ---: |
| 기존 InceptionV1을 당시 도구로 분석 | 약 0~5% |
| **같은 모델**을 개선된 도구로 분석 | 약 50% |
| 같은 구조를 현대적인 학습 설정으로 다시 훈련 | 약 70% |
| **같은 규모와 데이터**로 현대적인 ConvNeXt 모델을 훈련 | 약 99% |

수치는 저자의 평가이며, 각 개선의 효과를 정확히 몇 %씩 분리한 측정은 아닙니다. 다만 **모델 크기만으로 차이를 설명할 수는 없다**는 점을 보여줍니다. [원문](https://already-clean.goodfire.com/)

크기가 커지는 효과도 있습니다. 저자의 설명에 따르면, 작은 모델에서 “자동차 전반”을 맡던 뉴런의 역할이 큰 모델에서는 여러 자동차 종류나 자세를 담당하는 역할로 나뉠 수 있습니다. 역할이 구체적이면 연구자도 패턴을 알아보기 쉬워집니다. 하지만 조사할 뉴런의 수 역시 늘어나므로, 모델 전체를 이해하는 작업량까지 줄어든다는 뜻은 아닙니다.

<a id="qwen"></a>

## 4. Qwen도 대부분 해석할 수 있을까?

저자는 이미지 모델뿐 아니라 **현대 언어 모델에서도 뉴런의 많은 부분을 해석할 수 있다**고 주장합니다.

Qwen3-32B의 초기 조사에서는 다음과 같이 추정했습니다.

| 조사 결과 | 대략적인 비율 |
| --- | ---: |
| 주된 역할 하나로 설명됨 | 약 90% |
| 서로 관련된 역할 두 개로 설명됨 | 약 7% |
| 아직 이해하지 못함 | 약 2~4% |

여기서 센 것은 **뉴런 하나가 아니라, 뉴런이 큰 양수 값을 보이는 경우와 큰 음수 값을 보이는 경우 각각**입니다. 원문은 이를 **뉴런의 양·음 극(neuron poles)**이라고 부릅니다.

즉, “조사한 뉴런의 반응 대부분에 설명을 붙일 수 있었다”는 강한 주장입니다. 다만 **모델의 모든 답변과 행동을 이해했다는 뜻은 아닙니다.** 약한 반응까지 어디까지 포함하는지, 비슷한 역할을 하나로 묶는지에 따라 수치도 달라집니다. [원문](https://already-clean.goodfire.com/)

원문에는 **Qwen3.6**에서 마침표 출력을 억제하는 뉴런 등을 분석한 사례도 나옵니다. 위 비율은 Qwen3-32B에 관한 것이므로 두 모델의 결과를 구분해야 합니다. 조 단위 파라미터 모델에서도 일부 회로를 제시하지만, 전체 조사는 아직 하지 않았다고 밝힙니다.

### 이름 인식: 문자 조각에서 ‘Miyazaki’까지

Qwen3-32B에서는 초기 층의 문자 조각과 철자 패턴이 **‘Miyazaki’를 인식하는 뉴런**으로 모이고, 뒤쪽에서는 《센과 치히로의 행방불명》과 《바람계곡의 나우시카》 같은 연관 표현으로 이어집니다.

<img src="https://github.com/user-attachments/assets/a798e6a6-16cd-4bf3-87f1-a67b184221c0" alt="Qwen3-32B에서 문자 조각과 철자 패턴이 Miyazaki 뉴런과 작품 관련 뉴런으로 이어지는 회로" width="1040">

*그림 5. Goodfire 원문의 Qwen3-32B 이름 인식 회로를 캡처한 화면입니다. L0–L1의 문자 조각, L2–L3의 패턴, L4의 이름 인식, L6–L8의 연관 표현을 함께 보여줍니다. 연결 옆 숫자는 활성화 값이 아니라 연결 강도입니다. [출처: Goodfire — From spelling fragments to “Miyazaki”](https://already-clean.goodfire.com/#miyazaki-circuit)*

### 추론: ‘라이온 킹’ 단서에서 ‘Scar’라는 답으로

다음은 이름을 직접 알려주지 않고 ‘프라이드 랜드의 왕이 된 사자 새끼의 숙적’을 묻는 사례입니다. Qwen3-32B는 **라이온 킹과 관련된 정보와 적대 관계 단서**를 이용해 ‘Scar’를 답합니다.

<img src="https://github.com/user-attachments/assets/a6403c8b-dcb7-4531-9ec2-2b481a2eaec9" alt="Qwen3-32B가 라이온 킹과 숙적에 관한 단서를 이용해 Scar를 답하는 추론 회로" width="1040">

*그림 6. 왼쪽은 질문과 답변, 오른쪽은 Qwen3-32B의 MLP 뉴런을 묶어 표시한 회로입니다. 상자의 작은 숫자는 해당 그룹의 뉴런 수입니다. 원문 도식의 두 번째 사례를 캡처했습니다. [출처: Goodfire — Who stole the book?](https://already-clean.goodfire.com/#preeclampsia-circuit)*

### 기능 검증: 마침표에 반응하지만 다음 마침표는 억제한다

Qwen3.6의 이 뉴런은 여러 문맥의 마침표에서 활성화되지만, 출력에서는 다음 마침표의 가능성을 낮춥니다. **무엇에 반응하는지와 무엇을 하게 만드는지를 함께 봐야 한다**는 사례입니다.

<img src="https://github.com/user-attachments/assets/9849bb82-f48c-4bff-88e7-9dfdd09775ba" alt="왼쪽은 기존 연구의 뉴런이고 오른쪽은 Qwen3.6의 마침표 억제 뉴런인 비교 화면" width="1040">

*그림 7. 오른쪽이 Qwen3.6 분석입니다. 위는 활성화 사례, 아래는 출력을 억제하는 토큰입니다. 원문은 강하게 활성화되는 30개 테스트에서 이 뉴런을 제거하자 모두 다음 마침표의 가능성이 높아졌다고 보고합니다. 왼쪽은 비교 대상으로 제시된 Towards Monosemanticity 연구의 뉴런입니다. [출처: Goodfire — Don’t Say Dot](https://already-clean.goodfire.com/#period-neuron-figure)*

### 반응한 위치만 보면 역할을 잘못 짚을 수 있다

어떤 뉴런이 쉼표에서 반응한다고 해서 쉼표 자체를 담당한다고 볼 수는 없습니다. 그 위치에서 앞 문장의 내용을 정리해 다음 계산으로 넘길 수도 있습니다. 문장 전체의 맥락과, 그 뉴런이 다음 계산에 미치는 영향을 함께 봐야 합니다.

<details>
<summary><strong>더 알아보기: 양수와 음수를 왜 따로 볼까?</strong></summary>

뉴런은 숫자로 정보를 전달합니다. 큰 양수와 큰 음수는 다음 단계의 계산에 서로 다른 영향을 줄 수 있습니다. 따라서 양수 값이 큰 사례만 모으면 뉴런의 역할을 일부 놓칠 수 있습니다.

SwiGLU라는 계산 방식에서는 두 경로에서 나온 값을 곱합니다. 양수에 양수를 곱해도 양수가 되고, 음수에 음수를 곱해도 양수가 됩니다. **최종 결과의 부호가 같아도 내부에서 계산한 과정은 다를 수 있습니다.** 원문은 Qwen3.6의 일부 뉴런에서 이 차이를 나눠 보면 반응을 더 잘 이해할 수 있다고 설명합니다.

또 음수라는 이유만으로 특정 개념을 “억제한다”고 판단해서는 안 됩니다. 다음 단계의 연결 가중치가 그 값에 다시 곱해지기 때문에, 최종적으로 어떤 효과를 내는지 확인해야 합니다.

참고: [Goodfire 원문](https://already-clean.goodfire.com/), [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)

</details>

<a id="superposition"></a>

## 5. 중첩과 SAE에 대해 무엇을 다시 생각해야 할까?

### 한 뉴런이 여러 대상에 반응하는 이유

한 뉴런이 서로 무관해 보이는 여러 대상에 반응하는 현상을 **다의성(Polysemanticity)**이라고 합니다. 그 이유를 설명하는 중요한 가설 중 하나가 **중첩(Superposition)**입니다.

중첩은 **표현 공간의 차원 수보다 많은 특징을 서로 겹치는 방향에 담는 방식**입니다. 내부 상태를 기록하는 숫자가 100개라면, 각 특징에 숫자 하나씩을 배정하는 대신 여러 숫자의 조합으로 100개보다 많은 특징을 표현하는 셈입니다. 동시에 나타나는 특징이 적으면 서로 간섭을 줄일 수 있습니다. 단순화된 모델에서 이런 현상이 학습되는 사례도 있습니다. [Toy Models of Superposition](https://arxiv.org/abs/2209.10652)

Goodfire의 저자는 특히 **서로 관계없는 기능이 한 뉴런에 임의로 섞였다는 설명**에 의문을 제기합니다. 실제로 다시 관찰해 보니 공통된 형태나 의미를 찾을 수 있었던 경우가 많다는 것입니다.

### 관련된 패턴을 따라 역할이 넓어질 수도 있다

저자가 제안하는 설명 중 하나는 **재귀적 일반화(Recursive generalization)**입니다. 어려운 이름이지만, 관련된 패턴으로 조금씩 역할을 넓혀 간다는 뜻입니다.

```text
iy → diy → do it yourself
```

`iy`라는 문자 조합에 반응하던 뉴런이 `diy`에 반응하고, 이어서 같은 뜻인 `do it yourself`까지 반응하게 됐다고 생각해 볼 수 있습니다. 마지막 표현에는 `iy`가 없지만, 중간 단계를 따라가면 연결 이유를 이해할 수 있습니다. 이 흐름은 원문 예시를 단순화한 개념도입니다. [원문](https://already-clean.goodfire.com/)

공간이 부족한 모델에서도 저자는 비슷한 설명을 제시합니다. 한 층에서 사용하는 **채널**을 480개에서 10개로 줄여 다시 학습시켰습니다. 채널 하나는 이미지의 여러 위치에서 같은 계산을 맡습니다. 이 실험에서는 여러 곡선을 따로 구분하던 역할이 “곡선 전반”을 묶어서 다루는 식으로 넓어졌다고 보고합니다.

### SAE는 무엇을 하는 도구인가?

**SAE(Sparse Autoencoder, 희소 오토인코더)**는 모델 안의 복잡한 숫자 신호를 몇 가지 특징의 조합으로 나눠 보려는 도구입니다. 많은 후보 특징 중 일부만 사용해 원래 신호를 다시 만들도록 학습합니다.

Anthropic은 512개 뉴런으로 이루어진 층에서 4,000개 이상의 특징을 추출하고, DNA 서열이나 법률 언어와 관련된 특징을 보고했습니다. [Towards Monosemanticity](https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning)

2024년에는 이 접근을 Claude 3 Sonnet으로 확장했습니다. 아래의 **Golden bridge feature**은 영어뿐 아니라 한국어 등 여러 언어의 관련 표현과 golden bridge 이미지에도 반응합니다. 여기서 말하는 특징은 SAE로 추출한 활성화 패턴이며, ‘golden bridge만 담당하는 뉴런 하나’를 뜻하지 않습니다.

![golden bridge 이미지와 한국어를 포함한 여러 언어의 golden bridge 표현에 반응하는 특징](https://www-cdn.anthropic.com/images/4zrzovbb/website/c896a301ad3d1ef4237cb05b68d78b467c444097-2200x1284.png)

*그림 8. 주황색은 텍스트에서 golden bridge 특징이 활성화되는 위치를 표시합니다. 서로 다른 언어와 이미지에서 같은 대상에 반응하는 특징을 추출한 사례입니다. [출처: Anthropic, Mapping the mind of a large language model (2024)](https://www.anthropic.com/research/mapping-mind-language-model)*

| 분석 방법 | 무엇을 살펴보는가? |
| --- | --- |
| **원래 뉴런을 직접 분석** | 실제 모델의 뉴런이 무엇에 반응하고, 어떤 계산으로 이어지는지 본다. |
| **SAE 등으로 특징을 추출** | 복잡한 신호를 더 이해하기 쉬운 요소로 나누고, 그 요소의 역할을 본다. |

Goodfire의 글은 원래 뉴런을 직접 분석하는 방법의 가능성을 강조합니다. 다만 어느 방법이든 **설명이 읽기 쉬운지**와 **실제 모델의 계산을 정확히 설명하는지**를 따로 확인해야 합니다. 추출한 특징에 그럴듯한 이름을 붙였다고 해서 원래 계산을 모두 설명한 것은 아닙니다. [Circuit Tracing의 방법과 한계](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)

<details>
<summary><strong>더 알아보기: 여러 뉴런을 함께 쓰는 것과 중첩은 같은 말일까?</strong></summary>

색을 담당하는 요소와 모양을 담당하는 요소가 따로 있다고 생각해 봅시다. “빨강”과 “원형”을 조합하면 빨간 원을, “파랑”과 “원형”을 조합하면 파란 원을 표현할 수 있습니다.

하나의 대상을 여러 요소가 함께 표현하므로 **분산 표현(Distributed representation)**입니다. 여러 요소를 조합한다는 점에서는 **합성(Composition)**이라고 부릅니다. 각 요소의 역할이 분명한 상태에서도 많은 대상을 표현할 수 있습니다.

따라서 “뉴런 수보다 표현할 수 있는 대상이 많다”는 사실만으로 중첩이라고 결론 내릴 수는 없습니다. 실제 모델에서는 합성과 중첩이 함께 나타날 수도 있습니다.

참고: [Distributed Representations: Composition & Superposition](https://transformer-circuits.pub/2023/superposition-composition/index.html)

</details>

<a id="limits"></a>

## 6. 어디까지 이해했고, 무엇이 남았을까?

이 글은 **생각보다 많은 뉴런에 설명할 만한 역할이 있고, 원래 모델을 직접 조사하는 방법이 유용하다**는 주장을 펼칩니다. 동시에 일부 사례를 설명했다는 것과 모델 전체를 이해했다는 것은 구분해야 합니다.

### 언어가 달라도 공유되는 개념

Anthropic의 2025년 연구에서는 Claude 3.5 Haiku에 ‘작다의 반대’를 여러 언어로 물었을 때, **작음·반대·큼**에 해당하는 특징을 공유하는 경로가 관찰됐습니다.

![영어 중국어 프랑스어의 질문이 작음 반대 큼이라는 공통 특징을 거치는 회로](https://www-cdn.anthropic.com/images/4zrzovbb/website/e0e156ea6c912a385d66ed562187fced8c392a58-1650x750.png)

*그림 9. 영어·중국어·프랑스어의 입력과 출력은 다르지만, 그 사이에서 일부 개념과 계산 경로를 공유합니다. 특정 질문에 대한 회로를 단순화한 도식이며, 모든 언어 처리 과정이 동일하다는 뜻은 아닙니다. [출처: Anthropic, Tracing the thoughts of a large language model (2025)](https://www.anthropic.com/research/tracing-thoughts-language-model)*

### 2026년 사례: 내부 개념을 바꾸면 답도 바뀔까?

2026년 7월 Anthropic은 **J-space**라는 내부 표현 공간을 분석한 연구를 공개했습니다. 아래 실험에서는 질문과 답변에 직접 등장하지 않는 중간 개념을 찾아 바꾸고, 결과가 어떻게 달라지는지 확인했습니다.

![J-space의 거미를 개미로 바꾸자 답이 8에서 6으로 달라지고 시의 계획 단어를 바꾸자 결말이 달라지는 실험](https://www-cdn.anthropic.com/images/4zrzovbb/website/98aba4182963219d29291912a0c3d2c299f7f1b5-1760x760.png)

*그림 10. 왼쪽은 ‘거미줄을 만드는 동물의 다리 수’를 묻는 상황에서 내부의 ‘거미’ 표현을 ‘개미’로 바꾸자 답이 8에서 6으로 바뀌는 실험입니다. 오른쪽은 시의 결말로 계획된 단어를 바꾸자 문장의 결말도 달라지는 사례입니다. 내부 표현이 답변에 인과적으로 기여하는지 시험한 결과입니다. [출처: Anthropic, A global workspace in language models (2026)](https://www.anthropic.com/research/global-workspace)*

### 일부 계산을 이해하는 것과 모델 전체를 이해하는 것

예를 들어 프로그램의 함수 역할을 많이 알아도 모든 입력에서의 동작이나 모든 버그를 아는 것은 아닙니다. 뉴런의 역할을 알아내는 일과 긴 대화, 여러 단계의 추론, 드문 실패까지 설명하는 일 사이에도 차이가 있습니다.

좋은 설명인지는 다음처럼 확인할 수 있습니다.

1. **새로운 입력에서도 맞는가?** 처음 분석에 사용하지 않은 문장이나 이미지에서도 반응을 예측해 봅니다.
2. **뉴런을 바꾸면 예상한 변화가 생기는가?** 값을 키우거나 줄였을 때, 설명한 기능에 맞게 답이 달라지는지 확인합니다.
3. **다른 뉴런을 바꿔도 같은 결과가 나오는가?** 아무 곳이나 건드려도 생기는 변화라면, 처음의 설명은 약해집니다.
4. **다른 연구자와 모델에서도 확인되는가?** 분석 방법이나 사례가 달라져도 결과가 유지되는지 살펴봅니다.

“중첩이 전혀 없다”거나 “Qwen을 완전히 해독했다”는 결론은 아직 이릅니다. 하지만 **이해하기 어려워 보이던 뉴런을 다시 조사할 이유가 충분하다**는 것이 이 글에서 주목할 부분입니다.

<a id="references"></a>

## 참고 자료

- **중심 글:** Nick Cammarata, Goodfire — [Neural Networks Are Already Clean](https://already-clean.goodfire.com/)
- **중첩의 배경:** [Toy Models of Superposition](https://arxiv.org/abs/2209.10652)
- **분산 표현과 중첩의 차이:** [Distributed Representations: Composition & Superposition](https://transformer-circuits.pub/2023/superposition-composition/index.html)
- **SAE 연구:** [Towards Monosemanticity](https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning)
- **Claude의 개념 지도와 금문교 특징:** Anthropic — [Mapping the mind of a large language model (2024)](https://www.anthropic.com/research/mapping-mind-language-model)
- **다국어 개념 공유와 추론 회로:** Anthropic — [Tracing the thoughts of a large language model (2025)](https://www.anthropic.com/research/tracing-thoughts-language-model)
- **J-space와 내부 표현 개입:** Anthropic — [A global workspace in language models (2026)](https://www.anthropic.com/research/global-workspace)
- **뉴런을 그림으로 살펴보는 방법:** [Feature Visualization](https://distill.pub/2017/feature-visualization/)
- **게이트를 사용하는 계산 구조:** [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- **회로 추적의 방법과 한계:** [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)

번호를 붙인 본문 그림에는 Goodfire 원문의 이미지 4장(그림 3·4의 각 두 패널), Qwen 분석 화면 캡처 3장(그림 5·6·7), Anthropic 원문의 이미지 5장(그림 1·2·8·9·10)을 사용했습니다. 각 그림 아래에 한국어 설명과 원문 링크를 표시했습니다. Qwen 캡처는 GitHub 첨부 이미지 주소로, 나머지 본문 이미지는 원문의 전체 URL로 연결했습니다. 별도의 이미지 폴더 없이 이 README.md의 내용을 복사해 GitHub에서 사용할 수 있습니다.
