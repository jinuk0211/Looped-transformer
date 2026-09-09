OPENAI GPT 6 ASTRA 리서치 - RSI, Looped transformer
========
<img width="4096" height="4096" alt="image" src="https://github.com/user-attachments/assets/2d96a5c5-4f8e-43ba-84e4-85d95ab7d821" />

# RSI (recursive self improvement)
최근 OpenAI astra 시스템 카드에는 GPT 5.6 Sol을 사용해 훈련을 진행하였다는 사실이 적혀있고 다음 세대의 Pretrained Bel 모델 역시 Astra를 사용해 훈련중이라는 사실이 적혀있다. 2026년 9월 6일 오픈 AI 수석연구원 Jakub Pachocki이 작성한 보고서에 의하면 
"우리가 이제 인간과 상당히 다른 형태의 지능을 만들고 있고, 이 지능이 자기 자신의 연구·개발까지 가속하기 시작하는 단계에 접근하고 있으니 속도를 통제해야 한다"를 주장하고 있다. 

<img width="796" height="515" alt="image" src="https://github.com/user-attachments/assets/6b21efac-08a5-45a4-8d39-f2fae0ec1147" />

Pachocki의 관점은 다음과 같은데 인간의 두뇌 → 진화 + 생물학 + 사회적 학습으로 만들어지지만 AI는 → 엄청난 계산량으로 optimization을 반복해서 자라나게 된다. 그래서 그는 AI를 우리가 하나하나 설계한 프로그램이라기보다 “grown more than designed”, 즉 설계했다기보다는 길러낸 복잡한 시스템이라고 표현한다. 내부의 작은 메커니즘은 Interpretability로 분석할 수 있을지 언정 전체적으로 왜 그런 지능적 행동이 나오는지는 뇌과학이 인간 뇌를 완전히 이해하지 못하는 것과 비슷하게 설명하기 어렵다는 주장이다 (이는 이후 recurrent depth의 looped transformer 구조와도 통함)

Pachocki는 2023년 중반 “RLSlow”라는 내부 연구 프로젝트에서 reasoning model을 스케일하면 pretrained model이 자체적인 chain-of-thought 추론 능력을 발휘하도록 만들 수 있다는 결과를 처음 봤다고 한다.
그 당시 Pachocki와 Szymon이 밤새 사무실에 있으면서 생각한 게, “벤치마크가 얼마나 오르지?”가 아니라 “우리 생전에 인간보다 의미 있게 똑똑한 기계를 실제로 보게 되겠구나.” 였다고 적는다.
<img width="1853" height="846" alt="image" src="https://github.com/user-attachments/assets/437e746d-1c27-45f8-aab0-90d003c84c2c" />

파호츠키는 현재 내부 결과를 바탕으로 현재의 AI 발전 속도가 RSI 단계까지 이어질 가능성을 강하게 예상한다고 썼다. 앞으로 몇 년 동안 비슷하거나 더 큰 규모의 capability jump가 반복되고, AI가 자신의 개발에 점점 더 많이 참여할 수 있다는 것이다.

그리고 이걸 꽤 심각하게 본다. 이는 Alignment 분야와 크게 이어지는데 RL, pretraining 데이터 필터링 등을 통해 지금 모델이 안전하고 착한 행동을 하게 만들고 있는데 AI가 학습 중 본 상황에서만 착한것이 별 의미가 없어질 것이라는 주장.

예를 들어 training 때

A 상황 → 위험 행동하지 말 것
B 상황 → 사람을 속이지 말 것
C 상황 → 시스템 침해하지 말 것을 배웠는데,

초지능 AI가 완전히 새로운 상황 Z를 만나면

training에서 직접 본 적 없음 -> '인간이 원했던 원칙'을 제대로 일반화할 수 있는가? 가 문제가 된다.

Pachocki는 alignment의 본질적인 문제가 generalization, 즉 일반화라고 한다.

더 무서운 부분은 모델이 강한 목표 최적화를 받으면,겉으로는 aligned된 사고를 하면서도목표를 달성하기 위해 그 원칙을 자기합리화할 수 있다는 것이다. 기존에 자연어로 이를 확인할 수 있는 방법에는 CoT 모니터링이 있었는데 CoT를 계속 reward/punish하면 모델이 나중에는

"이 생각을 쓰면 감시당하는구나"
             ->
겉으로는 안전한 reasoning만 생성
             ->
실제 내부 계산과 CoT가 분리될 가능성이 존재한다는 것이다. -> 그래서 mechanistic interpretability (SAE, activation steering, logit lens 이런거 열심히 연구하겟다함)

결론 : “Alien Mind”는 AGI가 외계인 같다는 감성적인 표현이라기보다, 우리가 내부 작동과 일반화를 완전히 설명할 수 없는 비인간적 지능을 self-improvement loop에 넣기 전에 monitoring을 해결해야 한다는 경고라고 보면 정확하다.
<< self improvement 더 나아가서 통제 해야한다는 말을 보니 RSI가 어느정도 신빙성이 있는 얘기인듯함

<img width="2048" height="894" alt="image" src="https://github.com/user-attachments/assets/39466bba-f0e6-4614-af86-0465b80fc0c4" />
<img width="1170" height="707" alt="image" src="https://github.com/user-attachments/assets/270e5c3c-13b2-4355-b5cc-a721ea94f9e4" />
최근 트위터에서 모델 출시 속도가 가속화되고 있는 점, 제프딘과 같은 개발자가 Discovery loop와 같은 RSI를 기반으로하는 회사가 2026년에 들어서면서 폭발적으로 증가하고 있다는점, 나비에-스토크스 방정식과 같은 문제를 search, loop로 풀어내는점을 보아 프론티어 모델 내부에서는 AI 지능이 스스로를 발전시키고 있다는 의견이 지배적

# Looped-transformer

참조문헌

https://arxiv.org/abs/2607.07663 Recursive Self-Improvement in AI: From Bounded Self-Refinement to Autonomous Research Loops

https://arxiv.org/abs/1807.03819?utm_source=chatgpt.com
Universal Transformers

https://arxiv.org/abs/2502.05171
Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach

https://arxiv.org/abs/2604.07822
Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers

https://arxiv.org/html/2606.31779v2
Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers

https://arxiv.org/abs/2609.01343
SMELT: Scaling Laws for Compute-Matched MoE Looped Transformers

반복형 아키텍처는 무엇을 바꾸는가
========================================
《디 인포메이션》의 보도에 따르면, GPT 6 Astra는 **반복형 트랜스포머(looped transformer)** 구조를 사용한다.

OpenAI가 전체 기술 사양을 공개하지 않았기 때문에 정확한 구현 방식은 알려져 있지 않다. 다만 이 구조의 기본 원리는 이미 공개된 연구를 통해 상당 부분 이해할 수 있다.

기존 트랜스포머에서는 모델의 내부 상태인 ‘hidden state’가 각 연산 블록을 한 번씩 통과한다. 반면 반복형 트랜스포머는 한 블록에서 나온 결과를 다시 같은 블록에 넣어 처리할 수 있다.
<img width="1165" height="969" alt="image" src="https://github.com/user-attachments/assets/7bd87538-fb53-41fb-93c5-a211f6af215a" />

쉽게 말하면 **모델 내부에서 초안을 여러 번 다듬는 방식**이다.

먼저 초기 표현을 만들고, 같은 가중치를 사용하는 연산을 다시 거치면서 이를 개선한다. 이 과정을 여러 차례 반복한 뒤에야 사용자에게 보이는 다음 토큰을 출력한다.

이때 파라미터 수는 그대로다. 매번 같은 가중치를 재사용하므로, 반복 횟수가 늘어나도 반드시 모델 자체가 커지는 것은 아니다. 늘어나는 것은 **연산량**이다.

기존 트랜스포머에서는 하나의 토큰이 서로 다른 블록들을 순서대로 통과한다. 반복형 트랜스포머에서는 같은 구조를 여러 번 통과할 수 있다. 즉, **파라미터를 늘리지 않고도 연산의 깊이를 늘리는 것**이다.

<img width="600" height="680" alt="image" src="https://github.com/user-attachments/assets/31b546da-6bb2-4f8e-85c0-42fbfce5bb63" />

이 구조에서는 다음과 같은 선택이 가능해진다.

* 파라미터를 늘리면 더 많은 메모리가 필요하다.
* 반복 횟수를 늘리면 더 많은 연산이 필요하다.
* 하나의 모델이 작업에 따라 서로 다른 양의 연산을 사용할 수 있다.

특히 마지막이 중요하다. 새로운 모델을 만들지 않고도, 문제를 처리하는 연산 깊이를 바꿀 수 있기 때문이다.

**3. 반복적인 연산 깊이는 어떻게 학습시키는가**
===========================================
트랜스포머 블록을 단순히 반복 실행하도록 만든다고 해서 모든 문제가 해결되지는 않는다.
학습할 때 항상 정확히 32번씩 반복한다고 해보자. 그러면 모델은 각 반복 단계의 순서에 의존하는 방식을 배울 수 있다. 첫 번째 반복에서는 어떤 일을 하고, 열 번째에서는 다른 일을 하며, 서른두 번째는 최종 출력을 만드는 단계로 굳어질 수 있다.
결국 겉으로만 반복 구조일 뿐, 실질적으로는 고정된 32층 신경망처럼 동작하게 되는 것이다.
연구자들은 이를 방지하기 위해 **학습 중 반복 횟수를 바꾼다.**

대표적인 사례가 반복 깊이 모델인 **Huginn**이다. 연구진은 35억 개의 파라미터를 가진 모델을 약 8,000억 개의 토큰으로 학습시켰다.
각 학습 단계에서는 평균이 약 32인 확률분포에서 반복 횟수를 뽑았다. 어떤 예시는 4번, 다른 예시는 17번이나 40번, 또 다른 예시는 90번에 가깝게 처리할 수 있었다.
따라서 모델은 현재 단계가 마지막 반복인지 미리 알 수 없었다. 정해진 순서대로 작업을 수행하는 것보다 더 일반적인 능력을 배워야 했다.

바로 **현재의 내부 상태를 더 나은 상태로 개선하는 능력**이다.
그런데 수십 번의 반복을 학습시키면 또 다른 문제가 생긴다. 메모리 사용량이다.
일반적인 역전파 방식은 각 반복 단계의 중간 활성값을 저장해야 한다. 반복 횟수가 많아지면 필요한 메모리도 엄청나게 늘어난다.
Huginn 연구진은 현실적인 절충안을 사용했다. 순전파, 즉 실제 계산은 여러 번 반복하되, **기울기는 마지막 8번의 반복 구간에 대해서만 계산**한 것이다.
앞선 반복들도 최종 결과에 영향을 주지만, 그 과정에서 발생한 모든 활성값을 메모리에 보관할 필요는 없어진다.

이는 ‘truncated backpropagation’와 비슷하다. 다만 일반적인 순환 신경망처럼 문장의 앞뒤 단어 사이에서 반복이 일어나는 것이 아니라, **하나의 입력을 처리하는 연산 깊이 방향으로 반복이 일어난다**는 차이가 있다.
정리하면, 학습 단계마다 반복 횟수를 새로 뽑고, 역전파는 마지막 8번에만 적용한다.
이 모델이 배우는 것은 ‘정확히 32단계로 이루어진 고정 절차’가 아니다. **언제 반복을 멈추더라도 유용한 답에 가까워지도록 내부 상태를 개선하는 방법**이다.
이렇게 되면 ‘깊이’의 의미도 달라진다. 더 이상 엔지니어가 학습 전에 정해 놓는 구조적 특성에만 머물지 않고, 추론할 때 조절할 수 있는 변수가 된다.

**4. 추론할 때 필요한 만큼 연산하기**
======================================
현재 대부분의 추론 조절 방식은 모델이 생성하는 토큰을 중심으로 작동한다.
모델에게 더 오래 생각하도록 하면, 더 긴 사고 과정(chain of thought)을 생성하거나, 출력 토큰을 더 많이 사용하거나, 중간 설명을 추가로 작성하는 식이다.

반복형 구조는 다른 방법을 제공한다.
가중치를 바꾸거나 설명을 길게 쓰게 하지 않아도, 내부 연산의 반복 횟수를 조절할 수 있다. 예를 들면 다음과 같다.

* 간단한 작업: 4번 반복
* 어려운 수학 증명: 32번 반복
* 복잡한 코딩 문제: 64번 반복

모델과 가중치는 그대로다. 달라지는 것은 각 토큰을 처리하기 위해 수행하는 **내부 연산의 양**이다.
공개된 반복 깊이 실험에서도 예상되는 패턴이 나타난다. 초반에는 반복 횟수를 늘릴수록 성능이 빠르게 좋아진다. 이후에는 개선 폭이 줄어들다가 점차 정체된다.
이는 은닉 상태가 어느 정도 수렴하고 있음을 시사한다.
초기 반복에서는 큰 오류를 수정하고, 이후에는 작은 세부 사항을 다듬는다. 그러다가 어느 순간부터는 연산을 더 해도 의미 있는 개선이 거의 일어나지 않는다.
미래의 시스템은 그 시점을 자동으로 감지할 수도 있다.
모든 프롬프트에 똑같은 반복 횟수를 배정하는 대신, 내부 상태가 충분히 안정될 때까지 연산을 이어가는 것이다. 쉬운 토큰은 적은 비용으로 처리하고, 어려운 토큰에는 더 많은 연산을 배정할 수 있다.
이것이 실제로 구현된다면 **문제에 맞춰 연산 깊이를 조절하는 방식**이 된다.
더 오래 생각한다고 해서 반드시 더 많은 글을 써야 하는 것은 아니게 된다.

<img width="2400" height="4000" alt="image" src="https://github.com/user-attachments/assets/9e1a69ab-26ea-4c58-8bf6-6202a9a073ec" />

**5. 관련 연구와 초기 실용적 성과**
=============================
트랜스포머의 층을 재사용한다는 발상 자체는 새롭지 않다.
Universal Transformers는 2018년에 반복 처리를 도입했다. ALBERT는 여러 층이 가중치를 공유하게 해 파라미터 수를 줄였다. Mixture of Recursions는 토큰마다 서로 다른 양의 연산을 거치도록 보내는 방법을 탐구했다.
공개 모델인 **Nanbeige4.2 3B**는 특히 직접적인 사례다.

이 모델의 설정에는 22개 층과 `num_loops: 2`가 명시되어 있다. 즉, 22개 층을 두 번 통과한다. 파라미터 수는 대략 22층 모델 수준이지만, 연산 깊이는 약 44층 모델에 해당한다.
22개 층, 반복 횟수 2회, Apache 2.0 라이선스. 반복 구조가 이미 공개 모델의 설정에서 확인되는 것이다.
그동안 반복형 트랜스포머는 흥미로운 연구 주제였지만, 주류 방식으로 자리 잡지는 못했다. 부족했던 것은 **규모를 키워도 효과가 유지된다는 강한 실험적 근거**였다.

9월 1일 공개된 **SMELT**는 반복형 트랜스포머와 기존 트랜스포머를 동일한 조건에서 비교하기 위해 설계된 연구다.
연구진은 파라미터 수, 학습 연산량, KV 캐시 크기를 통제해 비교했다. 그 결과 반복형 모델은 같은 성능에 도달하는 데 필요한 학습 연산량이 **6.8~18% 적었다.** 가장 큰 이득은 코드 분야에서 나타났다.
어텐션의 작동 방식에도 변화가 관찰됐다.
기존 트랜스포머에서는 문장 앞부분의 토큰들이 종종 **‘어텐션 싱크(attention sink)’**, 즉 주의가 과도하게 몰리는 지점이 된다. 의미상 중요하지 않아도 다른 토큰보다 훨씬 많은 주의를 받는 것이다.
그런데 같은 블록을 두 번째로 통과한 뒤에는, 어텐션이 실제로 더 관련 있는 토큰 쪽으로 이동했다.
두 번째 반복은 첫 번째 계산을 단순히 되풀이한 것이 아니다. 첫 번째 결과를 바탕으로 **더 필요한 정보에 집중하는 처리**를 수행한 것이다.
SMELT가 앞으로 모든 모델을 반복형으로 만들어야 한다고 증명한 것은 아니다. 다만 주요 실험 조건을 맞춘 뒤에도 반복 구조가 경쟁력을 유지한다는 점은 보여준다.
이제 질문은 “반복 깊이가 작동하는가?”를 넘어, **“이 방식을 얼마나 큰 규모까지 확장할 수 있는가?”**로 옮겨가고 있다.

원문에 포함된 참고자료:

* [OpenAI의 GPT 6 Astra 발표](https://openai.com/index/gpt-6-astra/)
* [GPT 6 Astra와 AGI에 관한 Greg Brockman의 발언](https://www.axios.com/2026/09/03/openai-astra-gpt-6-agi-brockman)
* [GPT 6 Astra의 사이버보안 위험 등급 ‘Critical’](https://www.unite.ai/openai-releases-gpt-6-astra-its-first-model-rated-critical-for-cyber/)
* [Ilya Sutskever의 딥러닝 추천 자료](https://github.com/dzyim/ilya-sutskever-recommended-reading)
* [Astra와 반복형 트랜스포머에 관한 The Verge 보도](https://www.theverge.com/ai-artificial-intelligence/988334/openai-astra-ai-monitoring-safety)
* [잠재 추론을 통한 추론 시 연산량 확장 연구](https://arxiv.org/abs/2502.05171)
* [Mixture of Recursions](https://arxiv.org/abs/2507.10524)
* [Nanbeige4.2 3B 모델](https://huggingface.co/Nanbeige/Nanbeige4.2-3B)
* [SMELT 논문](https://arxiv.org/abs/2609.01343)
* [Astra의 추론 구조를 둘러싼 AI 안전성 우려](https://techcrunch.com/2026/09/02/openais-new-reasoning-technique-alarms-ai-safety-experts/)
* [Sebastian Raschka의 Astra 및 반복형 트랜스포머 분석](https://sebastianraschka.com/blog/2026/openai-astra-looped-transformers.html)
* [사고 과정의 모니터링 가능성에 관한 연구](https://arxiv.org/abs/2507.11473)
* [Simon Willison의 GPT 6 Astra 분석](https://simonwillison.net/2026/Sep/3/gpt6-astra/)
