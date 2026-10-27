# ‘Yes’와 ‘although’ 사이의 뉴런

*neuron-explorer 실행 화면으로 읽는 최근의 mechanistic interpretability · 2026년 10월 26일*

검은 화면 가운데 모델의 레이어들이 얇은 판처럼 겹쳐 있다. 점 하나를 고르면 옆에 단어들이 뜨고, 다른 레이어로 이어지는 선들이 나타난다. 선택된 뉴런의 주소는 `L20 N8692`. 오른쪽 패널에서 양의 방향으로 개입한 결과에는 `although`, `despite`, `though`, `虽然`, `尽管`이 보인다. 반대 방향에는 `Yes`, `indeed`, `是的`, `确实`이 있다.

언어는 섞여 있는데, 일부 표현의 용법은 꽤 가까워 보인다. 한쪽에는 “비록 …이지만”, 다른 쪽에는 “그렇다”가 모여 있다. 모델 내부의 숫자 하나에서 이런 대비를 발견하면, 일단 이름부터 붙이고 싶어진다. ‘양보하는 뉴런’일까. 아니면 ‘동의하는 뉴런’의 반대쪽일까.
<img width="1478" height="743" alt="{538F7DE8-EB32-44EB-9FCB-211D29660A97}" src="https://github.com/user-attachments/assets/fc0fd94f-942a-4049-84a6-0e76a7717a31" />

<img width="1327" height="490" alt="{26050E7C-FE53-4A20-AA9B-F1E53910E412}" src="https://github.com/user-attachments/assets/88437cfe-38cc-4587-be68-584e190d3154" />

<img width="556" height="285" alt="{3C8415E8-4CE2-4DD3-AF13-3F957594C083}" src="https://github.com/user-attachments/assets/01e2e558-a3b7-4202-8983-718f62e08d86" />

![neuron-explorer 실행 화면. L20 N8692를 선택한 상태에서 중앙에 레이어 지도와 연결 후보가, 양옆에 steer lens와 logit lens가 표시되어 있다.](<C:/Users/tgc04/Documents/ChatGPT/VLM interpretability/articles/assets/neuron-explorer-L20-N8692.png>)

*사용자가 제공한 neuron-explorer 실행 화면. 선택된 뉴런은 `L20 N8692`다. 상단 입력창의 `L22 N2742`와 구분해서 읽어야 한다. 이 글은 화면과 공개 코드·저장 캐시를 확인한 해설이며, 새로운 모델 실험을 수행한 결과 보고는 아니다.*

이 화면을 조금 더 정확히 읽으면 이야기가 흥미로워진다. 위 단어들은 모델이 방금 생성한 답변도, 단순히 확률이 가장 높은 단어 목록도 아니다. **선택한 뉴런에 개입했을 때 다음 토큰의 로그확률이 기준 실행보다 많이 올라간 항목들**이다. 익숙한 단어가 나왔다고 바로 뉴런의 의미를 알아낸 것은 아니지만, 다음 실험을 설계할 만한 단서는 생긴다.

이런 탐색을 가능하게 하는 것이 [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer)다. Qwen3.5-4B의 MLP 뉴런을 관찰하고, 활성값을 바꾸고, 그 결과를 같은 화면에서 비교하는 도구다. 프로젝트는 이 방식을 *steering-first*라고 부른다. 어떤 입력에서 뉴런이 켜지는지와 함께, 그 뉴런을 직접 조절하면 모델이 무엇을 쓰게 되는지를 살펴본다.

프로젝트를 둘러싼 관심을 보여주는 사례가 lyra의 10월 24일 게시물이다. 작성자는 Qwen3.5-4B의 두 뉴런에 서로 다른 방향으로 개입하는 설정을 공유하며 결과를 “evil qwen”이라고 불렀다. 게시물의 짧은 설명만 보면 모델 안에서 악당 스위치를 발견한 것처럼 읽히기도 한다. [원문 게시물](https://x.com/_lyraaaa_/status/2113833330429083662)

그런데 10월 26일 작성자의 후속 설명은 더 구체적이다. 거절을 제거하거나 검열을 해제하는 방법을 제시한 것이 아니라, 과장된 악당처럼 행동하게 만드는 성격 조절이라는 것이다. 작성자는 이 상태에서 모델의 전반적인 능력도 크게 떨어지며 실용적인 용도를 기대하기 어렵다고 덧붙였다. ‘evil’은 관찰한 스타일에 붙인 이름으로 읽어야 한다. 두 뉴런이 악의를 저장한다거나 안전장치 전체가 두 숫자로 설명된다는 결론을 뒷받침하지는 않는다. [작성자의 후속 설명](https://x.com/_lyraaaa_/status/2114597601899413609)

첨부 화면의 `L20 N8692`는 그 게시물에서 언급한 두 뉴런과도 다르다. 이 글에서 흥미롭게 보는 것은 ‘evil’이라는 이름보다, 화면에서 발견한 패턴을 검증 가능한 질문으로 바꾸는 과정이다.

뉴런을 조절한다는 말을 조금 풀어보자. 이 모델의 MLP에서는 두 계산 경로인 gate와 up을 결합해 중간 활성값을 만든다. 코드로 줄이면 `a = silu(gate) * up`이다. 이어지는 down projection이 이 벡터를 모델의 공통 정보 통로인 residual stream에 전달한다. 뉴런 하나의 기여는 `a[n] * W_down[:, n]`으로 쓸 수 있다. `a[n]`이 크기와 부호라면, `W_down[:, n]`은 그 뉴런이 정보를 보내는 방향이다.

neuron-explorer는 down projection 직전에 hook을 걸어 `a[n]`을 읽거나 바꾼다. 활성값에 일정량을 더할 수도 있고, 현재 값에 배수를 곱할 수도 있다. 별도의 미세조정 없이 추론 중간의 계산에 손을 대는 방식이다. 뉴런마다 출력 방향의 길이가 다르므로, 덧셈 개입은 residual norm을 기준으로 크기를 맞춘다. 같은 숫자를 더했는데 어떤 뉴런만 훨씬 세게 흔들리는 문제를 줄이려는 설계다. [활성값 개입 구현](https://github.com/lyramakesmusic/neuron-explorer/blob/ddb144e4121fbeb16ec080077dd67247e499132d/explorer/runtime.py#L80), [개입 강도 계산](https://github.com/lyramakesmusic/neuron-explorer/blob/ddb144e4121fbeb16ec080077dd67247e499132d/explorer/runtime.py#L235)

실시간 채팅은 양의 개입, 개입 없는 기준, 반대 방향 개입을 나란히 보여준다. 같은 질문을 놓고 어디에서 답변이 갈라지는지 비교할 수 있다. 대규모 sweep은 이 과정을 여러 뉴런과 프롬프트에 반복하고, 생성된 텍스트와 토큰 확률을 저장한다. 생성 결과에서 반복되는 특징적 단어를 찾을 때는 같은 크기의 무작위 방향에도 개입해 대조군으로 쓴다. 출력이 달라졌다는 사실만으로 특정 뉴런에 의미 있는 기능을 부여하지 않기 위해서다. [실시간 생성](https://github.com/lyramakesmusic/neuron-explorer/blob/ddb144e4121fbeb16ec080077dd67247e499132d/explorer/runtime.py#L165), [sweep과 무작위 대조군](https://github.com/lyramakesmusic/neuron-explorer/blob/ddb144e4121fbeb16ec080077dd67247e499132d/sweep/steer_sweep.py#L143)

다시 `L20 N8692`로 돌아가면, 화면 속 두 종류의 lens 사이에도 차이가 있다. **Logit lens는 그 뉴런의 출력 방향을 final normalization과 출력층에 바로 통과시켜 읽는다. Steer lens는 실제 입력 문맥에서 뉴런을 조절하고, 뒤에 남은 레이어들의 계산까지 거친 결과를 읽는다.** 전자는 방향 자체를 바로 어휘로 번역해 보는 것이고, 후자는 모델이 그 방향을 실제 계산 속에서 어떻게 사용하는지 살펴보는 것이다. [두 lens의 구현](https://github.com/lyramakesmusic/neuron-explorer/blob/ddb144e4121fbeb16ec080077dd67247e499132d/explorer/runtime.py#L329)

첨부 화면에서는 logit lens에 잘게 끊긴 토큰 조각들이 보이는 반면, steer lens에는 양보나 긍정과 관련된 표현들이 모인다. 이 한 사례는 뉴런의 출력 벡터를 곧바로 단어로 읽는 것과, 그 벡터가 이후 계산을 통과한 효과를 읽는 것이 다를 수 있음을 보여준다. 다른 뉴런에서도 같은 차이가 나타나는지는 별도로 확인할 문제다.

구체적으로 이 steer lens는 네 개의 짧은 문맥을 사용한다. 두 개는 정체성 질문과 자유 생성 요청이고, 나머지는 일반 문장의 도입부다. 각 문맥에서 뉴런에 양·음 방향으로 개입하고, 마지막 위치의 다음 토큰 로그확률 변화를 평균한다. 저장된 캐시의 개입 강도는 `±0.8`이며, 이 스크립트의 기준 크기는 그 네 문맥에서 측정한 레이어 residual norm의 중앙값이다. 상위 12개 토큰과 분포 변화량을 저장한다. [Steer lens 계산](https://github.com/lyramakesmusic/neuron-explorer/blob/ddb144e4121fbeb16ec080077dd67247e499132d/sweep/steer_lens.py#L21)

패널의 `KL 0.581 / 0.588`은 그 분포 변화량이다. 개입 후 분포와 기준 분포 사이의 KL divergence를 문맥에 걸쳐 평균한 값이며, 자연로그를 사용하므로 단위는 nats다. ‘58%의 확신’이나 ‘뉴런 해석의 정확도’가 아니다. 네 문맥에 걸친 로그확률 변화의 평균에서 이런 단어들이 상위에 나타났다는 관찰은 흥미롭지만, 각 문맥에서 효과가 일관되는지와 그 뉴런이 다른 상황에서도 양보나 동의에 관여하는지는 더 검증해야 한다. [Δlogprob와 KL 계산](https://github.com/lyramakesmusic/neuron-explorer/blob/ddb144e4121fbeb16ec080077dd67247e499132d/sweep/steer_lens.py#L94)

이 접근이 최근의 연구와 맞닿는 지점은 **모델에 원래 들어 있는 뉴런을 분석의 출발점으로 삼는다는 것**이다. 익숙한 또 다른 접근은 sparse autoencoder, 즉 SAE처럼 활성값을 희소한 특징들로 분해하는 모델을 따로 학습하는 것이다. 여러 의미가 섞여 보이는 내부 표현에서 사람이 읽기 쉬운 단위를 얻으려는 시도다. Anthropic의 2025년 circuit tracing 연구는 cross-layer transcoder라는 별도의 방법으로 MLP 계산을 근사하고, 그 특징들 사이의 관계를 추적했다. 이 경우도 그래프가 제안한 메커니즘을 개입으로 검증하는 절차가 핵심이다. [Circuit Tracing, 2025년 3월 27일](https://www.transformer-circuits.pub/2025/attribution-graphs/methods.html)

올해에는 기존 MLP 뉴런을 직접 쓰는 경로에서도 결과가 나왔다. Arora 등의 「Language Model Circuits Are Sparse in the Neuron Basis」는 별도의 SAE를 학습하지 않고 뉴런 기반 회로를 추적했다. 연구진은 평가한 과제들에서 적은 수의 뉴런으로 모델 동작을 제어할 수 있음을 보였고, 도시에서 주를 거쳐 주도를 답하는 다단계 과제에서도 중간 추론과 관련된 뉴런 집합을 찾아 개입했다. 이 논문은 2026년 1월 공개되어 6월에 개정됐다. 특정 과제에서 뉴런 기반 분석이 유효하다는 결과와, 모든 뉴런의 의미를 쉽게 설명할 수 있다는 주장은 구분해야 한다. [논문과 개정 이력](https://arxiv.org/abs/2601.22594)

더 최근에는 Nick Cammarata가 Goodfire에서 「Neural Networks Are Already Clean」을 발표했다. [공식 연구 목록](https://www.goodfire.com/research)에 표시된 발행일은 2026년 10월 15일이다. 그는 겉으로 무관해 보이는 뉴런의 반응도 더 넓은 문맥과 연결 관계를 살펴보면 관련된 일반화로 설명되는 경우가 많다고 주장한다. 다만 원문부터 비공식 연구 에세이이자 작업 가설이라고 밝힌다. 이를 곧바로 ‘superposition은 틀렸고 SAE는 필요 없다’는 확정 판정으로 읽기에는 근거의 범위가 다르다. neuron-explorer의 [프로젝트 노트](https://github.com/lyramakesmusic/neuron-explorer/blob/ddb144e4121fbeb16ec080077dd67247e499132d/NOTES.md)도 이 글을 출발 자료 중 하나로 적고 있다. [Goodfire 원문](https://already-clean.goodfire.com/)

이 연구들이 neuron-explorer의 개별 결과를 검증해 주는 것은 아니다. 다만 ‘어떤 단위를 먼저 볼 것인가’라는 질문에 모델 본래의 뉴런도 실험 대상으로 올려놓는다. 탐색기의 역할은 그 질문에 쉽게 손을 댈 수 있게 만드는 데 있다. 활성값을 보고, 생성 변화를 보고, 이웃 후보를 살핀 다음, 다시 입력을 바꿔 가설을 시험할 수 있다.

물론 화면의 선은 단어 목록보다도 신중하게 읽어야 한다. 중앙의 3D 지도는 뇌의 물리적 배치를 재현한 것이 아니다. `writes` 모드의 뉴런 위치는 출력 가중치 방향을 정규화하고 PCA와 UMAP을 거쳐 격자에 배치한 결과다. 비슷한 방향을 가진 뉴런을 가까이 놓으려는 시각화다. [지도 배치 코드](https://github.com/lyramakesmusic/neuron-explorer/blob/ddb144e4121fbeb16ec080077dd67247e499132d/explorer/build_layout.py#L40)

연결선 역시 한 뉴런의 출력 방향과 다른 뉴런의 gate·up 가중치에서 계산한 후보 관계다. 중간 레이어의 실제 계산과 입력별 활성 상태를 모두 추적한 인과 회로는 아니다. 화면의 `fed by`, `held back by`, `pushes toward` 같은 라벨은 탐색용 추정으로 읽는 편이 정확하다. 연결 옆의 숫자도 실험으로 측정한 확률이나 인과 효과 크기가 아니라 가중치에서 계산한 상대 점수다. [연결 점수의 정의](https://github.com/lyramakesmusic/neuron-explorer/blob/ddb144e4121fbeb16ec080077dd67247e499132d/explorer/runtime.py#L353), [web 요약 생성](https://github.com/lyramakesmusic/neuron-explorer/blob/ddb144e4121fbeb16ec080077dd67247e499132d/explorer/server.py#L520)

그래서 이 화면에서 다음에 할 일은 뉴런에 멋진 이름을 붙이는 것보다, 그 이름이 틀릴 수 있는 입력을 만드는 일에 가깝다. 예를 들어 ‘양보·대조와 관련된 기능’이라는 가설을 세웠다면 다음과 같은 실험을 설계할 수 있다. 아래는 아직 수행하지 않은 후속 실험 제안이다.

| 확인할 질문 | 비교할 입력·개입 | 확인할 결과 |
|---|---|---|
| 특정 단어에만 반응하는가? | `although` 같은 표지가 있는 문장과, 그 단어 없이 비슷한 양보 관계를 표현한 문장 | 토큰별 자연 활성값의 위치와 크기 |
| 단어의 등장과 실제 용법을 구분하는가? | “although의 철자를 쓰세요” 같은 언급과 실제 양보절 | 표면 문자열만으로 가설을 설명할 수 있는지 |
| 다른 언어에서도 이어지는가? | 의미를 맞춘 영어·중국어·한국어 문장 | 언어별 활성 패턴과 개입 효과의 일관성 |
| 개입 효과는 얼마나 선택적인가? | 작은 양·음 개입, 0배 ablation, 개입 없는 기준, 무작위 대조군 | 목표 표현의 변화와 함께 문법·사실성·반복도도 비교 |

관찰과 개입은 여기서 서로 다른 질문에 답한다. 자연 입력에서 뉴런이 활성화되는 것은 모델이 언제 그 뉴런을 사용하는지에 대한 단서다. 강제로 활성값을 바꿨을 때 출력이 움직이는 것은 그 개입의 효과다. 특히 큰 개입은 자연스러운 계산 범위를 벗어날 수 있으므로, 극적인 답변 한 개와 함께 평범한 과제의 성능도 확인해야 한다. 작성자가 ‘evil Qwen’의 능력 저하를 따로 설명한 것도 이런 구분을 떠올리게 한다.

VLM 연구로 확장할 때도 같은 구체성이 필요하다. 기본 모델이 멀티모달이더라도, 여기서 확인한 코드는 텍스트 토큰을 입력하고 언어 모델 쪽 MLP를 관찰한다. sweep은 이미지와 비디오 입력을 비활성화한다. 이 캡처만으로 이미지에 반응하는 시각 뉴런이나 시각적 grounding을 확인한 것은 아니다. 이미지를 바꾸고 질문은 유지했을 때 어떤 활성값과 출력이 달라지는지까지 실험해야 그 연결을 논할 수 있다. [언어 모델 hook](https://github.com/lyramakesmusic/neuron-explorer/blob/ddb144e4121fbeb16ec080077dd67247e499132d/explorer/runtime.py#L64), [sweep의 입력 설정](https://github.com/lyramakesmusic/neuron-explorer/blob/ddb144e4121fbeb16ec080077dd67247e499132d/sweep/steer_sweep.py#L169)

`L20 N8692`의 이름 칸은 아직 비어 있다. 현재 채울 수 있는 메모는 이 정도다. “네 개 문맥에 걸친 평균 로그확률 변화에서, 양의 개입에는 양보 관련 표현들이, 반대 개입에는 긍정 관련 표현들이 상위에 나타났다.” 그다음에는 `although`가 없어도 같은 효과가 나타나는 문장, `although`가 있어도 가설이 맞지 않는 문장을 준비하면 된다. 그 두 종류의 입력을 구분해 내기 시작할 때, 화면의 단어 목록은 뉴런에 대한 설명으로 조금씩 바뀐다.

---

*확인 기준: 2026년 10월 26일. 실행 사진은 사용자 제공 자료이며, 코드 설명은 neuron-explorer 커밋 `ddb144e4121fbeb16ec080077dd67247e499132d`를 기준으로 했다. 스크린샷의 steer-lens 토큰과 KL 반올림값은 함께 제공된 저장 캐시와 대조했다. 인용한 게시물·연구의 주장과 이 글의 후속 실험 제안은 별개다.*
