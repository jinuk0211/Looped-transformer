# A Mathematical Theory of the Jacobian Lens

> **Jacobian lens(J-lens)는 무엇을 찾아내는가?** LLM 내부에서 working memory처럼 보이는 방향을 찾아주는 J-lens가 실제로는 **크게 자라고(magnitude), token 위치를 넘어 일관되며(lag coherence), 문맥을 넘어 일관된(context coherence)** feature를 골라낸다는 것을 **Lyapunov 이론**으로 설명하고, toy model로 검증한 논문.

| 항목 | 내용 |
|---|---|
| Authors | Mitchell Ostrow (MIT BCS), Daoyuan Qian (MIT BCS), Toni J.B. Liu (Cornell Physics), Ila Fiete (MIT BCS) |
| arXiv | [2610.31955](https://arxiv.org/abs/2610.31955) (v1, 2026-10-22, cs.LG, CC BY 4.0) |
| 대상 기법 | **Jacobian lens (J-lens)**, Gurnee et al. (2026) |
| 검증 모델 | **Bayes filter**(정확한 해석해), **2-layer toy transformer**(HMM next-token prediction), 실제 **LLM 4개**의 Jacobian spectrum |
| Code | 공개 링크 없음 (alphaXiv 기준) |

---

## 목차

1. [TL;DR](#tldr)
2. [Background: Lens 기법과 J-lens](#1-background-lens-기법과-j-lens)
3. [Definitions: J-lens와 Lyapunov 관점](#2-definitions-j-lens와-lyapunov-관점)
4. [Theory: J-lens는 Coherent Mode를 고른다](#3-theory-j-lens는-coherent-mode를-고른다)
5. [Toy Model: Hidden Markov Model 과제](#4-toy-model-hidden-markov-model-과제)
6. [Bayes Filter의 J-lens: 정확한 해](#5-bayes-filter의-j-lens-정확한-해)
7. [Numerical Results](#6-numerical-results)
8. [실제 LLM의 Jacobian (Appendix B)](#7-실제-llm의-jacobian-appendix-b)
9. [Discussion](#8-discussion)
10. [Limitations](#9-limitations)
11. [Takeaways](#10-takeaways)

---

## TL;DR

- **J-lens란**: hidden state $`\mathbf{h}_\ell`$에 **여러 context와 미래 token 위치에 대해 평균한 Jacobian**을 곱한 뒤 unembedding해서 token으로 읽는 lens다. Gurnee et al.(2026)은 이것으로 LLM 안에서 **working memory 같은 방향(J-space)** 을 찾았다.
- **질문**: 평균한 Jacobian은 개별 Jacobian과 매우 다르다. 그렇다면 J-lens는 **정확히 어떤 feature를 골라내는가?**
- **답**: J-lens의 top singular direction은 세 가지 성질의 **곱**을 최대화한다.

```math
\sigma_1(\mathbf{J}_H) = \max_{\lVert \mathbf{v} \rVert = 1} \; \rho_c(\mathbf{v}) \, \rho_\tau(\mathbf{v}) \, a(\mathbf{v})
```

  - **Magnitude** $`a`$: layer를 지나며 **얼마나 크게 자라는가** (Lyapunov exponent와 대응)
  - **Lag coherence** $`\rho_\tau`$: 미래 **token 위치를 넘어 얼마나 일정한가** (회전하거나 빨리 사라지면 작아짐)
  - **Context coherence** $`\rho_c`$: **다른 문맥에서도 같은 방향으로 반응하는가**
- **검증**: 세 성질을 각각 따로 조절할 수 있는 **HMM 과제**를 만들고, **Bayes filter**의 J-lens를 해석적으로 계산했다. 2-layer transformer의 J-lens도 예측된 순서대로 latent를 골랐다. 두 latent의 순위가 **뒤바뀌는 지점까지** 예측했다(transformer 오차 약 0.01).
- **부산물**: J-lens의 singular value는 **explained variance와 무관**하다(Corollary 1.1). Gurnee et al.의 J-space가 activation variance의 **10% 이하**만 설명한 이유다.
- **새 도구**: 평균 대신 특정 주파수로 필터링하는 **ω-J-lens** family를 제안했다. 빠르게 변하는 정보(예: syntax)와 느리게 변하는 정보를 분리할 수 있다.
- **Global Workspace**: 저자들은 이 성질들이 "Global Workspace"를 뜻하는지에 대해서는 **입장을 취하지 않는다.**

---

## 1. Background: Lens 기법과 J-lens

### 1.1 Lens 계열 기법
- **Logit lens**, **tuned lens**: 중간 layer의 hidden state를 바로(또는 학습된 변환 후) unembedding해서 "지금 이 layer가 어떤 token을 생각하는지" 읽는다.
- **J-lens** (Gurnee et al. 2026): hidden state에 **transformer의 평균 Jacobian**을 먼저 곱한 뒤 unembedding한다. 즉 "이 방향을 조금 움직이면 **최종 output이 어떻게 바뀌는가**"를 많은 context와 미래 위치에 대해 평균한 것이다.
- Gurnee et al.은 J-lens로 LLM 안에서 **working memory representation을 닮은 linear direction**들을 찾았고, 이 방향들이 이루는 공간을 **J-space**라고 불렀다.

### 1.2 평균이 문제를 만든다
- **평균한 Jacobian은 개별 Jacobian과 크게 다를 수 있다.**
- **Figure 1**(이미지 없음): 오픈소스 LLM 4개에서, middle layer → final layer Jacobian의 top-100 singular value spectrum을 PCA로 비교했다. 개별 Jacobian(검정)과 평균 Jacobian(빨강)이 뚜렷하게 다른 위치에 놓인다. Prompt는 Wikipedia, Python code, math, news 등 다양한 dataset에서 뽑았다.
- **Appendix A**의 설명: 개별 matrix가 $`X_i = U P_i S P_i^\top V^\top`$처럼 singular value 순서만 무작위로 섞인 형태라면, 평균 $`\bar{X} = U \bar{S} V^\top`$의 diagonal은 $`\bar{s} = \tfrac{1}{d}\sum_j e^{-\beta j}`$로 수렴한다. 각 $`X_i`$는 $`e^{-\beta k}`$로 빠르게 decay하는 spectrum을 가져도, **평균은 평평한(flat) spectrum**이 된다.

### 1.3 기존 분석에 대한 비판
- Yan et al.(2026)은 **Stein's lemma**를 써서 J-lens를 least-squares map으로 해석했다.
- 저자들은 이를 받아들이지 않는다. Stein's lemma는 **Gaussian 분포를 가정**하는데, hidden state는 **non-Gaussian**이기 때문이다.
- 대신 dynamical system의 **Lyapunov theory**로 J-lens를 해석한다.

---

## 2. Definitions: J-lens와 Lyapunov 관점

### 2.1 J-lens 정의
- $`\mathbf{h}_{\ell,t}(x)`$: input $`x`$, layer $`\ell`$, token 위치 $`t`$의 activation
- $`L`$: final layer, $`\tau`$: lag(미래 token까지의 거리), $`H`$: 평균할 lag 개수, $`c = (x, t)`$: context

**Eq. 1, Jacobian**:

```math
\mathbf{K}(c, \ell, \tau) := \frac{\partial \mathbf{h}_{L,\, t+\tau}(x)}{\partial \mathbf{h}_{\ell,\, t}(x)}
```

**미래 lag 평균, 그리고 context 평균**:

```math
\mathbf{G}_c(\ell, H) = \frac{1}{H} \sum_{\tau=0}^{H-1} \mathbf{K}(c, \ell, \tau),
\qquad
\mathbf{J}_H(\ell) = \mathbb{E}_c \, \mathbf{G}_c(\ell, H)
```

**Eq. 2, lens**:

```math
\mathrm{lens}(\mathbf{h}_{\ell,t}) = \mathrm{softmax}\Big( \mathbf{W}_U \, \mathrm{norm}\big( \mathbf{J}_H(\ell) \, \mathbf{h}_{\ell,t} \big) \Big)
```

### 2.2 Lyapunov exponent
Dynamical system에서 작은 perturbation이 시간에 따라 **얼마나 빨리 커지는지**를 재는 양이다.

```math
\chi = \lim_{T \to \infty} \frac{1}{T} \log \frac{\lVert \delta\mathbf{h}(T) \rVert}{\lVert \delta\mathbf{h}(0) \rVert},
\qquad
\chi_i(T) = \frac{1}{T} \log s_i(T)
```

- 두 번째 식은 Jacobian의 $`i`$번째 singular value $`s_i(T)`$로 정의한 **finite-time Lyapunov exponent**다.
- **Transformer로의 대응**:
  - **Depth(layer)** = 시간(evolution coordinate)
  - **Token position** = 공간(spatial coordinate)
- 그래서 transformer는 "layer를 따라 시간 진화하고, token 축으로 공간적으로 펼쳐진" 시공간 dynamical system처럼 다룰 수 있다.

---

## 3. Theory: J-lens는 Coherent Mode를 고른다

### 3.1 Shared mode와 context-specific mode (Eq. 5)
Jacobian $`\mathbf{K}`$를 모든 context가 공유하는 부분 $`\mathbf{S}(\ell, \tau)`$와 context마다 다른 부분 $`\mathbf{R}(c, \ell, \tau)`$로 나눈다. 각 부분은 mode들의 합이고, mode의 가중치는 $`e^{T\chi + \tau\gamma}`$ 꼴이다.

- $`\gamma_i = \mu_i + \mathrm{i}\,\theta_i`$: $`\mu_i`$는 token 축 방향의 **decay rate**, $`\theta_i`$는 **회전(oscillation) 주파수**
- 미래 lag 평균은 다음 함수로 나타난다.

```math
f_H(\gamma) = \frac{1}{H} \sum_{\tau=0}^{H-1} e^{\tau\gamma}
```

### 3.2 평균이 하는 일
1. **Slow mode 분리** (Eq. 6–7): $`f_H`$는 **빨리 decay하거나 빨리 회전하는 mode를 억제**한다. Token 축 평균은 window의 **zero-frequency 성분**을 뽑아내는 것과 같다.
2. **Incoherent 방향 억제** (Eq. 8): context-specific 부분의 크기를 $`\mathbb{E}_c \lVert \mathbf{R}_c \rVert_2^2 \le M^2`$, $`M = \lVert \mathbf{S}_H \rVert_2`$로 가정한다. 평균한 context-specific 잔여는 $`\eta M^2`$($`0 \le \eta < 1`$)로 bound된다. Context 간 overlap $`\eta`$가 작으면, $`\mathbf{J}_H`$의 leading singular direction은 **shared part $`\mathbf{S}_H`$의 것에 가까워진다.**

→ Context마다 방향이 달라지는 mode는 평균 과정에서 **서로 상쇄**된다.

### 3.3 세 가지 coherence 성질 (Eq. 9–11)
정확히 공유되는 mode가 없는 일반적인 경우로 확장한다. 각 context의 평균 Jacobian을 SVD하면 $`\mathbf{G}_c = \sum_j s_{cj} \mathbf{u}_{cj} \mathbf{v}_{cj}^*`$이고(Eq. 9), 방향 $`\mathbf{v}`$에 대해 세 양을 정의한다(Eq. 10).

**Magnitude**:

```math
a(\mathbf{v}) = \mathbb{E}_{c,\tau} \lVert \mathbf{K}\mathbf{v} \rVert
```

**Lag coherence**:

```math
\rho_\tau(\mathbf{v}) = \frac{\mathbb{E}_c \lVert \mathbf{G}_c \mathbf{v} \rVert}{\mathbb{E}_{c,\tau} \lVert \mathbf{K}\mathbf{v} \rVert} \in [0, 1]
```

**Context coherence**:

```math
\rho_c(\mathbf{v}) = \frac{\lVert \mathbb{E}_{c,\tau} \mathbf{K}\mathbf{v} \rVert}{\mathbb{E}_c \lVert \mathbf{G}_c \mathbf{v} \rVert} \in [0, 1]
```

**Eq. 11**: top singular value는 세 성질의 곱의 최댓값이다. 다음 singular direction들은 앞의 방향들과 orthogonal하다는 조건 아래에서 같은 곱을 최대화한다.

```math
\sigma_1(\mathbf{J}_H) = \max_{\lVert \mathbf{v} \rVert = 1} \; \rho_c(\mathbf{v}) \, \rho_\tau(\mathbf{v}) \, a(\mathbf{v})
```

| 성질 | 의미 | 값이 커지는 조건 | 측정 축 |
|---|---|---|---|
| **Magnitude** $`a`$ | 방향을 따라 작은 변화가 output까지 **얼마나 증폭되는가** | Layer를 지나며 성장(Lyapunov exponent가 큼) | Depth |
| **Lag coherence** $`\rho_\tau`$ | 미래 여러 token에 걸쳐 **같은 방향으로 영향을 주는가** | 회전하지 않고 천천히 decay ($`\theta, \mu \approx 0`$) | Token position |
| **Context coherence** $`\rho_c`$ | **문맥이 바뀌어도** 같은 방향으로 반응하는가 | 문맥에 독립적인 autonomous mode | Context |

![Figure 2. Schematic of magnitude, lag coherence, and context coherence](https://paper-assets.alphaxiv.org/paper-figures/01a175f2-f04e-7af2-b709-cdf96803621e/01a178a5-3361-7b19-88d3-cecd7466c074.png)

*Figure 2 (alphaXiv가 논문에서 추출한 이미지). 세 성질의 개념도. **왼쪽(magnitude)**: perturbation이 한 방향으로는 커지고(큰 χ) 다른 방향으로는 줄어든다(작은 χ). **가운데(lag coherence)**: token 위치를 따라 일정한 lag-coherent 궤적과 흔들리는 lag-incoherent 궤적. **오른쪽(context coherence)**: 문맥이 달라도 같은 방향인 context-coherent 방향과 그렇지 않은 방향.*

### 3.4 Corollary 1.1: singular value ≠ explained variance
- J-lens가 고른 방향은 activation variance를 **아주 적게도, 아주 많이도** 설명할 수 있다. 즉 $`\mathbf{J}_H`$의 singular value는 explained variance를 결정하지 않는다.
- **증명 (Appendix D, 2차원 반례)**: Jacobian을 $`\mathbf{K}(c,\ell,\tau) = s\,(\mathbf{e}_1\mathbf{e}_1^* + \xi_c (-1)^\tau \mathbf{e}_2\mathbf{e}_2^*)`$로 둔다($`\xi_c = \pm 1`$이 같은 확률). 평균하면 $`\mathbf{J}_H = s\,\mathbf{e}_1\mathbf{e}_1^*`$가 되어, **coherent 방향 $`\mathbf{e}_1`$만 살아남고 진동하는 incoherent 방향 $`\mathbf{e}_2`$는 상쇄된다.** 두 방향의 activation variance와는 상관없이 그렇다.
- → Gurnee et al.의 J-space가 activation variance의 **작은 비율만 설명한 것**은 이상한 일이 아니라 이론적으로 자연스러운 결과다.

### 3.5 Coherent mode를 token에 연결하기 (Section 1.1)
$`\mathbf{J}_H = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^*`$이고 $`\mathbf{w}_q`$가 token $`q`$의 unembedding vector일 때, token $`q`$의 lens 방향은:

```math
\mathbf{j}_q := \mathbf{J}_H^* \mathbf{w}_q = \mathbf{V}\boldsymbol{\Sigma}\mathbf{U}^* \mathbf{w}_q = \sum_i \sigma_i \langle \mathbf{u}_i, \mathbf{w}_q \rangle \, \mathbf{v}_i
```

- 각 $`\mathbf{j}_q`$는 right singular mode $`\mathbf{v}_i`$들의 혼합이다.
- $`\sigma_i`$는 coherent response channel의 **세기**, $`\langle \mathbf{u}_i, \mathbf{w}_q \rangle`$는 그 channel이 token $`q`$와 **얼마나 정렬되는지**를 나타낸다.
- **J-space**는 activation을 $`\mathbf{j}_q`$들의 **sparse non-negative combination**으로 근사해서 정의된다. 결국 J-space는 살아남은 coherent mode들에 **token 이름표를 붙인 혼합**을 고르는 셈이다.

### 3.6 ω-J-lens family (Section 1.2)
- Token 축 평균은 **zero-frequency mode로의 Fourier projection**과 같다.
- 주파수 $`\omega`$를 바꾸면(또는 band-pass filter로 일반화하면) **ω-J-lens** family를 얻는다. $`\omega \in [-\pi, \pi]`$이며, token당 sampling 1회의 Nyquist limit으로 정해진다.
- ω-J-lens는 **lag coherence가 $`\omega`$와 맞는 latent 방향**을 선택적으로 키운다.
- 응용: **빠르게 변하는 정보**(예: syntax)와 **천천히 변하는 정보**를 분리할 수 있다.

---

## 4. Toy Model: Hidden Markov Model 과제

세 성질을 **하나씩 독립적으로 조절**할 수 있도록 설계한 next-token prediction 과제다.

### 4.1 Latent dynamics
각 latent $`X^{(i)}`$는 $`n_i`$개 state로 된 **ring** 위에 있다.

```math
X_{t+1} =
\begin{cases}
X_t + v + D_t \bmod n & \text{with prob. } \lambda \\
U_t & \text{with prob. } 1 - \lambda
\end{cases}
```

- $`\lambda`$: **persistence** (state를 유지할 확률) → **magnitude** $`a`$를 조절
- $`D_t \in \{-1, 0, 1\}`$: 확률 $`q, 1-2q, q`$의 diffusion
- $`U_t \sim \mathrm{Uniform}\{0, \dots, n-1\}`$: 무작위 reset
- $`v`$: 정수 velocity. Angular velocity $`\theta = 2\pi v / n`$ → **lag coherence** $`\rho_\tau`$를 조절

### 4.2 Observation
- Reveal 변수 $`B_t \in \{0, 1\}`$: 확률 $`\varepsilon = P(B_t = 1)`$로 latent를 보여준다.
- 여러 latent state를 하나의 $`n`$-ary token index $`Y_t`$로 합친다. Token 수는 $`1 + \prod_i n_i`$이고, 추가된 1개는 관측되지 않은 경우의 **null token**이다.
- 실험에서는 latent **3개**, $`n_i \in \{2, 8\}`$를 썼다.

### 4.3 Keys (Section 2.1)
- 항상 관측되는 **key state** $`W_t \in \{-1, +1\}`$를 추가한다. $`P(W_t = 1) = (1+\psi)/2`$이고, $`\psi \in [0, 1]`$는 **key bias**다.
- Observation은 $`Y_t = X_t + \tfrac{n}{2} \, \mathbf{1}(W_{t-1} = -1)`$가 된다. Model은 token을 해석하려면 **한 step 뒤를 돌아봐야** 한다.
- 이렇게 하면 **context coherence** $`\rho_c`$를 조절할 수 있다. Keyed latent 집합이 $`\mathcal{K}`$일 때 vocabulary는 $`(1 + 2\prod_i n_i) \, 2^{\lvert \mathcal{K} \rvert}`$개다.

### 4.4 조절 knob 정리

| Knob | 조절되는 성질 | 실험 값 |
|---|---|---|
| Persistence $`\lambda`$ | Magnitude $`a`$ | 0.8, 0.9, 0.95 |
| Velocity $`v`$ (→ $`\theta`$) | Lag coherence $`\rho_\tau`$ | $`v = 0, 2, 4`$ → $`\theta = 0, \pi/2, \pi`$ |
| Key bias $`\psi`$ | Context coherence $`\rho_c`$ | 0, 0.75, 1 |

### 4.5 Transformer
- **2-layer decoder-only transformer**, residual dimension **128**, layer당 **4 heads**
- Next-token prediction으로 학습했다(학습 세부 사항은 Appendix F).
- **Figure 3**(이미지 없음): (a) 과제의 graphical model. 세 latent가 각자 ring 위에서 진화하고, $`B_t`$는 $`Y_t`$가 latent를 보여줄지, $`W_{t-1}`$은 보여준 state를 뒤집을지 정한다. (b) Bayes filter의 graphical model. Belief $`m_t`$는 $`\alpha`$로 전파되거나, $`B_t = 1`$일 때 드러난 state로 교체된다.

---

## 5. Bayes Filter의 J-lens: 정확한 해

Toy 과제의 **최적 예측기**인 Bayes filter는 J-lens를 **해석적으로** 계산할 수 있다. 이것이 transformer 결과를 비교할 "정답지"가 된다.

### 5.1 Fourier mode로 쓴 belief
- Fourier mode $`\phi_j(x) = e^{2\pi i j x / n}`$
- Belief: $`m^{(j)}_t := \mathbb{E}[\phi_j(X_{t+1}) \mid Y_1, \dots, Y_t]`$, $`m^{(j)}_0 := 0`$

**Unkeyed update (Eq. 16)**:

```math
m^{(j)}_t = \alpha_j \Big[ (1 - B_t)\, m^{(j)}_{t-1} + B_t \, \phi_j(X_t) \Big],
\qquad
\alpha_j = \lambda \, c_j \, e^{i\theta_j}
```

- $`\theta_j = j\theta`$, $`c_j = 1 - 2q + 2q\cos(2\pi j / n)`$
- 관측이 없으면 belief를 $`\alpha_j`$로 **감쇠·회전**시키고, 관측이 있으면 드러난 state로 **교체**한다.

### 5.2 Keyed filter (Section 3.1)
Key는 관측 가능하고 역변환할 수 있으므로 $`\phi_j(Y_t) = W_{t-1}\phi_j(X_t)`$다.

```math
m^{(j)}_t = \alpha_j \Big[ (1 - B_t) m^{(j)}_{t-1} + B_t \mathbf{1}(W_{t-1} = 1) \phi_j(Y_t) + B_t \mathbf{1}(W_{t-1} = -1) W_{t-1} \phi_j(Y_t) \Big]
```

두 번째 항은 그대로인 state를, 세 번째 항은 key가 뒤집은 state를 **되돌리는** 역할을 한다.

### 5.3 Bayes filter의 J-lens (Section 3.2)

```math
K_j(c, \tau) = \frac{\partial m^{(j)}_{t+\tau}}{\partial \phi_j(X_t)} = \alpha_j^{\tau+1} \prod_{k=1}^{\tau} (1 - B_{t+k}) = \alpha_j^{\tau+1} A_\tau
```

- Lens는 latent와 mode에 대해 **block diagonal**이므로, **각 mode가 그 자체로 singular vector**다.
- **Step당 memory**: $`r_j := \lvert \alpha_j \rvert (1 - \varepsilon)`$
- $`N(c) \ge 1`$: 다음 reveal까지 남은 token 수, $`P(N = k) = (1-\varepsilon)^{k-1}\varepsilon`$
- $`\sum_{\tau=1}^{H} K_j(c,\tau) = \alpha_j S_{j,N(c)}`$, 여기서 $`S_{j,N} := \sum_{\tau=1}^{\min(N-1, H)} \alpha_j^\tau`$

**Large-H 극한에서의 닫힌 해**:

```math
a^{(j)} = \frac{\lvert \alpha_j \rvert \, r_j}{1 - r_j},
\qquad
\rho_\tau^{(j)} = \frac{(1 - r_j)\, \mathbb{E}_c \lvert S_{j,N} \rvert}{r_j},
\qquad
\rho_c^{(j)} = \lvert \psi \rvert \, \frac{r_j}{\lvert 1 - r_j e^{i\theta_j} \rvert \, \mathbb{E}_c \lvert S_{j,N} \rvert}
```

**Singular value (Appendix E, Eq. 44)**:

```math
\sigma^{(j)} = a^{(j)} \rho_\tau^{(j)} \rho_c^{(j)} = \frac{\lvert \alpha_j \rvert \, r_j}{\lvert 1 - r_j e^{i\theta_j} \rvert}
```

(unkeyed, 즉 $`\lvert \psi \rvert = 1`$인 경우)

**해석**:
- $`\theta_j = 0`$(정지한 latent)이면 $`\rho_\tau^{(j)} = 1`$, $`\rho_c^{(j)} = \lvert \psi \rvert`$다. 이때 persistence $`\lambda`$는 **magnitude $`a^{(j)}`$만** 바꾼다.
- $`\theta \ne 0`$이면 $`S_{j,N}`$의 항들이 회전하며 **부분적으로 상쇄**되어 $`\rho_\tau^{(j)}`$가 작아진다.
- 극단적인 경우 $`\theta_j = \pi`$(매 step 부호가 바뀌는 mode)는 $`\rho_\tau = (1 - r_j)/(1 + r_j)`$다.

### 5.4 Transformer와의 연결 (Section 3.3)
- Transformer의 J-lens는 $`\mathbf{W}_U \mathbf{J}_H`$다.
- Output이 softmax를 거치므로 lens는 **log-probability의 변화**를 잰다($`\partial \log p = \partial p / p`$).
- 이것은 $`a`$, $`\rho_\tau`$, $`\rho_c`$의 **순서를 보존하는 양수 재가중**이다. 따라서 Bayes filter에서 예측한 **순위**가 transformer에도 그대로 적용되어야 한다.

---

## 6. Numerical Results

**방법**: 세 latent에서 한 parameter만 바꿔 $`a`$, $`\rho_\tau`$, $`\rho_c`$ 중 하나를 선택적으로 조절한다. 각 latent의 Fourier mode(sin, cos)로 residual stream에 대한 **encoding model**을 fit한다. 그 subspace와 lens singular vector $`\mathbf{v}_i`$(input), $`\mathbf{u}_i`$(output) 사이의 **total cosine similarity**를 계산해, J-lens가 어느 latent를 먼저 골랐는지 본다. Decoding vector로도 같은 분석을 했다.

### 6.1 Magnitude 조절 (Section 4.1, Figure 4)
- **설정**: 세 latent의 persistence $`\lambda = 0.8, 0.9, 0.95`$, $`v = 0`$ ($`\rho_\tau`$, $`\rho_c`$ 고정)
- 모든 transformer가 **최적 Bayes filter만큼 잘 예측**했다.
- Bayes filter의 lens factor가 theory와 일치했다. Lens weight $`\lVert \mathbf{J}_H \mathbf{x} \rVert_2`$는 $`a`$를 따라가고, $`\rho_\tau`$와 $`\rho_c`$는 일정했다.
- **Transformer**도 layer마다 비슷한 factor를 보였고, J-lens가 latent를 **$`a`$ 순서대로** 골랐다(Figure 11–13). $`\mathbf{v}_i`$, $`\mathbf{u}_i`$ 모두 그랬고, decoding vector는 정렬도가 조금 낮았다.

### 6.2 Lag coherence 조절 (Section 4.2, Figure 5)
- **설정**: 8칸 ring, velocity $`v = 0, 2, 4`$ → $`\theta = 0, \pi/2, \pi`$ (완전 coherent → 최대 incoherent)
- J-lens가 **$`\rho_\tau`$ 순서대로** latent를 골랐다. 여러 seed의 transformer에서 일관됐다(Figure 15).
- Transformer에서 lag coherence는 주로 velocity로 조절되었고, $`a`$와 $`\rho_c`$는 덜 민감했다. Decoding vector 결과도 일관됐다(Figure 16).

### 6.3 ω-J-lens sweep (Section 4.2.1, Figure 6)
- Lag-coherence 과제에서 $`\omega`$를 전 범위로 sweep했다(transformer 두 layer와 Bayes filter).
- ω-J-lens는 $`\theta`$, 즉 lag coherence가 **$`\omega`$와 가장 잘 맞는 latent를 선택적으로 키웠다.** Theory 곡선(transformer에 맞게 rescale)과 측정값이 맞았다.

### 6.4 Context coherence 조절 (Section 4.3, Figure 7)
- **설정**: keyed latent마다 $`n = 2`$ state, $`\psi = 0, 0.75, 1`$ (incoherent → coherent). Bayes filter theory에서는 $`\rho_c = \psi`$다.
- Bayes filter와 transformer 모두 $`\psi`$를 따라 latent 순위를 예측대로 매겼다(Figure 17).
- 다만 이 실험은 **noise가 더 컸다.** 여러 lens 방향이 reveal state $`B_t`$와 key state $`W_{t-1}`$에 정렬됐다(Figure 19). → J-lens가 **지도(supervision) 없이 다른 feature도 찾아낼 수 있다**는 뜻이다.
- **Transformer는 $`\rho_c`$의 범위를 압축**했다. $`\psi = 0`$인 latent에서는 Bayes filter보다 훨씬 높고, coherent한 latent에서는 더 낮았다. 저자들은 transformer가 **문맥에 독립적인 정보를 선호**할 수 있다고 해석한다.

### 6.5 두 latent의 순위 역전 예측 (Section 4.4, Figure 8)
- **설정**: key 없는 HMM, 두 latent L0($`\theta_0 = 0`$)와 L1($`\theta_1`$). L0의 persistence $`\lambda_0`$를 sweep하고 $`\lambda_1`$은 고정한다.
- Theory는 J-lens에서 두 latent의 순위가 **뒤바뀌는 persistence $`\lambda_0^*`$** 를 예측한다.
- **Bayes filter**: 다양한 $`\lambda_1`$, $`\theta_1`$에서 예측한 crossing 지점이 실제와 일치했다.
- **Transformer**: $`\lambda \in [0, 1]`$ 전 범위에서 예측이 작은 bias로 일반화됐다. 오차는 **약 0.01**이고, layer마다 반대 방향이었다. $`\sigma_0 - \sigma_1`$는 $`\lambda_0`$에 대해 **단조 증가**했다.

![Figure 8. Predicted singular-value crossing](https://paper-assets.alphaxiv.org/paper-figures/01a175f2-f04e-7af2-b709-cdf96803621e/01a178a5-357d-7b1b-a3de-bb403fe73db1.png)

*Figure 8 (alphaXiv가 논문에서 추출한 이미지). J-lens에서 두 latent의 순위 역전 예측. **(a)** Bayes filter: $`\lambda_0`$를 바꿀 때 두 latent의 singular value(원: 측정, 선: large-H 해석해). 점선은 예측한 $`\lambda_0^*`$. **(b)** $`\theta_1`$에 따른 예측 $`\lambda_0^*`$와 실제 crossing(원). **(c)** Toy transformer 두 layer에서 $`\lambda_0 - \lambda_0^*`$에 따른 $`\sigma_0 - \sigma_1`$. 실선은 평균, 음영은 표준편차이며, 각 곡선은 $`\theta_1`$ 값 4개의 평균이다.*

### 6.6 결과 요약

| 실험 | 조절한 성질 | Bayes filter | Transformer |
|---|---|---|---|
| 4.1 | Magnitude ($`\lambda`$) | Theory와 일치 | 예측 순서대로 선택 |
| 4.2 | Lag coherence ($`\theta`$) | Theory와 일치 | 예측 순서대로 선택 (seed 간 일관) |
| 4.2.1 | ω-J-lens | $`\omega`$와 맞는 latent 강조 | 같음 |
| 4.3 | Context coherence ($`\psi`$) | Theory와 일치 | 예측 순서대로 선택, 단 $`\rho_c`$ 범위 압축 |
| 4.4 | 순위 역전 $`\lambda_0^*`$ | 정확히 예측 | 오차 약 0.01 |

---

## 7. 실제 LLM의 Jacobian (Appendix B)

- **Prompt**: **96개**, 각각 57 tokens로 자름(최소 400자). Dataset별 처음 200개 training example에서 고정 seed로 뽑았다.
- **Dataset**: Wikipedia, CodeParrot, CodeXGLUE, Gretel SQL, OpenWebMath, GSM8K, DialogSum, CNN/DailyMail, poems, LEDGAR, recipes. (본문에는 "16개 dataset, 각 6개 prompt"라고 되어 있지만 목록에는 11개만 있다.)
- **Jacobian**: middle block $`\ell`$에서 마지막 block으로, **한 token 앞**을 본다. $`J_{c,t} = \partial h_{L-1}^{(c)}(t+1) / \partial h_\ell^{(c)}(t)`$
- **개수**: model당 $`N = 96 \times 8 = 768`$개 matrix. 평균 $`\bar{J} = \tfrac{1}{N}\sum_{c,t} J_{c,t}`$
- **분석**: top-100 singular value를 log scale로 바꾼 뒤 PCA (Figure 1)
- **Figure 9**: LLM 4개의 Jacobian singular value spectrum과 추정한 power law (음영: Jacobian 간 표준편차)
- **Figure 10**: top-100 left/right singular vector subspace를 Grassmannian 거리 기반 PCA로 투영
- 사용한 LLM 4개의 이름은 읽은 범위에서 확인하지 못했다.

---

## 8. Discussion

### 8.1 세 성질의 물리적 의미
- **Magnitude**: depth를 따라 지수 $`\chi`$로 성장한다. 시간에 따른 성장을 재는 **표준 Lyapunov exponent**와 같은 역할이다.
- **Lag coherence**: token 위치를 따라 측정한다. Mode는 속도 $`\mu`$로 decay하고 주파수 $`\theta`$로 회전한다. 가장 coherent한 mode는 $`\theta, \mu \approx 0`$이며, 물리학의 **hydrodynamic Lyapunov mode**와 연결된다.
- **Context coherence**: $`\rho_c`$를 최대화하면 **autonomous mode**($`\rho_c \approx 1`$)가 선택된다. Transformer가 context-incoherent latent에 대해 Bayes filter보다 높은 $`\rho_c`$를 가진 feature를 학습한 것은 **symbolic processing의 mechanism**일 수 있다고 저자들은 제안한다.

### 8.2 Gurnee et al.(2026) 결과에 대한 설명

| Gurnee et al.의 관찰 | 이 논문의 설명 |
|---|---|
| J-space가 activation variance의 **10% 이하**만 설명 | Singular value는 explained variance와 무관 (Corollary 1.1) |
| "Y를 풀면서 X에 집중하라" 같은 과제에서 X에 대한 feature를 찾음 | X는 여러 token에 걸쳐 유지되므로 **lag coherence**가 높음 |
| 평균하면 **넓게 말로 표현 가능한(verbalizable) 개념**이 나옴 | 여러 문맥에서 일관된 개념은 **context coherence**가 높음 |

### 8.3 기타 시사점
- **Belief state**: 알려진 belief state를 linear decoding하는 기존 방법(Shai et al.)과 달리, J-lens는 belief state를 **지도 없이** 찾아낼 수 있을 것이다.
- **Future directions**:
  - ω-J-lens로 **local(빠르게 변하는) 정보**를 잡기
  - $`\rho_c`$를 **최소화**해서 context-incoherent 정보를 분리하기
  - **생물학적 neural data**에 적용하기
- **Global Workspace**: 이 성질들이 "Global Workspace"를 의미하는지에 대해서는 입장을 취하지 않는다.

---

## 9. Limitations

- **Toy model만 검증**: 과제가 배우기 쉽고, coherence를 조절하기 쉬운 toy model에서만 이론을 검증했다(저자 언급).
- **실제 LLM 분석은 제한적**: LLM 4개에 대해서는 Jacobian spectrum의 성질(평균의 효과, power law)만 보였고, 세 coherence 성질을 직접 측정하지는 않았다.
- **Appendix B의 불일치**: dataset 수가 본문 설명(16개)과 목록(11개)이 다르다.
- **Context coherence 실험의 noise**: lens 방향이 의도한 latent 외에 reveal·key state에도 정렬되어 결과가 더 noisy했다.

---

## 10. Takeaways

1. **"평균"이 핵심이다**: J-lens는 많은 context와 미래 위치에 대해 Jacobian을 평균하는데, 이 평균이 **coherent하지 않은 mode를 걸러내는 filter** 역할을 한다. 그래서 J-lens가 찾은 feature는 "가장 중요한" feature가 아니라 **"가장 일관된" feature**다.
2. **세 성질의 곱**: J-lens의 top 방향은 magnitude × lag coherence × context coherence를 최대화한다. 어떤 feature가 J-lens에 잡히지 않는다면, 셋 중 하나가 작기 때문일 수 있다(예: 빠르게 바뀌는 syntax 정보).
3. **J-space가 variance를 적게 설명해도 괜찮다**: singular value와 explained variance는 별개다. J-space의 작은 variance 비율을 "중요하지 않다"는 증거로 읽으면 안 된다.
4. **Dynamical systems 언어로 interpretability 보기**: transformer를 "depth = 시간, token = 공간"인 시공간 시스템으로 보면 Lyapunov exponent, hydrodynamic mode 같은 물리학 도구를 interpretability에 쓸 수 있다.
5. **새 도구 ω-J-lens**: 주파수를 바꿔 lens를 거는 것만으로 빠른 정보와 느린 정보를 분리할 수 있다. 기존 J-lens는 $`\omega = 0`$인 특수한 경우다.
6. **해석적 정답지의 가치**: Bayes filter처럼 J-lens를 정확히 계산할 수 있는 model을 두면, transformer의 J-lens가 무엇을 골랐는지 **정량적으로 예측하고 검증**할 수 있다(순위 역전 지점까지 오차 약 0.01).

---

## References & Links

- Paper: https://arxiv.org/abs/2610.31955
- alphaXiv: https://www.alphaxiv.org/abs/2610.31955

<sub>작성 메모: arXiv HTML 원문의 본문(Section 1–5)과 Appendix A–E 일부를 기준으로 정리했다. 페이지가 매우 길어 Appendix F–J(transformer 학습 세부, alignment 값, 보충 결과)와 순위 역전 임계값 공식(Eq. 46)은 읽지 못했다. 그림은 GitHub에서 표시되도록 alphaXiv가 논문에서 추출한 이미지 2개(Fig. 2, 8)만 넣었고, 나머지 그림은 설명으로 대신했다. 수식은 GitHub 전용 문법(```` ```math ```` 블록과 `` $`...`$ ``)으로 썼다. 캡션은 원문을 번역하지 않고 요약해서 다시 썼다.</sub>
