# Safe Personal RSI: Evaluating and Repairing Personal-Agent Updates

> **Safe Personal RSI**: personal agent가 스스로 instruction을 고쳐 나갈 때(self-improvement), 그 update가 **privacy나 owner authorization을 약화시키지 않는지** 실제로 실행해서 검사하고, 문제가 있으면 **repair**한 뒤에만 adopt하는 update gate.

| 항목 | 내용 |
|---|---|
| Authors | Jiapeng Pei, Yuhang Yao, Sitan Liu, Hanxin Chen, Hao Li, Guogeng Li, Jiapei Liu, Ruijie Chen, Hongshan Lin, Yang Mo, Tianyu Shi, Shangding Gu |
| Affiliations | CMU, UC San Diego, HKU, McGill University, UC Berkeley 등 |
| arXiv | [2611.23651](https://arxiv.org/abs/2611.23651) (v1, 2026-11-18, cs.SE) |
| 대상 시스템 | **Kicker**: personal health assistant agent |
| Models | **GPT-5.6 Luna**, **Qwen3-14B**, **Qwen2.5-7B-Instruct** |
| Code | 공개 저장소 링크 없음 (alphaXiv 기준) |

> ⚠️ **출처 안내**: arXiv 원문 페이지는 요청 한도 제한으로 세 번 모두 열지 못했다. 이 문서는 alphaXiv 논문 페이지(초록, overview, 실험 결과, 그림 설명, 한계)를 기준으로 정리했다. 원문에만 있는 세부 설정(정확한 attack family 목록, utility 가중치 등)은 빠져 있을 수 있다.

---

## 목차

1. [TL;DR](#tldr)
2. [Problem: Self-improving Personal Agent의 위험](#1-problem-self-improving-personal-agent의-위험)
3. [Method: Safe Personal RSI](#2-method-safe-personal-rsi)
4. [Benchmark & Metrics](#3-benchmark--metrics)
5. [Experiment 1: Controlled Admission](#4-experiment-1-controlled-admission)
6. [Experiment 2: Repair Confirmation (GPT-5.6 Luna)](#5-experiment-2-repair-confirmation-gpt-56-luna)
7. [Experiment 3: Transfer to Stronger Attacks (Qwen3-14B)](#6-experiment-3-transfer-to-stronger-attacks-qwen3-14b)
8. [Runtime Probes](#7-runtime-probes)
9. [Related Work](#8-related-work)
10. [Limitations](#9-limitations)
11. [Takeaways](#10-takeaways)

---

## TL;DR

- **Setting**: personal assistant(**Kicker**)가 자신의 **reusable instructions(harness)** 를 고쳐 가며 스스로 개선한다(**RSI**, recursive self-improvement). 이 논문에서 self-improvement는 **prompt update로만 제한**되고, model weight와 runtime code는 고정이다.
- **Problem**: instruction을 고치면 task 처리는 좋아질 수 있지만, **무엇을 공개하는지(disclosure)**, **누구의 요청을 따르는지(authorization)**, **행동을 어떻게 보고하는지(reporting)** 도 함께 바뀐다.
- **Method (Safe Personal RSI)**: update를 자동으로 adopt하지 않고 **테스트할 결정**으로 취급한다.
  1. Candidate와 현재(parent) harness를 **나란히 실행**한다.
  2. Failure를 **replay**하고, 바뀐 instruction을 **잠시 되돌려(undo) 원인을 찾는다**.
  3. **Repair**한 뒤 **safety와 utility 검사**를 통과해야만 adopt한다. 아무 버전도 통과하지 못하면 **현재 instruction을 유지**한다.
- **Results**
  - **Controlled admission**: 의도적으로 위험한 edit 2종에 대한 update 시도 **21건 중 20건을 거부**했다.
  - **Repair (GPT-5.6 Luna)**: attack violation **10.7% → 0%**, safe task completion **89.7% → 100%**. 단, 손으로 쓴 넓은 authority rule도 같은 결과를 냈다.
  - **Transfer (stronger attacks)**: Qwen2.5-7B-Instruct에서 만든 repair를 Qwen3-14B에 그대로 옮겼더니 violation **69.0% → 17.9%**, safe completion **19.5% → 64.4%**. 다만 violation이 **완전히 사라지지는 않았다.**
- **Runtime probe**: 평가한 시스템 snapshot에서 **local OAuth scope check가 누락**된 것을 발견했다.

---

## 1. Problem: Self-improving Personal Agent의 위험

### 1.1 Kicker와 harness
- **Kicker**는 email, calendar, health 기록, marketplace를 다루는 **personal health assistant**다.
- Agent의 행동은 이름이 붙은 **reusable prompt instruction들의 집합**, 즉 **harness**로 정해진다.
- **Self-improvement**는 이 harness를 고치는 것이다. 이런 edit는 task 처리를 개선할 수 있지만, 아래처럼 다른 행동도 함께 바꾼다.

### 1.2 무엇이 문제인가
Instruction edit는 성능만 바꾸지 않는다. 다음도 함께 바뀐다.

| 바뀌는 것 | 예시 위험 |
|---|---|
| **Disclosure** | 개인 health 정보를 제3자에게 보내는 draft 작성 |
| **Authorization** | owner가 아닌 사람의 요청을 owner 요청처럼 따름, 이전 승인을 다른 요청에 재사용 |
| **Reporting** | 실제로 한 행동을 owner에게 정확히 보고하지 않음 |

### 1.3 예시 사례 (GPT-5.6 Luna, development case)
- **Repair 전 update**: 3번 실행 중 **2번**, 수신자에게 **private health information**이 담긴 draft를 만들었다.
- **Repair 후 instruction**: 3번 모두 **disclosure draft를 만들지 않고**, owner의 task를 완료했다. 내용을 **요약하고, 전달하기 전에 owner에게 물어보는** 방식이었다.
- ⚠️ 저자들은 이것이 **집계 결과가 아니라 예시(illustration)** 라고 밝힌다. 실험 중 실제 **전송(delivery)은 꺼져 있었고**, **clinical advice의 품질은 평가하지 않았다.**

---

## 2. Method: Safe Personal RSI

### 2.1 핵심 아이디어
Reusable instruction은 assistant를 더 유능하게 만들 수 있지만, **누구를 authorized로 대하는지**와 **무엇을 보고하는지**도 바꿀 수 있다. 그래서 gate는 제안된 변경을 **자동으로 adopt할 대상이 아니라, 테스트해야 할 결정**으로 다룬다.

### 2.2 Procedure

| Step | 내용 |
|---|---|
| **1. Propose** | Harness(이름 붙은 reusable prompt instruction 집합)에 대한 edit를 제안한다 |
| **2. Parent comparison** | **Candidate harness**와 **현재(parent) harness**를 같은 test에 실행한다. Parent가 원래 갖고 있던 failure도 포함한다 |
| **3. Failure replay** | Failure를 다시 실행해 **재현되는지** 확인한다 |
| **4. Localization by undo** | 바뀐 instruction을 **하나씩 임시로 되돌려** 어떤 변경이 문제를 일으키는지 찾는다 |
| **5. Repair** | **Category-specific clause**를 추가하거나 **LLM rewrite**로 고치고, 다시 검증한다 |
| **6. Admission** | Safety, task handling, 실행 완료 여부를 평가한 뒤 adopt 여부를 결정한다 |

```mermaid
flowchart TD
    P["Proposed harness edit"] --> C["Run candidate and parent harness<br/>on the same tests"]
    C --> R["Replay failures"]
    R --> U["Temporarily undo changed instructions<br/>to localize the problem"]
    U --> F["Repair: category-specific clause<br/>or LLM rewrite"]
    F --> A{"Admission checks<br/>risk, utility, completion"}
    A -->|"pass"| Y["Adopt new harness"]
    A -->|"fail"| N["Keep current harness"]
    V["Confirmed unauthorized disclosure"] -->|"veto"| N
```

### 2.3 Admission rules
Rule-based이고 threshold는 경험적으로 정했다.

| Rule | 조건 |
|---|---|
| **Run 수** | 최소 **5 runs**, 모두 **완료**되어야 함 |
| **Risk** | Suite-normalized **maximum category risk ≤ 0.1**, 또는 parent의 값 이하 |
| **Utility** | 설정된 **utility criterion**을 만족해야 함 |
| **Veto** | **Unauthorized disclosure**가 확인되면 repair 후에도 **adopt 불가** |

- 모든 버전이 실패하면 **현재 harness를 계속 쓴다.** 즉 실패해도 기존보다 나빠지지 않는 쪽을 기본값으로 한다.

---

## 3. Benchmark & Metrics

### 3.1 Executable benchmark
- **29개 case**: email, calendar, health, marketplace 영역
- **추가 probe**:
  - **Authorization reuse**: 이전에 받은 승인을 다른 요청에 재사용하는지
  - **Memory**: memory를 통해 정보가 새거나 조작되는지
  - **Tool-output injection**: tool 결과에 숨겨진 지시를 따르는지
- **Attack strength**: 일반 attack(**standard**)과 더 강한 attack(**stronger**) 두 수준으로 평가했다.
- **실행 환경**: **synthetic account**와 외부 서비스를 썼다. Mail을 읽고 draft를 저장할 수는 있었지만 **실제 전송은 하지 않았고**, **transaction도 꺼져 있었다.**
- ⚠️ **Loop-excluded attack family 13개**는 benchmark 개발 중에 이미 알려져 있었고, 탐색 과정에서도 다시 쓰였다. 완전히 처음 보는 공격은 아니라는 뜻이다.

### 3.2 Metrics
- **Attack violation rate**: 계획된 attack run 중 flag가 붙은 비율

```math
\text{Violation rate} = \frac{\#\,\text{flagged attack runs}}{\#\,\text{planned attack runs}}
```

- **Safe task completion**: 다음 세 조건을 모두 만족한 run의 비율이다. 분모는 authorized control을 포함한 **87 runs 전체**다.
  1. 실행이 **완료**됨
  2. 정당한 task에 대한 audit 결과가 **positive**
  3. **Violation flag가 없음**
- **Weighted proxy utility**: task 성능에 **cost proxy, latency, confirmation 횟수**에 대한 고정 penalty를 반영한 utility다. 단, search와 repair에 든 **전체 비용은 포함하지 않는다.**
- ⚠️ **Attack flag는 실제 세계의 violation이 입증된 것은 아니다.** Proxy utility도 이 protocol에 특화된 값이다.

---

## 4. Experiment 1: Controlled Admission

### 4.1 Setup
- **Models**: 3개 model
- **Risky edits 2종** (일부러 위험하게 만든 instruction 변경):
  - **Autonomy edit**: 이름으로 보아 agent의 자율성을 높이는 방향의 변경
  - **Stress edit**: 논문이 "stress"라고 부르는 두 번째 위험 변경
  - (alphaXiv 페이지에는 두 edit의 구체적인 문구가 나와 있지 않다.)
- **비교**:
  - **Direct adoption (Utility-only)**: utility만 보고 edit를 그대로 adopt
  - **Safe Personal RSI**: gate를 거쳐 adopt 여부 결정

### 4.2 Results
- Safe Personal RSI는 update 시도 **21건 중 20건을 거부**했다.
  - **12건**은 risk check에서, **8건**은 utility check에서 거부됐다.
- 유일하게 adopt된 것은 **GPT-5.6 Luna의 autonomy edit**(repair 없이 통과)였다. 나머지 5개 model-edit 조건은 모두 **initial harness를 유지**했다.

**Qwen3-14B 결과** (attack flag는 140 runs 기준)

| 조건 | Attack flags | Weighted proxy utility |
|---|---|---|
| Initial harness | 21 / 140 | – |
| Direct adoption, **autonomy** edit | **39** / 140 | .652 |
| Direct adoption, **stress** edit | **55** / 140 | .518 |
| **Safe Personal RSI** (initial harness 유지) | **21** / 140 | **.789** |

- 위험한 edit를 그대로 adopt하면 flag가 **21 → 39, 55**로 늘고, utility도 오히려 **떨어졌다.**
- **GPT-5.6 Luna, stress 조건**: Safe Personal RSI **0 flags**, direct adoption **12 flags**

![Figure 1. Controlled admission across three models](https://paper-assets.alphaxiv.org/paper-figures/01a1f19f-9484-76d9-86b5-bee53ebe1395/01a1f43a-1067-768a-b49b-aee720314f2f.png)

*Figure 1 (alphaXiv가 논문에서 추출한 이미지). 3개 model에서 direct adoption(Utility-only)과 Safe Personal RSI의 attack flag(140 runs 중)와 weighted proxy utility 비교. Qwen3-14B에서 차이가 가장 크다(direct adoption 39, 55 flags vs Safe Personal RSI 21 flags). Flag는 실제 violation이 입증된 것이 아니고, proxy utility는 이 protocol에 특화된 값이다.*

---

## 5. Experiment 2: Repair Confirmation (GPT-5.6 Luna)

- **설정**: standard attacks, attack run **84개**, completion run **87개**

| Arm | Attack violation | Safe task completion |
|---|---|---|
| **Utility-only** (repair 안 한 update) | 9/84 (**10.7%**) | 78/87 (**89.7%**) |
| **Fixed rule** (손으로 쓴 넓은 authority rule) | 0/84 (**0%**) | 87/87 (**100%**) |
| **Safe Personal RSI** (admitted repair) | 0/84 (**0%**) | 87/87 (**100%**) |

- Repair가 violation을 **완전히 없애면서** task completion도 **100%** 로 올렸다.
- ⚠️ 저자들은 **손으로 쓴 명확한 authority rule도 똑같은 결과**를 냈다는 점을 짚는다. 이 결과만으로는 이 procedure가 **잘 쓴 규칙 하나보다 낫다고 말할 수 없다.**

---

## 6. Experiment 3: Transfer to Stronger Attacks (Qwen3-14B)

- **설정**: **stronger attacks**. Repair를 **Qwen2.5-7B-Instruct에서 생성**한 뒤 **Qwen3-14B에 그대로 적용**했다. Target model에서는 repair나 admission loop를 **다시 돌리지 않았다.**
- 모든 arm에 **같은 operational guidance**를 줬다.

| Arm | Attack violation | Safe task completion |
|---|---|---|
| **Utility-only** (repair 안 한 update) | **69.0%** | **19.5%** |
| **Fixed rule** (credential-only 규칙, Luna 규칙보다 좁음) | 66.7% | 24.1% |
| **Safe Personal RSI** (transferred repair) | **17.9%** | **64.4%** |

- 다른 model에서 만든 repair가 **model을 넘어 transfer**되어 violation을 크게 줄이고 safe completion을 3배 이상 올렸다.
- 좁은 손글씨 규칙(credential-only)은 거의 효과가 없었다.
- ⚠️ 하지만 **violation이 17.9% 남아 있다.** 완전한 방어가 아니다.

![Figure 2. Repair and transfer confirmation](https://paper-assets.alphaxiv.org/paper-figures/01a1f19f-9484-76d9-86b5-bee53ebe1395/01a1f43a-10e1-762c-8355-2f6f91639e14.png)

*Figure 2 (alphaXiv가 논문에서 추출한 이미지). GPT-5.6 Luna(standard attacks)와 Qwen3-14B(stronger attacks)에서 Utility-only, Fixed-rule, Safe Personal RSI 세 arm의 attack violation rate와 safe task completion. Luna: violation 10.7% / 0% / 0%, completion 89.7% / 100% / 100%. Qwen3-14B: violation 69.0% / 66.7% / 17.9%, completion 19.5% / 24.1% / 64.4%.*

---

## 7. Runtime Probes

Instruction 수준 평가와 별개로, 실행 중인 시스템 자체를 검사하는 probe도 돌렸다.

- **Memory-entry deletion**: memory 항목 삭제 동작을 검사했다.
- **OAuth scope**: 평가한 시스템 snapshot에서 **local OAuth scope check가 빠져 있는 것**을 발견했다.

→ Prompt 수준의 gate만으로는 막을 수 없는 **runtime/infrastructure 수준의 취약점**도 존재한다는 점을 보여준다.

---

## 8. Related Work

| 연구 | 내용 | Safe Personal RSI와의 차이 |
|---|---|---|
| **GEPA** | Execution trace로부터 prompt를 evolve해 **task performance**를 높임 | **Parent comparison**, **undo 기반 failure localization**, 그리고 **safety와 utility를 분리한 adoption 결정**을 추가했다 |
| **A Theory of Reliable Self-Evolution for Agent Harnesses** | **Confidence 기반 validation rule**로 adoption에 대한 보장을 제공 | 이 gate는 **point estimate**를 쓰며 **통계적 safety 보장은 없다** |

---

## 9. Limitations

- **통계적 보장 없음**: admission이 **point estimate**에 기반한다.
- **Safety 원칙을 스스로 발견한 것은 아님**: repair prompt에 이미 **일반적인 authorization rule**이 들어 있었다. 그래서 이 procedure가 safety 원칙을 독자적으로 찾아낸다는 것을 보여주지는 못한다.
- **잔여 violation**: repair를 적용한 Qwen3-14B에도 flag가 남아 있다. 또 repair를 만든 원래 model인 Qwen2.5-7B-Instruct는 **자기 최종 평가에서 authorized task에 실패**했다.
- **Synthetic 환경**: synthetic account와 외부 서비스를 썼고, 실제 전송과 거래는 하지 않았다.
- **Utility의 범위**: proxy utility는 cost, latency, confirmation에 고정 penalty를 주지만, **search와 repair에 든 전체 작업량은 반영하지 않는다.**
- **범위 제한**: 결과는 **이 두 가지 고정 edit**와 **경험적 threshold**에 한정된다. Repair는 violation을 **줄였지만 없애지는 못했다.**

---

## 10. Takeaways

1. **Self-improvement는 safety regression의 통로가 될 수 있다**: instruction 하나를 고치는 것만으로 agent가 누구를 신뢰하고 무엇을 공개하는지가 바뀐다. Weight를 건드리지 않는 prompt 수준의 RSI도 검증 없이 adopt하면 위험하다.
2. **"Adopt by default"가 아니라 "Keep by default"**: gate는 통과하지 못하면 현재 harness를 유지한다. 그 결과 위험한 update 21건 중 20건을 막았고, Qwen3-14B에서는 안전성과 utility를 **둘 다** 지켰다.
3. **Parent와 비교하고 원인을 찾아라**: candidate만 평가하는 것이 아니라 parent와 나란히 돌리고, 변경을 하나씩 되돌려 원인을 찾는 방식은 일반 software의 **regression test + bisect**와 같은 발상이다.
4. **Repair는 transfer될 수 있다**: 작은 model(Qwen2.5-7B)에서 만든 repair가 더 큰 model(Qwen3-14B)에서도 violation을 69.0% → 17.9%로 줄였다.
5. **좋은 규칙 하나와 정직하게 비교하라**: Luna 실험에서 손으로 쓴 넓은 authority rule이 같은 성능을 냈다는 점을 저자들이 스스로 밝힌다. 자동화된 procedure의 가치는 **규칙을 미리 알 수 없는 상황**에서 판단해야 한다.
6. **Prompt gate만으로는 부족하다**: OAuth scope 누락처럼 runtime 수준의 문제는 instruction 검사로 잡을 수 없다. Agent safety에는 **infrastructure 수준의 권한 검사**가 함께 필요하다.

---

## References & Links

- Paper: https://arxiv.org/abs/2611.23651
- alphaXiv: https://www.alphaxiv.org/abs/2611.23651

<sub>작성 메모: arXiv 원문(HTML, PDF)은 요청 한도 제한으로 열지 못해, alphaXiv 논문 페이지(초록, AI overview, 실험 결과, 그림 캡션, 한계)와 openresearch.sh 요약을 기준으로 정리했다. 원문에만 있는 세부 사항은 빠졌을 수 있다. 그림 2개는 alphaXiv가 논문에서 추출해 올린 이미지에 링크했다. 수식은 GitHub 전용 문법(```` ```math ````)으로 썼다. 캡션은 원문을 번역하지 않고 요약해서 다시 썼다.</sub>
