Отлично. Ниже production-ready версии **AGENTS.md** и **Harness Engineering Checklist** для системы Identity-Preserved Photo Editing.

---

# 📁 AGENTS.md

**Project:** Identity-Preserved Photo Editing System
**Owner:** AI Platform Team
**Primary KPI:** Identity Similarity ≥ 0.85

---

# 1. System Overview

Система построена как multi-agent pipeline с жесткими quality gates.

```
User Request
    ↓
Orchestrator Agent
    ↓
Generation Agent
    ↓
Identity Validator
    ↓
Quality Gates
    ↓
Delivery / Reject
```

---

# 2. Agent Registry

---

## 🧠 1. Orchestrator Agent

**Role:** Управление пайплайном и маршрутизация.

### Inputs

* reference_image
* edit_prompt
* edit_type
* control_type
* optional mask

### Responsibilities

* Валидировать входные данные
* Запустить embedding extraction
* Определить:

  * img2img / inpaint
  * controlnet type
  * LoRA loading
* Запустить generation agent
* Запустить validators

### Failure Modes

* No face detected
* Invalid mask
* Missing LoRA

### Output

* job_id
* generation status
* metadata

---

## 🎨 2. Generation Agent

**Role:** Image synthesis.

### Stack

* SDXL Base
* SDXL Refiner
* IP-Adapter Face
* Identity LoRA
* ControlNet
* Inpainting (optional)

### Responsibilities

* Apply identity conditioning
* Inject LoRA
* Apply ControlNet
* Respect mask
* Generate image
* Save intermediate outputs (optional debug mode)

### Config Defaults

```
CFG: 6–8
Steps: 25–35
LoRA scale: 0.6–1.0
Refiner strength: 0.2–0.4
```

### Output

* generated_image
* inference metadata

---

## 👤 3. Identity Validator Agent

**Role:** Проверка стабильности личности.

### Stack

* ArcFace embedding extractor
* Cosine similarity module

### Responsibilities

* Extract embedding from reference
* Extract embedding from generated image
* Compute similarity
* Reject if < threshold

### Thresholds

```
similarity ≥ 0.85 → pass
0.75–0.85 → soft warning
< 0.75 → reject
```

---

## 🧾 4. Prompt Adherence Agent

**Role:** Semantic validation.

### Stack

* CLIP encoder

### Responsibilities

* Compute image-text similarity
* Reject if below threshold

### Threshold

```
CLIP score ≥ 0.8
```

---

## 🧪 5. Artifact Detection Agent

**Role:** Quality control.

### Checks

* Face distortion
* Hand anomaly
* Blurriness
* Extreme noise

### Reject Conditions

* Face artifact probability > 0.3
* Blurry score above threshold
* Hand anomaly detected

---

## 🗂 6. Dataset Preparation Agent (Training)

**Role:** Identity LoRA dataset prep.

### Responsibilities

* Face alignment
* Deduplication
* Caption generation
* Data validation

---

## 🏋️ 7. LoRA Training Agent

**Role:** Train identity LoRA.

### Config

```
Rank: 8–32
LR: 1e-4 – 5e-5
Epochs: 10–20
Regularization images: enabled
Token: <person_token>
```

### Success Criteria

* No identity drift
* No overfitting
* Similarity ≥ 0.85

---

# 3. Agent Communication Protocol

### Message Schema

```json
{
  "job_id": "uuid",
  "reference_image_url": "...",
  "prompt": "...",
  "config": {...},
  "status": "processing | rejected | completed",
  "metrics": {
    "identity_similarity": 0.91,
    "clip_score": 0.83,
    "artifact_score": 0.02
  }
}
```

---

# 4. Quality Gates Order

1. Face detected?
2. Identity similarity ≥ threshold?
3. CLIP adherence ≥ threshold?
4. Artifact detection pass?
5. Latency acceptable?

---

# 5. Observability Hooks

Each agent must log:

* execution time
* GPU memory usage
* error codes
* rejection reason

---

# 6. Failure Handling Strategy

| Failure        | Strategy                    |
| -------------- | --------------------------- |
| Identity drift | Lower strength + re-run     |
| Face collapse  | Apply face mask lock        |
| High latency   | Reduce steps                |
| LoRA overfit   | Retrain with regularization |

---

---

# 🛠 Harness Engineering Checklist

Production readiness checklist для ML system.

---

# 1️⃣ Data & Identity Harness

### ☐ Face Detection Reliability

* [ ] 99% detection rate
* [ ] Edge cases tested (profile, occlusion)

### ☐ Identity Similarity Benchmark

* [ ] Same person similarity ≥ 0.9 (baseline)
* [ ] Different person ≤ 0.6
* [ ] Pose invariant tests
* [ ] Lighting variation tests

### ☐ Embedding Drift Tests

* [ ] Background change
* [ ] Hairstyle change
* [ ] Clothing change
* Drift ≤ 0.05

---

# 2️⃣ Generation Harness

### ☐ Prompt Robustness

* [ ] Neutral prompt
* [ ] Complex styling
* [ ] Extreme lighting
* [ ] Multi-object scene

### ☐ CFG Sensitivity Sweep

* [ ] 4, 6, 8, 10 tested
* [ ] Optimal selected

### ☐ LoRA Scale Sweep

* [ ] 0.5–1.2 tested
* [ ] No identity collapse

---

# 3️⃣ Quality Gate Harness

### ☐ Artifact Detection Accuracy

* [ ] False positive < 5%
* [ ] False negative < 5%

### ☐ CLIP Threshold Calibration

* [ ] Distribution analyzed
* [ ] Threshold justified statistically

---

# 4️⃣ Performance Harness

### ☐ Latency Benchmark

* [ ] Single GPU
* [ ] Concurrent jobs
* [ ] P95 measured
* [ ] P99 measured

### ☐ Memory Stability

* [ ] No VRAM leak after 100 runs
* [ ] Stable GPU utilization

---

# 5️⃣ Failure Injection Tests

### ☐ Corrupted image

### ☐ No face input

### ☐ Extreme prompt

### ☐ Very small image

### ☐ Mask mismatch

System must:

* Reject gracefully
* Log reason
* Return structured error

---

# 6️⃣ Monitoring & Alerts

### ☐ Metrics tracked

* identity_similarity
* clip_score
* rejection_rate
* latency

### ☐ Alerts configured

* identity drift spike
* rejection rate > 10%
* latency spike > 15 sec

---

# 7️⃣ Reproducibility

### ☐ Seed control

### ☐ Model version tracking

### ☐ LoRA version control

### ☐ Config hash logging

---

# 8️⃣ A/B Testing Harness

### ☐ Baseline vs LoRA

### ☐ IP-Adapter only vs LoRA+IP

### ☐ ControlNet on/off

### ☐ User rating comparison

---

# 9️⃣ Security & Abuse

### ☐ Prompt filtering

### ☐ Deepfake misuse policy

### ☐ Identity ownership verification

### ☐ Rate limiting

---

# 🔥 Production Readiness Gate

System can be marked **Production Ready** only if:

* Identity Similarity ≥ 0.85
* Prompt adherence ≥ 0.8
* Face distortion < 3%
* P95 < 10 sec
* Rejection rate < 8%
* No memory leak
* All harness tests passed