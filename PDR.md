составь PDR-tasks c DOR/DOD :

📌 PDR: Identity-Preserved Photo Editing System
1. Executive Summary

Цель:
Создать систему генерации и редактирования изображений, которая:

Принимает: reference photo + text prompt

Генерирует: фотореалистичный результат

Гарантирует: стабильную идентичность человека

Поддерживает: локальное редактирование (волосы, одежда, фон, эмоции)

Работает серийно (персональные фотосессии / бренд-амбассадор)

2. Бизнес-цели
Use Cases

AI-фотосессии

Генерация маркетингового контента

Персональные аватары

Серийные брендовые кампании

UGC automation

KPI

Identity Similarity ≥ 0.85 (ArcFace cosine similarity)

Face distortion rate < 3%

Prompt adherence ≥ 0.8 (CLIP score)

P95 latency < 10 сек

User satisfaction ≥ 4.5/5

3. Архитектура системы
🧠 Core Model Stack
1️⃣ Base Generator

SDXL Base

SDXL Refiner

2️⃣ Identity Conditioning

IP-Adapter Face (SDXL)

ArcFace embedding extractor (для identity control)

Identity LoRA (обученная на 20–50 фото)

3️⃣ Structural Control

ControlNet Depth

ControlNet OpenPose (если меняется поза)

ControlNet SoftEdge (для одежды)

4️⃣ Local Editing

SDXL Inpainting

Авто-сегментация:

Face parsing

Hair mask

Clothing mask

Background segmentation

4. Архитектурная схема (логика пайплайна)
Input:
  - reference_image
  - edit_prompt
  - edit_mask (optional)

Step 1: Extract identity embedding (ArcFace)
Step 2: Apply IP-Adapter Face conditioning
Step 3: Inject Identity LoRA
Step 4: Apply ControlNet (depth/pose)
Step 5: Inpainting (если локальное редактирование)
Step 6: Generate via SDXL
Step 7: Refiner pass
Step 8: Identity validation
Step 9: Quality gates
5. Обучение Identity LoRA
Dataset Requirements

20–50 изображений

Разные:

ракурсы

освещение

эмоции

фоны

Без:

фильтров

тяжелой ретуши

обрезанного лица

Preprocessing

Face alignment

Удаление дубликатов

Авто-caption + ручная чистка

Баланс разнообразия

Training Config

Rank: 8–32

LR: 1e-4 – 5e-5

Epochs: 10–20

Regularization images: yes

Text token: <person_token>

6. Метрики качества
6.1 Identity Consistency

ArcFace cosine similarity

Face embedding drift

Pose-invariant similarity

6.2 Prompt Adherence

CLIP similarity

Semantic instruction match

6.3 Image Quality

FID (optional)

Face artifact detection

Hand anomaly detection

7. Инференс-конфигурация
Recommended Defaults
Parameter	Value
CFG	6–8
Steps	25–35
Strength (img2img)	0.3–0.6
Refiner	0.2–0.4
LoRA scale	0.6–1.0
8. Инфраструктура
Hardware

GPU: RTX 4090 / A100 / H100

VRAM: 24GB minimum

Batch size: 1–4

Backend

FastAPI

Redis queue

Celery workers

MinIO (image storage)

Postgres (metadata)

9. Риски и Mitigation
Risk	Mitigation
Лицо “плывет”	Identity LoRA + lower strength
Модель меняет лицо при смене фона	Inpainting + face mask
Переобучение LoRA	Regularization images
Prompt ломает лицо	Hard face lock mask
High latency	Quantization / scheduler tuning
10. Roadmap
Phase 1 – Baseline (2 недели)

SDXL + IP-Adapter

ControlNet depth

Identity similarity validation

Phase 2 – Identity LoRA (2 недели)

Dataset collection

LoRA training

Benchmark

Phase 3 – Production Hardening (2 недели)

Quality gates

Auto masking

Monitoring

A/B testing

11. Расширения

Multi-character support

Video consistency (frame-to-frame identity)

GAN-based face refinement

Reinforcement tuning по user feedback