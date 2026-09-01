---
title: AerialSegmentation
emoji: 🌖
colorFrom: pink
colorTo: red
sdk: gradio
sdk_version: 6.26.0
python_version: '3.11'
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Local Docker Run

If you want to test or host this on your laptop without burning Hugging Face quota, use Docker.

```bash
docker build -t aerialsegmentation .
docker run --rm -p 7860:7860 aerialsegmentation
```

Then open `http://localhost:7860`.

Native Windows `python app.py` is not the easy path for this project because Detectron2 needs a Linux-style build setup. Docker or WSL is the practical local option.

# AI-Based Automated Urban Parcel Mapping and Cadastral Feature Extraction System

**Problem Statement ID:** 26012 | **Organization:** Ministry of Rural Development, Dept. of Land Resources (DoLR) | **Theme:** Smart Automation

## Overview
Automates extraction of parcel boundaries, building footprints, and road networks from drone/orthorectified imagery using AI-based image segmentation — reducing the manual effort involved in cadastral surveys and urban land record preparation.

## Core Novelty
1. **Three specialized models instead of one generalist model** — a land-use classifier, a building-vs-parcel separator, and a dedicated boundary-delineation model, each trained for its own task rather than one model doing everything.
2. **Automated topology validation** — every generated parcel is checked against neighboring parcels and existing GIS land records for overlaps, gaps, and mismatches before being marked ready for use.
3. **Confidence-based human-in-the-loop routing** — each parcel carries a confidence score; low-confidence parcels are automatically flagged and routed to a field surveyor instead of being silently accepted.
4. **Continuous feedback loop** — surveyor corrections are captured and fed back into retraining, so accuracy improves with real-world use instead of staying frozen after the first training run.

**Why this closes the gap:** Vision APIs and ML models already prove feature extraction is possible; government drone programmes already prove the imagery pipeline works. What's missing is the bridge between raw AI output and legacy GIS workflows — a validated, editable handoff a land-records officer can trust. That's the gap this fills.

## Current Implementation Status (MVP)
What's actually built and running today:
- Single Mask R-CNN (Detectron2, ResNet-50 FPN) instance segmentation model, 2 classes: `building`, `boundary_or_road`
- Gradio web demo — upload a drone image, get back segmented masks + a detection summary
- CPU fallback (contour-based) when GPU/Detectron2 isn't available
- Dockerized for reproducible local/hosted deployment (Hugging Face Spaces)

## Roadmap to Full System
- Split `boundary_or_road` into separate parcel-boundary and road classes; add dedicated land-use classification model
- Georeferencing — ingest GeoTIFF ORI/DSM/DTM with CRS, output real-world coordinates instead of pixel space
- Vectorization — convert raster masks to clean GeoJSON/Shapefile polygons
- Topology validation module — overlap/gap detection against existing GIS parcel layers
- Confidence scoring + surveyor review queue
- Feedback-driven retraining pipeline
- Web-GIS visualization and editing dashboard

## Tech Stack
- **Model:** Detectron2 (Mask R-CNN, COCO-InstanceSegmentation config)
- **Interface:** Gradio
- **Deployment:** Docker, Hugging Face Spaces
- **Core libraries:** PyTorch, OpenCV, NumPy

## Setup

### Docker (recommended)
```bash
docker build -t aerialsegmentation .
docker run --rm -p 7860:7860 aerialsegmentation
```
Then open `http://localhost:7860`.

### Notes
- Native Windows `python app.py` is not the easy path — Detectron2 needs a Linux-style build, so Docker or WSL is the practical local option.
- Model weights (`model_final.pth`, ~351MB) are tracked via Git LFS — run `git lfs install` before cloning, or `git lfs pull` after, to fetch the real weights.

## Repository Structure
```
├── app.py             # Gradio app + inference pipeline
├── model_final.pth    # Trained Mask R-CNN weights (Git LFS)
├── requirements.txt   # Python dependencies
├── Dockerfile          # Container build
├── run_local.bat       # Windows helper script for Docker build+run
```