# MTMDC Multi-Camera Tracking Framework

## Introduction
This project provides an end-to-end PyTorch framework for **person detection**, **Re-ID embedding extraction**, **single-camera tracking**, and **multi-camera global tracking (MCTA)**.  
It is built for the **AI-Hub MTMDC (Multi-Target Multi-Domain Camera)** dataset and follows the modular philosophy of MMDetection/MMTracking while keeping the entire pipeline lightweight and fully custom.

---

## Major Features
- **Faster R-CNN + FPN** detector
- **ROI-based ReID Head** (256-d embedding)
- **ReID-assisted Single-Camera Tracking** (SORT-style)
- **Multi-Camera Global Tracking (MCTA)** via tracklet building and embedding association
- **Annotation-driven frame sampling** (23 FPS annotation → 30 FPS video alignment)
- **Iteration-based validation** (Single + Multi)
- **Checkpoint Manager** (iter checkpoints + best single + best MCTA)
- **Extensive export tools**: CSV, MOT, COCO, per-camera AVI, merged multi-camera AVI
- **YAML configuration** & structured logging (ETA, iter progress, losses)

---

## Data (AI-Hub MTMDC)
The dataset provides synchronized multi-camera surveillance videos with person-level annotations.

**Key characteristics**
- Raw video: **30 FPS**
- Annotation: **23 FPS (7362 annotated frames per scenario)**
- Per-frame JSON labels containing:
  - Bounding boxes
  - Tracking IDs
  - Person-level identities

**Merge split archives**
```bash
cat MTMDC.zip.part* > MTMDC.zip
unzip MTMDC.zip

**Directory structure**
videos/train/s01/camera01.avi
annotations/train/s01/camera01/*.json

## Installation
conda create -n mtmdc python=3.9 -y
conda activate mtmdc

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install opencv-python tqdm pyyaml matplotlib pandas

##  Results

The framework provides:

Single-Camera Tracking

MOTA, IDF1, FP, FN, IDSW

Multi-Camera Global Tracking (MCTA)

Global IDF1

Global MOTA

Global ID consistency across cameras

Output formats

CSV tracking results

MOTChallenge text files

COCO-Tracking JSON

Per-camera AVI tracking

Multi-camera merged AVI

Outputs are stored under:

results/<timestamp>/
    ├── single/<iter>/
    ├── mcta/<iter>/
    └── checkpoints/

## Citation
@misc{mtmdc2025,
  title        = {MTMDC Multi-Camera Tracking Framework},
  author       = {Your Name},
  year         = {2025},
  howpublished = {GitHub Repository},
}
