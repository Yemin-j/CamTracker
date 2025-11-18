# MTMDC-MultiCamera-Tracking

A unified PyTorch-based pipeline for **Object Detection + Re-Identification +  
Single-Camera Tracking + Multi-Camera Tracking (MCTA)**  
designed for the **AIHub MTMDC (Multi-Target Multi-Domain Camera) Dataset**.

This framework is heavily inspired by **MMDetection/MMTracking**  
while maintaining a **lightweight, fully customized implementation**.

---

## 🚀 Features

- **Faster R-CNN + FPN Detector**
- **ROI-based ReID Embedding Head**
- **Single-Camera Tracking (SORT-like + ReID matching)**
- **Multi-Camera Tracking (Tracklet building + Global ID Association)**
- **23FPS Annotation ↔ 30FPS Video Alignment 처리**
- **Iteration-level Validation (Single + Multi MCTA)**
- **Best-model Checkpointing**
- **CSV / MOT / COCO / AVI Visualization Export**
- **YAML-based Config System**
- **Logger-based Training Status (ETA, Loss, Iter progress)**

---

## 📂 Project Structure


---

## 📦 Installation
conda create -n mtmdc python=3.9 -y
conda activate mtmdc

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

pip install opencv-python tqdm pyyaml matplotlib pandas


---

## 📘 Dataset (AIHub MTMDC)

### 1. Download 영상 + annotation

AIHub → MTMDC 링크에서 다운로드  
(데이터가 분할 압축되어 있으므로 아래 명령어로 병합)

```bash
cat MTMDC.zip.part* > MTMDC.zip
unzip MTMDC.zip

data/
  ├── videos/
  │      ├── train/s01/camera01.avi
  │      ├── train/s01/camera02.avi
  │      └── ...
  └── annotations/
         ├── train/s01/camera01/*.json
         ├── train/s01/camera02/*.json
         └── ...

python train.py --config configs/mtmdc.yaml

results/<timestamp>/
    ├── train.log
    ├── checkpoints/
    │       ├── iter_3000.pt
    │       ├── best_single.pt
    │       └── best_mcta.pt
    ├── single/
    │       ├── 3000/
    └── mcta/
            ├── 3000/
