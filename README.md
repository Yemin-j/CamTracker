---

# MTMDC Multi-Camera Tracking Framework

An end-to-end PyTorch framework for  
**Person Detection → ReID Feature Extraction → Single-Camera Tracking → Multi-Camera Global Tracking (MCTA)**  
designed specifically for the **AI-Hub MTMDC Dataset**.

The project follows MMDetection/MMTracking’s modular philosophy  
but is **lightweight, fully custom, and highly extensible**.

---

## Key Features

-  **Faster R-CNN + FPN** detector  
-  **ROI-based ReID embedding head (256-dim)**  
-  **Single-camera ReID-assisted SORT tracking**  
-  **Multi-camera Global Tracking (MCTA)** 
-  YAML config system + ETA/Loss logging  

---

## Dataset: AI-Hub MTMDC

The MTMDC dataset contains synchronized multi-camera surveillance videos  
with frame-level person annotations.

### Characteristics
- Raw video: **30 FPS**
- Annotation frame_id: **23 FPS × 320 sec = 7362 frames**
- Per-frame `.json` with:
  - bounding boxes  
  - track IDs  
  - person identities  

---

## Merge Split Archives

```bash
cat MTMDC.zip.part* > MTMDC.zip
unzip MTMDC.zip 
```
---

## Installation
```
conda create -n mtmdc python=3.9 -y

conda activate mtmdc

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

pip install opencv-python tqdm pyyaml matplotlib pandas
```

---

## Results
- MOTA
- IDF1
- The result video will be displayed later

---

## Citation
@misc{mtmdc2025,
  title        = {MTMDC Multi-Camera Tracking Framework},
  author       = {Yemin-J},
  year         = {2025},
  howpublished = {GitHub Repository},
}
---
