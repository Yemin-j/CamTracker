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

## Dataset: VisDrone-MOT2019

The VisDrone-MOT2019 dataset is a large-scale aerial multi-object tracking benchmark captured by UAVs (drones).
It contains diverse urban scenarios with varying altitudes, lighting conditions, object densities, and camera motions.

### Characteristics

Raw video FPS: 25–30 FPS

Resolution: 1920×1080 or 2688×1520 (depending on the sequence)

Scenes: Crowded urban streets, intersections, highways, residential zones

Annotations:

- Per-frame bounding boxes (x, y, w, h)
- Track IDs (consistent identity across frames)
- Occlusion ratios
- Visibility information
- Object categories (pedestrians, vehicles, bicycles, tricycles, etc.)

Sequence Split:
- Train: 40 sequences
- Validation: 16 sequences
- Test: 40 sequences (labels with held)

## Installation
```
conda create -n mtmdc python=3.9 -y  
conda activate mtmdc

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118  
pip install opencv-python tqdm pyyaml matplotlib pandas
```

---

## Results
1 Epoch
- MOTA : 18.5
- IDF1 : 30.5

https://github.com/Yemin-j/MultiCamTracker/releases/download/tracking-vis-v1/uav0000086_00000_v_track.mp4

---

## Citation

@misc{mtmdc2025,  
  title        = {MTMDC Multi-Camera Tracking Framework},  
  author       = {Yemin-J},  
  year         = {2025},  
  howpublished = {GitHub Repository},  
}

---
