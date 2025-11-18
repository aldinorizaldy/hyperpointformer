<div align="center">

# **HyperPointFormer**
### **Multimodal Fusion in 3-D Space With Dual-Branch Cross-Attention Transformers**

<img width="1000" src="https://github.com/user-attachments/assets/16e5c637-26ae-42a3-935d-7e7b15315434" />

**HyperPointFormer** is a fully 3D deep learning framework for multimodal 3D Point Clouds data for urban classification.

</div>

---

## 🌐 Overview

Most existing approaches convert 3D point clouds into 2D raster grids, limiting spatial reasoning and preventing true 3D predictions. **HyperPointFormer** removes this constraint by directly processing raw point clouds.

Key features:

- **Fully 3D multimodal fusion** (hyperspectral + LiDAR/photogrammetry)
- **Dual-branch Transformer**  
  - Geometry branch  
  - Spectral branch  
- **Cross-attention fusion** across modalities and across scales
- **End-to-end 3D predictions** that can be projected to 2D, unlike the reverse

---

## ⚙️ Installation

HyperPointFormer follows the installation approach of  
[`Pointnet_Pointnet2_pytorch`](https://github.com/yanx27/Pointnet_Pointnet2_pytorch/)  
with additional [`DGL library`] (https://www.dgl.ai/).

## ⚙️ Training and Testing
```
python train_semseg.py
python test_semseg.py
```

Cite paper here:
> A. Rizaldy, R. Gloaguen, F. E. Fassnacht and P. Ghamisi, "HyperPointFormer: Multimodal Fusion in 3-D Space With Dual-Branch Cross-Attention Transformers," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 18, pp. 21254-21274, 2025, https://doi.org/10.1109/JSTARS.2025.3595648

