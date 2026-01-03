# Graph WaveNet for Deep Spatial-Temporal Graph Modeling
This is the original pytorch implementation of FP-GTCN in the following paper: 
[FP-GTCN: A GCN-TCN-based Soft Sensor with a Flexible Prior Process Graph for Trifluoroethanol Distillation]



## ⚠️Important Note
This project uses PyTorch Geometric (PyG) with CUDA 11.7, which requires special installation steps. 
Do NOT run `pip install -r requirements.txt` directly. Instead, follow these steps:
### Step1:Install PyTorch (with CUDA 11.7) and its related packages first:
```
# pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
```
### Step2: Install PyG dependencies compatible with PyTorch 2.0.1 (CUDA 11.7):
```
# pip install torch-scatter==2.1.2+pt20cu117 \
             torch-sparse==0.6.18+pt20cu117 \
             torch-cluster==1.6.3+pt20cu117 \
             torch-spline-conv==1.2.2+pt20cu117 \
   -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
```
### Step3:Install the remaining project dependencies (from the requirements.txt file):
```
# pip install -r requirements.txt
```
Following these steps ensures that the correct CUDA-enabled PyTorch and PyG packages are installed, preventing any compatibility issues.
