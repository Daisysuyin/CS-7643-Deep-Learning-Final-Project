# 🧠⚛️ Hybrid Quantum–Classical Vision Models with CUDA-Q  
**CS 7643: Deep Learning — Final Project**

This project explores whether structured quantum components can enhance image-classification performance when integrated into modern deep-learning architectures. The study evaluates several classical and hybrid models on the Fashion-MNIST dataset, including a CUDA-Q–based Quantum Transformer that replaces dot-product attention with a quantum similarity kernel.

The full implementation includes classical CNN and ViT baselines, hybrid models with variational quantum circuits (VQCs), and a Quantum Vision Transformer (Quantum-ViT) using simulated quantum self-attention. All experiments, logs, checkpoints, and generated plots are included for full reproducibility.

---

## 📁 Project Structure
``` 
cs7643_project_quantum_transformer/
│
├── .vscode/
│   └── settings.json
│
├── __pycache__/                        # Python cache files
│
├── checkpoints/
│   ├── classical_cnn_best.pt
│   ├── classical_cnn_last.pt
│   ├── classical_vit_best.pt
│   ├── classical_vit_last.pt
│   ├── hybrid_cnn_best.pt
│   ├── hybrid_cnn_last.pt
│   ├── hybrid_vit_best.pt
│   ├── hybrid_vit_last.pt
│   ├── quantum_vit_qattn_best.pt
│   └── quantum_vit_qattn_last.pt
│
├── results/
│   ├── classical_cnn.csv
│   ├── classical_vit.csv
│   ├── hybrid_cnn.csv
│   ├── hybrid_vit.csv
│   ├── quantum_vit_qattn.csv
│   └── logs/                           # additional structured logs
│
├── figures/
│   ├── loss_curves.png
│   └── val_accuracy.png
│
├── configs/
│   ├── classical_cnn_config.json
│   ├── classical_vit_config.json
│   ├── hybrid_cnn_config.json
│   ├── hybrid_vit_config.json
│   └── quantum_vit_qattn_config.json
│
├── src/
│   ├── data_utils.py
│   ├── models.py
│   ├── quantum_attention.py
│   ├── quantum_head.py
│   ├── train_classical.py
│   ├── train_hybrid.py
│   ├── train_quantum_vit.py
│   └── utils.py
│
├── notebooks/
│   ├── run_all_experiments.ipynb
│   └── checkpoint_analysis.ipynb
│
├── CS7643_project_report.pdf
├── README.md
├── environment.yml
├── requirement.txt
└── requirements-checkpoint.ipynb         # metadata autosave from Jupyter

```
---

## 🚀 Models Implemented

### **1. Classical Baselines**
- **CNN**: Two convolutional blocks + MLP head  
- **Vision Transformer (ViT)**:
  - Patch size 7×7 → 16 tokens  
  - 2 Transformer encoder layers  
  - 64-dim token embeddings  

---

### **2. Hybrid Models with Variational Quantum Circuits (VQC)**

Implemented using **CUDA-Q** and trained end-to-end with PyTorch via the parameter-shift rule.

- Hybrid CNN + VQC  
- Hybrid ViT + VQC  

Each VQC uses:
- 4 qubits  
- 2 layers  
- Ring entanglement  
- Expectation value over \( H = \sum_i Z_i \)

These models test whether quantum circuits can effectively replace classifier heads.

---

### **3. Quantum Transformer (Quantum-ViT)**

Inspired by the official **CUDA-Q Quantum Transformer tutorial**.

Replaces dot-product attention with a **quantum similarity kernel**:
$s_{ij} = \langle \psi(x_{ij}, \theta) \mid H \mid \psi(x_{ij}, \theta) \rangle$


This enhances token interactions using quantum embedding, while preserving the ViT architecture.

---

## 📊 Experimental Results

### **Dataset**
- Fashion-MNIST (28×28 grayscale, 60k/10k split)

### **Hardware**
- NVIDIA RTX 5090  
- CUDA-Q 0.13.0  
- PyTorch 2.9.1  

### **Validation Results**

| Model                 | Best Val Acc | Best Val Loss |
|----------------------|--------------|----------------|
| Classical CNN        | 10.0%        | 2.303          |
| Classical ViT        | 10.0%        | 2.303          |
| Hybrid CNN + VQC     | 16.4%        | 2.295          |
| Hybrid ViT + VQC     | 10.0%        | 2.303          |
| **Quantum-ViT (QAttn)** | **84.1%**     | **0.435**       |

### **Observations**
- Classical baselines remained near random-guess performance in the captured final runs.  
- VQC classifier heads underperform due to strong information bottlenecks.  
- **Quantum attention dramatically improves performance**, achieving 84.1% accuracy.  
- Results indicate that quantum modules are more effective inside attention mechanisms than as classifiers.

---

## 📈 Figures & Logs

Plots are automatically generated in:
```
  figures/loss_curves.png
  figures/val_accuracy.png
```
Training logs (CSV) are saved in:
```
  results/*.csv
```
Checkpoints (best + last) in:
```
  checkpoints/
```

---

## 🛠️ Installation

### **1. Clone the repository**
```
git clone https://github.com/<your-username>/cs7643_project.git
cd cs7643_project
```
### **2. Create the conda environment**
```
conda env create -f environment.yml
conda activate cs7643_project
```

### **3. Verify CUDA-Q**
```
python -c "import cudaq; print(cudaq.__version__)"
```

## ▶️ Running Experiments
### Option A — Unified Notebook

Open: run_experiments.ipynb


Runs:

- CNN baseline
- ViT baseline
- Hybrid models
- Quantum Transformer

### Option B — Command Line

Example:
```
python src/train_quantum_vit.py --epochs 3 --batch-size 32
```
## 📚 References
@misc{cudaqtransformer,
  author       = {{NVIDIA}},
  title        = {{CUDA-Q} Quantum Transformer Tutorial},
  howpublished = {\url{https://nvidia.github.io/cuda-quantum/latest/applications/python/quantum_transformer.html}},
  year         = {2024}
}

@article{gebru2018datasheets,
  title={Datasheets for Datasets},
  author={Gebru, Timnit and Morgenstern, Jamie and Vecchione, Briana and others},
  journal={Communications of the ACM},
  year={2021}
}

@article{schuld2021quantumkernels,
  title={Supervised Quantum Machine Learning Models Are Kernel Methods},
  author={Schuld, Maria},
  journal={Quantum},
  year={2021}
}


