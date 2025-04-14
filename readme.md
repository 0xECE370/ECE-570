
# Masked Autoencoders on CIFAR-10

This project implements a **Masked Autoencoder (MAE)** based on **Vision Transformers (ViTs)** for self-supervised image reconstruction on the CIFAR-10 dataset. It evaluates how patch size, masking ratio, and training duration affect reconstruction quality, using both **quantitative (perceptual loss)** and **qualitative (image visualization)** metrics.

The model uses a masked perceptual loss computed from intermediate VGG features to guide semantically meaningful reconstructions, especially under high masking conditions.

---

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/0xECE370/ECE-570.git
cd ECE-570
```

2. (Optional) Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Running the Model

Open `main.ipynb` and run the cells **or** modify the `runMAE` function as needed.

### Example 1: Run with patch size 4, 3 masking ratios, and 200 epochs
```python
runMAE(patch_size=4, epoch_size=200, masking_ratios=[25, 50, 75])
```

### Example 2: Run with patch size 8 and high masking ratios
```python
runMAE(patch_size=8, epoch_size=300, masking_ratios=[75, 80, 85])
```

### Example 3: Fast test run with patch size 2 and short training
```python
runMAE(patch_size=2, epoch_size=10, masking_ratios=[50])
```
