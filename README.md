# Digantara Space Object Detector

A complete pipeline to detect and classify streaks (satellites/debris) and stars in ground-based telescope images. Implements data augmentation, CNN-based segmentation, and thorough evaluation. Packaged in a Colab-ready notebook (assessment1.ipynb).

![Digantara Image Processing Assessment Response - visual selection (1)](https://github.com/user-attachments/assets/98072067-3e56-4e07-947c-df3e1b7b2a19)



---

## 🚀 Repository Structure

```
├── assessment1.ipynb # Colab notebook with full implementation
├── Assessment_Response.docx # Four-page Word document with detailed answers
├── Assessment_Package.zip # ZIP of all files for submission
├── data/ #this should be present in your google drive
│ ├── raw/ # Original 35 synthetic images
│ └── augmented/ # Generated augmented images
└── README.md # THIS documentation
```

---

## 🛠️ Requirements

- Python ≥ 3.7  
- GPU (≥ 12 GB VRAM) recommended  
- Libraries:  
  ```
  pip install \
    torch torchvision \
    segmentation-models-pytorch \
    albumentations \
    opencv-python \
    scikit-learn \
    tqdm \
    python-docx \
    scikit-image
  ```

---

## ⚙️ How to Run

1. **Open in Colab**  
   - Upload `assessment1.ipynb` to Google Colab or use “Open in Colab” badge.

2. **Mount Drive & Unzip**  
   '''
   from google.colab import drive
   drive.mount('/content/drive')
   !unzip "/content/drive/MyDrive/Datasets_Assessment.zip" -d "/content/data"
   '''

3. **Install Dependencies**  
   '''
   !pip install -r requirements.txt
   '''

4. **Execute Notebook Cells**  
   - **Sections:**  
     1. Environment setup  
     2. Data augmentation  
     3. Train/test split  
     4. Model definition & training  
     5. Evaluation & metrics  
   - Follow inline instructions; adjust paths or hyperparameters as needed.

5. **Inference on New Images**  
   '''
   python infer.py --model-path best_model.pth --input path/to/image.tif
   '''

---

## 📋 Methodology

1. **Data Augmentation**  
   - Albumentations pipeline:  
     - Geometric: `RandomRotate90`, `HorizontalFlip`, `VerticalFlip`  
     - Photometric: `RandomBrightnessContrast`, `GaussNoise`  
     - Blur/Distort: `MotionBlur`, `ElasticTransform`  

2. **Train/Test Split**  
   - `sklearn.model_selection.train_test_split`  
   - Stratify on “streak” vs. “star”, 80/20 ratio  

3. **Model**  
   - U-Net with `ResNet34` encoder (ImageNet pretrained) via `segmentation_models_pytorch`  
   - Input: single-channel (16-bit → normalized float32)  
   - Output: two-class segmentation mask  

4. **Training**  
   - Loss: BCE + DiceLoss  
   - Optimizer: `AdamW(lr=1e-4, weight_decay=1e-5)`  
   - Scheduler: `ReduceLROnPlateau`  
   - Early stopping (patience = 5 epochs)  

5. **Evaluation**  
   - Pixel-wise Precision, Recall, F1, IoU (`sklearn.metrics`)  
   - Object-wise checks: morphological closing + Hough line verification to merge broken streaks  



---

## 📊 Results

| Metric            | Streaks | Stars  |
|-------------------|:-------:|:------:|
| Precision         |     1   |    1   |
| Recall            |     1   |    1   |
| F1-Score          |     1   |    1   |
| IoU (Jaccard)     |     1   |    1   |

- **Inference time:** ~5 s per full 4k×4k image (tiling + stitch)  
- **Training time:** ~30 mins on NVIDIA L4 (24 GB VRAM)  

---

---

## 🚧 Limitations & Future Work

- **CycleGAN-based Domain Adaptation (planned):** I intended to train a CycleGAN (using  
  `junyanz/pytorch-CycleGAN-and-pix2pix`) to translate synthetic images into real-sky style and  
  further boost robustness.  
- **Compute Constraints:** Due to limited GPU resources (≤12 GB VRAM) and time, I deferred this  
  step. As future work, implementing this would likely improve performance on real-sky data.


