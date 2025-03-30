# 🌿 Deep Learning for Rare Species Classification

**Master’s in Data Science 2024/2025 – Group Project**

This repository contains the source code, data processing pipeline, and model training workflow for our deep learning project on **rare species image classification**. The goal is to predict the **biological family** of a given organism based on its image.

The project is based on the **BioCLIP** dataset, introduced in [Stevens et al., 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Stevens_BioCLIP_A_Vision_Foundation_Model_for_the_Tree_of_Life_CVPR_2024_paper.html), which provides rich metadata and high-resolution imagery sourced from the Encyclopedia of Life (EOL).

---

## 📁 Project Structure

```
📦 project-root/
├── data/
│   ├── raw/              # Original images and metadata.csv
│   ├── processed/        # Train/Val/Test splits
├── notebooks/            # Jupyter notebooks for EDA, training, evaluation
├── models/               # Saved model checkpoints
├── src/                  # Core modules (dataloaders, training loops, utils)
├── outputs/              # Evaluation results, confusion matrices, etc.
├── report/               # Scientific report (PDF or LaTeX)
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## 📊 Objective

Train a model to **classify the family** of an organism from its image using deep learning. Your model must generalize well to unseen data, so the test set should remain untouched during training and validation.

---

## 🔍 Dataset

- **Source:** Encyclopedia of Life (EOL)
- **Metadata:** Includes image paths, kingdom, phylum, and family labels
- **Challenge:** Highly imbalanced classes, inter-class visual similarity, fine-grained classification

---

## 🧠 Approach

We experimented with various techniques, including:

- Image preprocessing and augmentations
- Baseline CNNs and transfer learning with pre-trained models (e.g., ResNet, EfficientNet, ViT)
- Custom training and evaluation pipelines
- Hyperparameter tuning with Keras Tuner

For more details, refer to the [report](report/) folder.

---

## 📈 Evaluation

Our evaluation strategy includes:

- Accuracy and top-k accuracy
- Confusion matrices
- Error analysis (e.g., misclassified species by visual similarity)
- Visual inspection of predictions

---

## 🛠️ Setup & Dependencies

### Requirements

```bash
pip install -r requirements.txt
```

### Python Libraries Used

- `tensorflow` / `keras`
- `pandas`
- `numpy`
- `matplotlib` / `seaborn`
- `scikit-learn`
- `opencv-python`

---

## 📄 Deliverables

- ✅ Source code and data processing pipeline
- ✅ Trained model and evaluation scripts
- ✅ 5-page scientific report (in English)
- ✅ Metadata and group info files (as per Moodle submission guidelines)

---

## 👥 Team

> Replace the placeholder with your actual names and student IDs.

- **Benedikt Ruggaber** – 20240500 
- **Daan Van Holten** – 2024002  
- **Jose Cavaco** – 20240513  
- **Philippe Duntranoit** – 2024004  
- **Joshua Wehr** – 20240501 

---

## 🧪 Future Work

Given more time, we would explore:

- Fine-tuning foundation models like BioCLIP or CLIP directly
- Multi-label classification for hierarchical taxonomy (e.g., kingdom → phylum → family)
- Integration of textual metadata as multimodal input
- Class balancing strategies (e.g., focal loss, SMOTE for embeddings)

---

## ⚠️ Academic Integrity

The collaboration between groups and the use of AI-generated code is strictly forbidden as per course policy.  
Violation of this rule will result in immediate disqualification from the curricular unit in both epochs.
