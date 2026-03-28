# Calorie Estimator AI 🍔🥗

An image-based food nutrition analysis system that uses deep learning to identify food items from photos and estimate their macronutrient content (calories, protein, carbohydrates, and fats).

## Project Overview

This project implements a two-stage pipeline:

1. **Food Classification** — A fine-tuned CNN (EfficientNet/ResNet-50) classifies the food item in the image
2. **Nutrition Estimation** — The predicted food category is mapped to a nutritional database to retrieve macronutrient values

## Team Members

- Arsham Alishirkouhi
- Gonul Eda Koker
- Michael Hoshen

_San José State University — CMPE 189, Spring 2026_

## Repository Structure

```
Calorie_Estimator_AI/
├── notebooks/
│   ├── exploration.ipynb        # EDA and dataset exploration
│   └── model_comparison.ipynb  # ResNet-50 vs EfficientNet comparison
├── src/
│   ├── preprocessing.py         # Data loading and augmentation
│   ├── model.py                 # Model architecture and training
│   └── predict.py               # Inference pipeline
├── app/
│   └── app.py                   # Streamlit web interface
├── data/
│   └── nutrition_database.csv   # Nutritional values per food category
├── requirements.txt
└── .gitignore
```

## Dataset

We use the [Food-101 dataset](https://www.kaggle.com/dansbecker/food-101), which contains 101,000 images across 101 food categories.

## Setup & Installation

```bash
git clone https://github.com/arsham-shirkouhi/Calorie_Estimator_AI.git
cd Calorie_Estimator_AI
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run app/app.py
```

## Model Performance (Preliminary)

| Model        | Accuracy | Final Train Loss |
| ------------ | -------- | ---------------- |
| ResNet-50    | 91.5%    | 0.415            |
| EfficientNet | 90.56%   | 0.247            |

Both models achieved relatively similar accuracy after 3 epochs of training on a 10,000-image subset of Food-101. ResNet-50 achieved slightly higher accuracy (91.5%) while EfficientNet-B0 converged faster and reached a lower average training loss (~0.25 per batch vs ~0.41 per batch), suggesting better generalization potential with more training data. Based on these results, EfficientNet-B0 is the preferred model for our final pipeline due to its efficiency and lower loss.

## Tech Stack

- Python 3.10+
- PyTorch & torchvision
- Streamlit
- Pandas, NumPy, Matplotlib
