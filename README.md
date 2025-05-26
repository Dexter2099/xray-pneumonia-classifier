# X-ray Pneumonia Classifier

## Project Overview
This repository provides a simple convolutional neural network (CNN) built with PyTorch to classify chest X-ray images as **Pneumonia** or **Normal**. The code is meant as an introductory example of medical image analysis and includes utilities for training, evaluation and Grad-CAM visualization.

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/xray-pneumonia-classifier.git
   cd xray-pneumonia-classifier
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the Chest X-ray dataset from Kaggle using the [Kaggle API](https://github.com/Kaggle/kaggle-api) or from [Kaggle - Chest X-ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). Extract the `chest_xray` folder into `data/raw/` so that the path looks like `data/raw/chest_xray/train/...`. The dataset is large and is **not included** in this repository.

## Training
Train the model from the command line:
```bash
python main.py --mode train
```
Model checkpoints are saved to `outputs/checkpoints/`.

## Evaluation
After training, evaluate the model:
```bash
python main.py --mode eval
```
Evaluation metrics and confusion matrix images will be written to `outputs/visualizations/`.

## Grad-CAM demo
Run Grad-CAM on a specific image to visualize what the model focuses on:
```bash
python main.py --mode gradcam --image path/to/image.jpeg
```
The resulting heatmap is stored in `outputs/visualizations/`.

## Notebook walkthrough
- `notebooks/01_EDA.ipynb` contains a short exploratory data analysis.
- `notebooks/02_Model_Demo.ipynb` offers a guided demo showing how to load the trained model and run predictions on sample images.

---
Created as a beginner-friendly yet medically relevant deep learning project.
