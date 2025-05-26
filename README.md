# X-ray Pneumonia Classifier

This project builds a Convolutional Neural Network (CNN) model using PyTorch to classify chest X-ray images as either **Pneumonia** or **Normal**. It is designed for beginners looking to explore medical imaging and deep learning.

## 🩺 Project Motivation

Pneumonia is a potentially serious lung infection. Early detection through chest X-ray interpretation is critical. This project mimics a clinical workflow where a model assists radiologists by classifying X-ray scans. The dataset is sourced from [Kaggle - Chest X-ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

## 📂 Project Structure

- `data/`: Raw and processed image data
- `notebooks/`: Jupyter notebooks for EDA
- `src/`: Source code for data handling, modeling, training, and visualization
- `tests/`: Unit tests
- `outputs/`: Saved model checkpoints and visual outputs
- `main.py`: Entry script for training and evaluation
- `requirements.txt`: Project dependencies

## 🚀 Features

- PyTorch-based CNN for binary image classification
- Medical image preprocessing
- Evaluation using Accuracy, Precision, Recall, and Confusion Matrix
- Grad-CAM visualization to interpret model attention
- Config-driven design for modular experimentation

## 🛠️ Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/xray-pneumonia-classifier.git
    cd xray-pneumonia-classifier
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset from Kaggle and place the `chest_xray` folder into `data/raw/`.

4. Train the model:
    ```bash
    python main.py --mode train
    ```

5. Evaluate and visualize:
    ```bash
    python main.py --mode eval
    ```

## 📊 Results

- Training Accuracy: TBD
- Validation Accuracy: TBD
- Confusion Matrix: TBD
- Grad-CAM Visuals: See `outputs/visualizations/`

## 📎 References

- [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Papers on pneumonia detection and Grad-CAM

---

Created as a beginner-friendly yet medically relevant deep learning project.
