# Chinese Spam SMS CNN Classification

This repository contains a PyTorch 1D CNN classification model to classify Chinese spam and ham SMS messages.

## Project Structure
- `model.py`: TextCNN model architecture (1D CNN + Max Pooling + Fully Connected).
- `dataset.py`: Text cleaning, tokenization using `jieba`, building vocabulary, and dataset loading (specifically designed for Tab-separated values: `label\ttext`).
- `train.py`: Training script which splits data, calculates **Accuracy, Precision, Recall, Loss**, and saves `training_curves.png` showing the loss/accuracy changes.
- `predict.py`: Inference script to run test predictions.
- `requirements.txt`: Python dependencies.

## Installation
Ensure you have the required packages installed:
```bash
pip install -r requirements.txt
```

## Dataset format
The model expects a dataset at `data/spam_sms.txt` with the following structure:
```text
label\ttext
```
Where:
- `label`: `0` for Normal (Ham), `1` for Spam.
- `text`: Raw Chinese message text.

## Usage
1. Place your dataset under `data/spam_sms.txt`.
2. Run the training script:
   ```bash
   python train.py
   ```
3. After training finishes, the best model parameters will be saved as `best_model.pth`, the vocabulary as `vocab.pkl`, and the training progress plot as `training_curves.png`.
4. Test predictions:
   ```bash
   python predict.py
   ```
