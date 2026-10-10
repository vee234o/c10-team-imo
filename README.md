# Latent Probe - Mental Health Sentiment Classification

## Problem Statement

Traditional keyword filters fail to detect chronic stress and burnout in digital communications because subtle distress is masked by indirect language rather than explicit cries for help. Without early detection, these quiet struggles compound into severe mental health crises.

## Solution

Latent Probe is a resource-efficient machine learning pipeline that identifies nuanced markers of stress in text. By extracting deep semantic features from a frozen large language model and classifying them via a high-margin support vector machine, the system provides accurate screening while minimizing false alarms and preserving data privacy.

## Technologies Used

* **Google Gemma-2-2b**: Frozen pre-trained model utilized for extracting rich latent representations from Layer 23 hidden states.
* **Support Vector Classifier (SVC)**: High-margin linear classifier trained on scaled embeddings to distinguish between stressed and non-stressed text.
* **Joblib**: Serialization library used for persisting and loading model and scaler artifacts.
* **PyTorch & Transformers**: Frameworks handling tensor manipulation, masked mean pooling, and tokenization.
* **Scikit-Learn**: Library providing feature scaling (`StandardScaler`) and evaluation metrics.

## Repository File Structure

* **`pipeline.ipynb`**: The complete end-to-end Python/Colab notebook containing text preprocessing, mini-batch feature extraction, model evaluation, and inference logic.
* **`stress_svc_probe.joblib`**: The serialized Support Vector Classifier model file storing the trained decision boundary weights.
* **`feature_scaler.joblib`**: The serialized Scikit-Learn `StandardScaler` artifact ensuring consistent normalization of embedding features during testing and inference.