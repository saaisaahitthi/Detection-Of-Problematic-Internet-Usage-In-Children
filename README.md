#Problematic Internet Usage Prediction : A Multi-Modal Deep Learning Approach

## 📌 Project Overview

Problematic Internet Use (PIU) among children and adolescents is a growing public health concern, traditionally diagnosed through extensive psychological profiling. This project proposes an accessible, non-invasive alternative: predicting the **Severity Impairment Index (sii)** using easily obtainable physical fitness indicators and continuous wearable actigraphy data.

Unlike standard approaches that flatten time-series data for traditional machine learning models, this project introduces a **Hybrid Deep Learning Architecture**. By processing cross-sectional clinical data alongside dynamic, sequential activity data simultaneously, the model achieves highly accurate and ordinally sound predictions.

---

## 📊 The Dataset & Modalities

The dataset comprises highly heterogeneous data spanning multiple modalities. Due to extreme class imbalance in the target variable (`sii`)—where the 'Severe' class represented only ~1% of the data while the 'None' class represented 58%—the 'Moderate' and 'Severe' classes were merged to formulate a stable **3-class classification problem**.

### 1. Tabular Modality (Static Data)

* **Feature Count:** 40 cross-sectional features.
* **Data Types:** Demographics, engineered physical metrics (e.g., age-based BMI normalization), cardiovascular indicators (resting heart rate), and standard physical fitness test results.

### 2. Time-Series Modality (Dynamic Data)

* **Format:** 81-timestep sequential actigraphy data captured via wearable accelerometers.
* **Purpose:** Captures daily circadian rhythms, physical movement variations, and sedentary behavior patterns over time.

---

## ⚙️ Data Engineering & Preprocessing Pipeline

To ensure robust model generalization and prevent data leakage, a strict preprocessing pipeline was implemented:

1. **Participant-ID Data Splitting:** The dataset was divided into Training (60%), Validation (20%), and Test (20%) sets grouped entirely by unique participant IDs. This strictly prevents temporal data leakage, ensuring no time-series segments from a single participant appear in both the training and test sets.
2. **Missing Value Imputation:** Utilized K-Nearest Neighbors (KNN, `k=5`) imputation to handle missing values in the static clinical data, fitted exclusively on the training set to prevent look-ahead bias.
3. **Feature Scaling:** Standardized all continuous variables to accelerate neural network convergence.

---

## 🧠 Hybrid Model Architecture

The core innovation of this project is a dual-branch neural network implemented in **TensorFlow/Keras**. It processes both modalities independently before fusing their learned representations.

### Branch 1: Tabular Data Processor (ANN)

Designed to extract non-linear representations from the 40 static features.

* **Input:** Flat vector of 40 features.
* **Layers:**
* `BatchNormalization` for input stabilization.
* `Dense` (128 neurons, ReLU) + `Dropout` (0.4).
* `Dense` (64 neurons, ReLU) + `Dropout` (0.4).


* **Output Vector:** 64-dimensional feature representation.

### Branch 2: Time-Series Processor (GRU)

Designed to handle the 81-timestep sequential data. A stacked Gated Recurrent Unit (GRU) is used for its efficiency in capturing long-term dependencies in temporal data.

* **Input:** Sequence of shape `(81, 1)`.
* **Layers:**
* `GRU` (64 units, `return_sequences=True`) + `Dropout` (0.4).
* `GRU` (32 units, `return_sequences=False`) + `Dropout` (0.4).


* **Output Vector:** 32-dimensional sequential representation.

### Fusion & Classification Block

* **Concatenation:** The 64D ANN output and 32D GRU output are merged into a 96D vector.
* **Integration:** `Dense` layer (64 neurons, ReLU) + `Dropout` (0.4).
* **Output:** `Dense` layer with 3 neurons and `Softmax` activation to output a probability distribution across the three severity classes.

---

## 🚀 Training Strategy

The model was compiled and optimized to handle complex multi-modal gradients and the inherent target class imbalance:

* **Optimizer:** `Adam` with a learning rate of `0.0005` and a gradient `clipnorm=1.0` to prevent exploding gradients during RNN backpropagation.
* **Loss Function:** `sparse_categorical_crossentropy`.
* **Class Weighting:** Dynamic class weights were applied during training to heavily penalize misclassifications of the minority classes.
* **Callbacks:**
* `EarlyStopping`: Monitored validation loss with a patience of 5 epochs.
* `ReduceLROnPlateau`: Dynamically reduced the learning rate upon validation stagnation to allow for finer weight adjustments prior to convergence.



---

## 📈 Results & Evaluation

The hybrid model was evaluated on the strictly isolated 200-participant Test Set and compared against baseline ensemble models (LightGBM, XGBoost, Random Forest).

| Metric | Score |
| --- | --- |
| **Test Accuracy** | **89.00%** |
| **Quadratic Weighted Kappa (QWK)** | **0.8172** |

**Key Takeaways:**

* The hybrid model vastly outperformed the strongest baseline (LightGBM: 80.48% Accuracy, 0.7209 QWK).
* The high QWK score (0.8172) indicates strong **ordinal soundness**; the model accurately grasps the severity progression, meaning that when errors do occur, they are almost exclusively confined to adjacent severity classes.

---

## 📁 Repository Structure

```text
├── BTP_final.ipynb          # End-to-end pipeline (Data Loading, Processing, Model, Evaluation)
├── README.md                # Project documentation

```

## 🛠️ Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. The primary dependencies are:

```bash
pip install pandas numpy scikit-learn tensorflow lightgbm xgboost matplotlib seaborn

```

### Execution

1. Clone the repository:
```bash

```



git clone https://github.com/saaisaahitthi/Internet-Addiction-Severity-Prediction.git

```
2. Navigate to the directory and launch Jupyter:
   ```bash
cd Internet-Addiction-Severity-Prediction
jupyter notebook

```

3. **Data Configuration:** Open `BTP_final.ipynb`. *Note: The original dataset is not included in this repository due to privacy and size constraints.* To run the notebook, you will need to provide your own formatted dataset and update the file paths in the data ingestion cells accordingly.
4. Execute the cells sequentially to reproduce the preprocessing, training, and evaluation pipelines.
