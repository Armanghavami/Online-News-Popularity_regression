# Online News Popularity Prediction — Deep Feedforward Neural Network

This project predicts the **popularity of online news articles** (measured by number of shares) using a **configurable deep feedforward neural network** implemented in **PyTorch**.
It includes data preprocessing, training, validation, testing, experiment tracking with **Weights & Biases (wandb)**, hyperparameter optimization via **Sweeps**, and **SHAP explainability** for model interpretation.

---

## Project Overview

The goal of this project is to predict article popularity based on content and metadata features.
The pipeline is fully modular, allowing configurable architectures, data preprocessing methods, and training hyperparameters.

Key features:

* Configurable MLP architecture (`flexible_model`)
* Early stopping and learning rate scheduling for robust training
* SHAP explainability for feature importance analysis
* Weights & Biases integration for tracking experiments
* Optional hyperparameter optimization using **W&B Sweeps**

---

##  Project Structure

```
Online News Popularity_regression/
│
├── config.py                  # Hyperparameters, training settings, data paths
├── data.py                    # Data loading, normalization, and splitting
├── model.py                   # Fixed and flexible neural network definitions
├── train.py                   # Training loop with validation and early stopping and saving the model
├── evaluate.py                # Test evaluation
├── main.py                    # Main script to run full pipeline
├── saved_w.pth                # saved model weights
└── sweep_main.py              # W&B Sweep main and configuration (hyperparameter search)
```

---

## How It Works

1. **Configuration**
   Loads training parameters and model architecture from `config.py`.

2. **Data Preprocessing**

   * Loads the dataset using `data_load()`
   * Normalizes features using `StandardScaler` or `MinMaxScaler`
   * Splits the data into train, validation, and test sets

3. **Model Definition**

   * `flexible_model()` builds a feedforward neural network with configurable layers

4. **Training**

   * Uses **Adam optimizer** and **MSE loss**
   * Employs **ReduceLROnPlateau scheduler** to adapt learning rate
   * Implements **early stopping** to prevent overfitting
   * Saves the final model

5. **Evaluation**

   * Computes **MSE, MAE, R², RMSE** on train, validation, and test sets
   * Test evaluation done via `evaluate.test()` function

6. **Explainability (SHAP)**

   * Uses `shap.DeepExplainer` to interpret feature importance
   * Generates waterfall plots to visualize the effect of each feature

7. **Experiment Tracking**

   * Uses **Weights & Biases (wandb)** to log hyperparameters, metrics, and model weights
   * Optionally disables wandb logging via `WANDB_MODE=disabled`

8. **Hyperparameter Optimization (Sweep)**

   * W&B Sweep automates hyperparameter tuning: learning rate, batch size, weight decay, layer configuration, and scheduler patience
   * Sweep method: Bayesian optimization
   * Objective: minimize validation MSE

---

## Example Configuration (`config.py`)

```python
config = {
    "data_path": "data/OnlineNewsPopularity.csv",
    "batch_size" :128,
    "data_normalization_method" :"z_score",
    "train_size" :0.7,
    "epochs":30,
    "learning_rate":0.05,
    "samples_count":3950,

    "weight_decay":0.0002930081658981912,
    "lr_factor":0.5,
    "lr_patioence":3,

    "early_stopping_delta":0,
    "early_stopping_patience":20,

    "l1_lambda":0.0006115926667062532,
    "saved_model":True,
    "continue_training":True,


    "layer_config":[
  {"size":128, "batch_norm":True, "dropout":0.6, "activation_function":"leaky_relu"},
  {"size":64, "batch_norm":True, "dropout":0.6, "activation_function":"leaky_relu"},
  {"size":64, "batch_norm":True, "dropout":0, "activation_function":"leaky_relu"},
  {"size":64, "batch_norm":True, "dropout":0, "activation_function":"leaky_relu"},
  {"size":1, "batch_norm":False, "dropout":0, "activation_function":"None"}
]

}
```

---

## Results without hiperpramter search (wandb sweep)

| Metric   | Train  | Validation | Test   |
| -------- | ------ | ---------- | ------ |
| **MSE**  | 0.736  | 0.779      | 1.047  |
| **MAE**  | 0.637  | 0.661      | 0.763  |
| **R²**   | 0.150  | 0.154      | -0.65  |
| **RMSE** | 0.858  | 0.883      | 1.023  |
### Note on Data and Performance

The dataset chosen contains **unstable features and a high number of outliers**. This was intentional to simulate **real-world data challenges** and to test model robustness.  
- The negative test R² (-0.65) reflects that some predictions were significantly far from the true values due to extreme outliers.  
- Despite this, the training and validation metrics (R² ≈ 0.15) show the model learned some patterns in the majority of the data.  

This setup demonstrates how the model performs under **noisy, real-world conditions**, which is important for practical applications.

**Feature Importance (SHAP)**:

* Top features: `tokens_title`, `kw_max_max`, `kw_min_min`
* Waterfall plots show each feature's contribution to the model's predictions

---

## Hyperparameter Optimization (W&B Sweep)

* Method: Bayesian optimization
* Objective: minimize validation loss
* Tuned parameters:

  * Learning rate: 0.0001 → 0.01
  * Batch size: 32, 64, 128
  * Weight decay: 0, 1e-5, 1e-4
  * Layer configuration: [128,64,32,1], [256,128,64,1], etc.
  * Scheduler patience: 5, 10


## Technologies Used

* Python 3.12
* PyTorch
* pandas / NumPy / scikit-learn
* Weights & Biases (wandb)
* SHAP (explainability)
* Matplotlib / Seaborn (visualizations)

