
# **Protein-Ligand Binding Prediction**

![](/images/RF_image.png)

## **Project Description**
This project focuses on developing a machine learning pipeline to predict binding affinities between protein-ligand pairs, leveraging provided UniProt and PubChem IDs. The approach includes exploratory data analysis, feature extraction, model training, and evaluation. The goal is to accurately predict binding affinities on a held-out test set using open-source libraries, while ensuring modular, reproducible, and well-documented code.

### **Problem Statement**
The problem aims to:
1. Build a model to predict binding between protein/molecule pairs using UniProt IDs and PubChem IDs with confirmed binding affinity.
2. Expand the dataset with synthetic negative examples to balance training.
3. Extract additional features to enhance model performance.
4. Use auxiliary datasets and cutting-edge ML techniques for optimal results.

---

## **Workflow Overview**

### **Notebook 1: Data Preparation (`ML_protein_binnding_data_prep.ipynb`)**

#### **Objectives**
1. Perform exploratory data analysis (EDA) and clean the dataset.
2. Generate synthetic negative examples by creating non-binding protein-ligand pairs.
3. Extract features for proteins using UniProt UniRep and ligands using PubChem data.
4. Save the processed dataset for downstream modeling.

#### **Key Steps**
- **EDA and Cleaning**: Handle duplicates and inconsistencies in the dataset.
- **Feature Extraction**:
  - **Proteins**: Extract embeddings using UniRep.
  - **Ligands**: Generate molecular fingerprints using RDKit and PubChemPy.
- **Output**:
  - A cleaned and feature-enriched dataset ready for model training.

#### **Dependencies**
- Libraries: `pandas`, `numpy`, `seaborn`, `matplotlib`, `RDKit`, `PubChemPy`, `transformers`.

---

### **Notebook 2: Model Training and Evaluation (`ML_protein_binnding_model_building.ipynb`)**

#### **Objectives**
1. Train multiple models (Logistic Regression, Random Forest, XGBoost, and a Neural Network) on the prepared dataset.
2. Evaluate model performance and identify the best-performing approach.
3. Compare results as a function of dataset size and model complexity.

#### **Key Steps**
- **Model Training**:
  - Train models using scikit-learn, XGBoost, and PyTorch.
- **Evaluation**:
  - Use accuracy, precision, recall, and F1 score metrics.
  - Visualize performance comparisons across models.
- **Output**:
  - Trained models and performance metrics for each approach.

#### **Dependencies**
- Libraries: `pandas`, `numpy`, `seaborn`, `matplotlib`, `transformers`, `torch`, `sklearn`, `xgboost`.

---

## **How to Use**

### **1. Setup**
Ensure the following dependencies are installed. Every Notebook has its own requirements (requirements_data_prep.txt, requirements_model_build.txt):
```bash
pip install -r requirements.txt
```

### **2. Run Data Preparation**
1. Open `ML_protein_binnding_data_prep.ipynb`.
2. Execute all cells sequentially to generate the processed dataset.

### **3. Train Models**
1. Open `ML_protein_binnding_model_building.ipynb`.
2. Load the processed dataset from the previous step.
3. Execute the notebook to train and evaluate models.

### **4. Inference on Test Set**
1. Save your test set in the expected format.
2. Load the trained model pipeline.
3. Run the inference script to predict binding affinities.

---

## **Project Highlights**
1. **Feature Engineering**: Automated feature extraction using state-of-the-art tools.
2. **Reproducibility**: Clear separation of data preparation and modeling pipelines.
3. **Model Comparisons**: Comprehensive performance analysis across multiple algorithms.

---

## **Future Directions**
1. Incorporate additional auxiliary datasets to enhance feature quality.
2. Explore advanced architectures like transformer-based models for protein-ligand embeddings.
3. Optimize the pipeline for large-scale datasets using distributed processing.

---
