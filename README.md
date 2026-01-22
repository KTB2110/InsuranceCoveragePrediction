# Predicting Healthcare Pricing with Insurance Data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

A machine learning project to predict hospital pricing using demographic, geographic, and insurance-related features. Developed as part of coursework at the **University of Michigan School of Information**.

## Authors

- **Saipranav Avula**
- **Krishna Tej Bhat**
- **Louhith Umashankar**

---

## Motivation

In 2021, CMS required hospitals to publish machine-readable files of all items and services, as well as negotiated rates with insurance companies. For citizens and foreigners alike, insurance industry practices are opaque and hospital prices can vary dramatically in different regions of the United States. The recent publication of data provides a wealth of information which can be used as inputs with advanced ML techniques to determine if there are relationships between hospitals, insurance companies, and prices.

**Our goal:** Create a model that can predict the price of a given item on a Michigan hospital's chargemaster using data from insurance companies, item descriptions, and demographic information about the hospital's location.

---

## Key Findings

| Model | Test RMSE (Full Dataset) | Test RMSE (Filtered ≤$35,000) |
|-------|--------------------------|-------------------------------|
| Baseline (Global Mean) | 24,284.30 | 8,078.53 |
| Baseline (Mean Price per Cluster) | 17,126.03 | 5,114.09 |
| XGBoost | 19,437.81 | — |
| Random Forest | 14,027.57 | 4,852.60 |
| **Neural Network** | **6,550.17** | **4,270.19** |

The **Neural Network** achieved the best performance, significantly outperforming baseline predictions. Limiting the model to a narrower price range (≤$35,000) further improved performance across all models.

---

## Repository Structure

```
InsuranceCoveragePrediction/
│
├── Dataset_prep.ipynb                          # Data preprocessing and train/test split
├── final_eda.ipynb                             # Exploratory data analysis with geospatial visualization
├── price_transparency_census_eda.ipynb         # Census data integration and analysis
├── FeatureEngineering.ipynb                    # Feature engineering and clustering analysis
├── price_transparency_embeddings_clusters.ipynb # Medical condition embeddings using BioLord Transformer
├── FilteredDFBaselines.ipynb                   # Baseline model benchmarking
├── RandomForest.ipynb                          # Random Forest and XGBoost models with t-SNE visualization
├── xgboost_grid_search.ipynb                   # XGBoost hyperparameter tuning via grid search
├── price_transparency_neural_net.ipynb         # Neural network model with cross-validation
│
├── README.md
└── LICENSE
```

---

## Methodology

### Data Preprocessing
- Filtered for hospitals only in **Michigan** and retained conditions with prices > $1,000
- Applied **one-hot encoding** to insurer data
- Utilized **BioLord Transformer** to embed medical condition descriptions
- Performed **BIRCH clustering** on condition embeddings for dimensionality reduction
- Conducted **t-SNE** for cluster visualization
- Joined **U.S. Census demographic data** by ZIP code
- Engineered features for primary insurance providers (Medicaid, UHC, Aetna, Cigna)
- Verified if hospitals offered in-house insurance

### Train-Test Split
- **80-20 split** with each hospital exclusively assigned to either Train or Test set (no overlap)
- Ensures model generalization to unseen hospitals

### Models Implemented

1. **Baseline Models**
   - Global mean price prediction
   - Mean price per BIRCH cluster

2. **Random Forest Regressor**
   - Hyperparameters tuned: `n_estimators`, `max_depth`, `max_features`
   - Best configuration: 100 estimators, no max depth, sqrt max features

3. **XGBoost**
   - 3-Fold Cross-Validation with Grid Search
   - Parameters tuned: `learning_rate`, `max_depth`, `gamma`, `subsample`

4. **Neural Network (MLP)**
   - 5-Fold Cross-Validation using GroupKFold
   - Log scaling applied to price to handle outliers
   - Architecture includes batch normalization and dropout layers
   - Retrained best model on all training data for final evaluation

### Feature Importance
Top predictive features from Random Forest:
1. **Condition Cluster** (most important)
2. **Inpatient/Outpatient indicator**
3. **Zipcode**

---

## Installation

### Prerequisites
- Python 3.8+
- Google Colab (recommended) or local Jupyter environment

### Dependencies

```bash
pip install pandas numpy scikit-learn xgboost tensorflow
pip install sentence-transformers transformers
pip install geopandas folium plotly
pip install dask tqdm seaborn matplotlib
```

### Running the Notebooks

1. Clone the repository:
   ```bash
   git clone https://github.com/KTB2110/InsuranceCoveragePrediction.git
   cd InsuranceCoveragePrediction
   ```

2. Download the required datasets (see [Datasets](#datasets) section below)

3. Open notebooks in Google Colab or Jupyter and run cells sequentially

**Recommended execution order:**
1. `Dataset_prep.ipynb` — Preprocess raw data
2. `final_eda.ipynb` — Exploratory analysis
3. `price_transparency_census_eda.ipynb` — Census data integration
4. `FeatureEngineering.ipynb` — Feature engineering
5. `price_transparency_embeddings_clusters.ipynb` — Generate embeddings and clusters
6. `FilteredDFBaselines.ipynb` — Establish baselines
7. `RandomForest.ipynb` — Train tree-based models
8. `xgboost_grid_search.ipynb` — Tune XGBoost
9. `price_transparency_neural_net.ipynb` — Train neural network

---

## Dataset

The project uses the **Healthcare Chargemaster Data** from Kaggle, which contains hospital pricing information and negotiated rates with insurance companies.

**Source:** [Healthcare Chargemaster Data on Kaggle](https://www.kaggle.com/datasets/jpmiller/healthcare)

---

## Results Visualization

### Model Comparison
The Neural Network achieved the lowest RMSE of **6,550.17** on the full dataset and **4,270.19** on the filtered dataset (prices ≤ $35,000).

### Key Insights
- **Condition clustering** provided valuable insights into pricing patterns, highlighting opportunities to analyze healthcare costs further
- The **variability of prices** among insurers and conditions suggests that narrowing the price range improves model performance
- **Geographic and demographic features** (ZIP code, census data) contribute meaningfully to price prediction

---

## Future Work

- Incorporate additional features such as hospital-specific operational metrics and administrative costs
- Explore temporal data to analyze pricing trends over time
- Extend the model to states other than Michigan
- Standardize and accurately cluster itemized descriptions of medical treatments for better price determination across condition categories

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

