
# GeneticOversampler

**GeneticOversampler** is a Python-based library designed to address the class imbalance problem in datasets through oversampling. It uses advanced techniques such as genetic algorithms, clustering, hybrid distance metrics, and machine learning models to generate synthetic samples, improving model performance on imbalanced data.

---

## Features

1. **Genetic Algorithm for Oversampling**:
   - Implements evolutionary techniques to generate synthetic minority class samples.
   - Includes mutation, crossover, and fitness evaluation processes.
2. **Clustering-Based Sampling**:
   - Uses the CFSFDP (Clustering by Fast Search and Find of Density Peaks) algorithm to identify regions like inland, borderline, and trapped for guided oversampling.
3. **Hybrid Distance Metrics**:
   - Introduces HEEM (Hybrid Entropy-Enhanced Metric) for improved distance calculations across mixed data types.
4. **KNN and MICE Imputation**:
   - Handles missing data using advanced imputation methods.
5. **Evaluation**:
   - Provides tools for model evaluation using metrics such as ROC-AUC, precision, recall, and F1-score.

---

## Algorithms

### 1. Genetic Algorithm for Oversampling
The core of the project is a genetic algorithm that generates synthetic samples based on the minority class distribution.

#### Steps:
1. **Initialization**:
   - A population of synthetic samples is generated from minority class neighbors.
2. **Fitness Evaluation**:
   - Combines machine learning model confidence with domain-based penalties.
   - Example formula for fitness evaluation:
     \[
     F_{\text{fitness}} = \max \left( \alpha \cdot P_{\text{model}} + \beta \cdot \left[ 1 - \text{sigmoid}(L) \right], 0 \right)
     \]
   - Where:
     - \( P_{\text{model}} \): Prediction confidence of the ML model.
     - \( L \): Domain-based loss.
     - \( \alpha, \beta \): Weighting coefficients.
3. **Crossover and Mutation**:
   - Produces new samples by combining features from parent samples with random mutations for diversity.

#### Fitness Function Examples:
- **Inland/Borderline Loss**:
  \[
  L = 0.5 \cdot \left[ \text{MaxDist}(I, \text{neighbors})^2 + \max(0, \text{Margin} - \text{MaxDist}(I, M))^2 \right]
  \]
- **Trapped Loss**:
  \[
  L = 0.5 \cdot \left( \text{cosine similarity penalty} + \text{distance-based penalty} \right)
  \]

---

### 2. CFSFDP Clustering
Identifies minority class samples as:
- **Inland**: High-density regions with fewer majority class neighbors.
- **Borderline**: Samples close to majority class boundaries.
- **Trapped**: Minority samples surrounded by majority samples.

#### Density Calculation:
\[
\text{Density}(i) = \sum_{j \neq i} \exp \left( -\frac{d_{ij}^2}{d_c^2} \right)
\]
Where:
- \( d_{ij} \): Distance between samples \( i \) and \( j \).
- \( d_c \): Cutoff distance (quantile-based).

#### Importance Score:
\[
\text{Importance} = \sigma_w \cdot \text{Normalized}(\sigma) + (1 - \sigma_w) \cdot \text{Normalized}(\text{Density})
\]

---

### 3. HEEM (Hybrid Entropy-Enhanced Metric)
Handles mixed data types (categorical and continuous) by computing weighted distances.

#### Formula:
\[
\text{Distance}(i, j) = \sqrt{ \sum_{k} \left( \frac{|x_{ik} - x_{jk}|}{4 \cdot \text{std}(k)} \right)^2 + \sum_{l} w_l \cdot \mathbb{1}(x_{il} \neq x_{jl}) }
\]
Where:
- \( w_l \): Entropy-derived weight for categorical feature \( l \).

---

### 4. KNN and MICE Imputation
To handle missing data:
- **KNN Imputation**: Finds nearest neighbors and imputes missing values based on similarity.
- **MICE (Multiple Imputation by Chained Equations)**:
  - Iteratively predicts missing values using Random Forest regressors.

---

### Workflow

1. **Data Preprocessing**:
   - Normalize features.
   - Handle missing values using KNN or MICE.
2. **Clustering**:
   - Apply CFSFDP to categorize minority samples into inland, borderline, and trapped regions.
3. **Synthetic Sample Generation**:
   - Use the genetic algorithm to generate synthetic samples based on clustering results.
4. **Model Training**:
   - Train ML models on the balanced dataset.
5. **Evaluation**:
   - Measure model performance using metrics such as ROC-AUC, F1-score, etc.

---
