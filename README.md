# GeneticOversampler
A Python implementation that combines genetic algorithms and clustering techniques to address class imbalance in datasets through oversampling.

## Overview
`GeneticOversampler` uses a hybrid approach, integrating genetic algorithms (GA) and clustering methods, to generate synthetic samples for minority classes in imbalanced datasets. This innovative combination enhances the diversity and relevance of synthetic samples, improving downstream machine-learning model performance.
Certainly! Let me break down the README with more detailed explanations of each part. I will include additional descriptions, step-by-step explanations of algorithms, and how the components interact with each other in the **GeneticOversampler** project.

---

## Key Features

1. **Genetic Algorithm (GA) for Synthetic Oversampling**:
   - Simulates evolution to create realistic synthetic data points for the minority class.
   - Features crossover, mutation, and fitness evaluation to generate samples.
   
2. **CFSFDP Clustering**:
   - Identifies regions of data (inland, borderline, trapped) to guide the oversampling process.
   - Ensures synthetic samples are placed in meaningful regions of the feature space.

3. **Hybrid Entropy-Enhanced Metric (HEEM)**:
   - A distance metric specifically designed for mixed-type data (numerical + categorical).
   - Improves the clustering and fitness evaluation of samples.

4. **Advanced Data Imputation**:
   - Handles missing data using **KNN Imputation** and **MICE (Multiple Imputation by Chained Equations)**.

5. **Model Evaluation**:
   - Includes tools to compute classification metrics like **ROC-AUC**, **Precision**, **Recall**, **F1-Score**, etc.
   - Visualizes model performance with ROC curves.

---

## Algorithms

### 1. Genetic Algorithm (GA) for Oversampling

The genetic algorithm is central to this project. It is used to generate synthetic data points for the minority class based on evolutionary principles.

#### Step-by-Step Process:

1. **Initialization**:
   - A population of synthetic samples is created using minority class neighbors as "parents."
   - Random noise or small variations are added to diversify the population.

2. **Fitness Evaluation**:
   - Combines machine learning (ML) confidence scores with domain-based penalties.
   - **Fitness Formula**:
     \[
     F_{\text{fitness}} = \max \left( \alpha \cdot P_{\text{model}} + \beta \cdot \left[ 1 - \text{sigmoid}(L) \right], 0 \right)
     \]
     - \( P_{\text{model}} \): Prediction confidence from the ML model.
     - \( L \): Domain-based loss, capturing spatial relationships and distances.
     - \( \alpha, \beta \): Weights for ML confidence and domain loss.

3. **Crossover**:
   - Combines genes (features) from two parent samples to create a new sample.
   - For **categorical features**, selects the value from one parent or mutates to a new category.
   - For **continuous features**, performs arithmetic crossover:
     \[
     G_{\text{new}} = \alpha \cdot G_{\text{parent1}} + (1 - \alpha) \cdot G_{\text{parent2}}
     \]
     Where \( \alpha \) is a randomly chosen weight.

4. **Mutation**:
   - Introduces small random changes to the genes (features) of synthetic samples.
   - Prevents premature convergence and ensures diversity in the generated samples.

5. **Selection**:
   - Retains the most "fit" samples based on the fitness score for the next generation.

6. **Output**:
   - Synthetic samples that balance the class distribution.

#### Fitness Function for Different Regions:
- **Inland/Borderline Loss**:
  \[
  L = 0.5 \cdot \left[ \text{MaxDist}(I, \text{neighbors})^2 + \max(0, \text{Margin} - \text{MaxDist}(I, M))^2 \right]
  \]
  - \( \text{MaxDist}(I, \text{neighbors}) \): Maximum distance to inland neighbors.
  - \( \text{MaxDist}(I, M) \): Maximum distance to majority class neighbors.
  - \( \text{Margin} \): Distance threshold for borderline points.

- **Trapped Loss**:
  \[
  L = 0.5 \cdot \left( \text{cosine similarity penalty} + \text{distance-based penalty} \right)
  \]
  - Incorporates penalties for being too close to majority samples or deviating from minority clusters.

---

### 2. CFSFDP Clustering

CFSFDP (Clustering by Fast Search and Find of Density Peaks) is used to classify minority samples into **inland**, **borderline**, and **trapped** regions.

#### Step-by-Step Process:

1. **Density Calculation**:
   - For each data point \( i \), compute its density using a Gaussian kernel:
     \[
     \text{Density}(i) = \sum_{j \neq i} \exp \left( -\frac{d_{ij}^2}{d_c^2} \right)
     \]
     - \( d_{ij} \): Distance between points \( i \) and \( j \).
     - \( d_c \): Cutoff distance (quantile-based threshold).

2. **Importance Score**:
   - Combines density and distance to higher-density points:
     \[
     \text{Importance} = \sigma_w \cdot \text{Normalized}(\sigma) + (1 - \sigma_w) \cdot \text{Normalized}(\text{Density})
     \]

3. **Cluster Center Selection**:
   - Points with the highest importance scores are selected as cluster centers.

4. **Region Classification**:
   - **Inland**: Points in high-density regions with fewer majority neighbors.
   - **Borderline**: Points near the decision boundary between classes.
   - **Trapped**: Minority points surrounded by majority points.

---

### 3. HEEM (Hybrid Entropy-Enhanced Metric)

HEEM is a custom distance metric designed for datasets with both categorical and continuous features.

#### Formula:
\[
\text{Distance}(i, j) = \sqrt{ \sum_{k} \left( \frac{|x_{ik} - x_{jk}|}{4 \cdot \text{std}(k)} \right)^2 + \sum_{l} w_l \cdot \mathbb{1}(x_{il} \neq x_{jl}) }
\]
Where:
- \( x_{ik} \): Continuous feature \( k \) of sample \( i \).
- \( x_{il} \): Categorical feature \( l \) of sample \( i \).
- \( w_l \): Entropy-derived weight for categorical feature \( l \).

Entropy-based weights ensure categorical features with high variability contribute more to the distance calculation.

---

### 4. KNN and MICE Imputation

Handles missing values in the dataset using:
- **KNN Imputation**:
  - Finds the \( k \)-nearest neighbors of a sample and imputes missing values based on their values.
- **MICE**:
  - Iteratively predicts missing values using models (e.g., Random Forest) trained on other features.

---

## Workflow

1. **Data Preprocessing**:
   - Normalize data.
   - Handle missing values using **KNN** or **MICE**.

2. **Clustering**:
   - Apply **CFSFDP** clustering to classify minority samples into inland, borderline, and trapped regions.

3. **Synthetic Sample Generation**:
   - Use the **genetic algorithm** to generate synthetic samples based on clustering results.
   - Ensure generated samples respect the feature space (e.g., maintain categorical distributions).

4. **Train Models**:
   - Train machine learning models on the oversampled dataset.

5. **Evaluate Models**:
   - Use metrics like **ROC-AUC**, **Precision**, **Recall**, **F1-Score** to assess performance.
