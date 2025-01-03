from constants import *
from matplotlib import pyplot as plt



def plot_roc_auc(df):
    """
    Plots the ROC AUC curves for models based on input DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing TPR, FPR, and AUC values for models.
    
    Returns:
    None
    """
    # Initialize a figure for the ROC plot
    fig1 = plt.figure(figsize=[12, 12])
    ax1 = fig1.add_subplot(111, aspect='equal')

    # Plot the diagonal (random chance baseline)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')

    # Iterate through each column in the DataFrame to plot individual ROC curves
    for col in df.columns:
        # Retrieve TPR, FPR, and AUC values for the model
        mean_tpr = df.loc['tpr', [col]].to_numpy()[0]
        mean_fpr = df.loc['fpr', [col]].to_numpy()[0]
        mean_auc = df.loc['AUC', [col]]

        # Plot the ROC curve with the corresponding AUC label
        plt.plot(mean_fpr, mean_tpr, label=f'{str(col)}: AUC = {round(float(mean_auc), 2)}', lw=2, alpha=1)

    # Add labels, legend, and title for the plot
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")

    # Annotate regions for accuracy
    plt.text(0.32, 0.7, 'More accurate area', fontsize=12)
    plt.text(0.63, 0.4, 'Less accurate area', fontsize=12)

    # Show the plot
    plt.show()



def train_model(train_set, test_set, model, random_state=2021, deep=False, stacking=False):
    """
    Trains a given model on training data and evaluates it on test data.

    Parameters:
    train_set (tuple): Tuple containing training features (X_train) and labels (Y_train).
    test_set (tuple): Tuple containing test features (X_test) and labels (Y_test).
    model: Machine learning or deep learning model to be trained.
    random_state (int): Seed for reproducibility. Default is 2021.
    deep (bool): Flag to indicate if a deep learning model is used. Default is False.
    stacking (bool): Flag to indicate if a stacking model is used. Default is False.

    Returns:
    tuple: Predictions and probability predictions for the test set.
    """
    # Extract train and test sets
    X_train, Y_train = train_set
    X_test, Y_test = test_set

    if deep:
        # Configure learning rate scheduler and compile the model
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-1 / 10**(epoch / 50))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                      loss='binary_crossentropy', 
                      metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        # Train the deep learning model
        model.fit(X_train, Y_train, epochs=40, validation_data=test_set, verbose=False)

    if stacking:
        # Train stacking models
        model.fit(X_train, Y_train)

    if deep or stacking:
        # Generate predictions for deep or stacking models
        probability_predictions = model.predict(X_test)
        predictions = copy.deepcopy(probability_predictions)
        predictions[predictions >= 0.5] = 1.  # Apply threshold
        predictions[predictions < 0.5] = 0.
        predictions = predictions.flatten()
    else:
        # Train traditional ML models and generate predictions
        model.fit(X_train, Y_train)
        probability_predictions = model.predict_proba(X_test)[:, 1]
        predictions = probability_predictions >= 0.46  # Custom threshold
        predictions = predictions.astype(float).flatten()

    return predictions, probability_predictions


def evaluation(true_labels, predicted_labels, probabilities, mean_fpr, average='binary'):
    """
    Computes classification metrics and ROC AUC.

    Parameters:
    true_labels (array): True labels for the test data.
    predicted_labels (array): Predicted labels from the model.
    probabilities (array): Predicted probabilities for positive class.
    mean_fpr (array): Mean false positive rate for ROC curve interpolation.
    average (str): Type of averaging for metrics. Default is 'binary'.

    Returns:
    dict: Dictionary containing classification metrics and ROC AUC values.
    """
    # Compute confusion matrix components
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = precision_score(true_labels, predicted_labels, average=average)
    recall = recall_score(true_labels, predicted_labels, average=average)
    specificity = tn / (tn + fp)
    F1 = f1_score(true_labels, predicted_labels, average=average)
    
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(true_labels, probabilities)
    tprs = interp(mean_fpr, fpr, tpr)
    roc_auc = AUC(fpr, tpr)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'F1': F1,
        'tprs': tprs,
        'roc_auc': roc_auc,
        'mean_fpr': mean_fpr,
    }


def k_fold(X_fold, Y_fold, model, n_splits=10, shuffle=True, random_state=2021, average='binary', deep=False, stacking=False, test_samples=None, augmented_data=None):
    """
    Performs K-Fold cross-validation for a given model.

    Parameters:
    X_fold (array): Features for cross-validation.
    Y_fold (array): Labels for cross-validation.
    model: Machine learning or deep learning model to be evaluated.
    n_splits (int): Number of folds for cross-validation. Default is 10.
    shuffle (bool): Whether to shuffle data before splitting. Default is True.
    random_state (int): Seed for reproducibility. Default is 2021.
    average (str): Type of averaging for evaluation metrics. Default is 'binary'.
    deep (bool): Flag to indicate if the model is deep learning. Default is False.
    stacking (bool): Flag to indicate if stacking is being used. Default is False.
    test_samples (tuple): Optional test samples for evaluation.
    augmented_data (tuple): Optional augmented training data.

    Returns:
    np.array: Aggregated metrics (accuracy, precision, recall, F1, AUC, etc.).
    """
    # Initialize lists to store metrics across folds
    precision, recall, F1, accuracy, specificity, tprs, aucs = ([] for _ in range(7))
    mean_fpr = np.linspace(0, 1, 100)  # Generate evenly spaced FPR values for interpolation

    if test_samples:
        # If test samples are provided, evaluate directly on them
        X_test, Y_test = test_samples
        predictions, probability_predictions = train_model((X_fold, Y_fold), (X_test, Y_test), model, deep=deep, stacking=stacking)
        metrics = evaluation(Y_test, predictions, probability_predictions, mean_fpr, average=average)
        return np.array(
            [
                metrics['accuracy'],
                metrics['recall'],
                metrics['specificity'],
                metrics['precision'],
                metrics['F1'],
                AUC(mean_fpr, metrics['tprs']),
                metrics['tprs'],
                mean_fpr,
            ], dtype=object
        )
    else:
        # Perform K-Fold cross-validation
        cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for train_index, test_index in cv.split(X_fold):
            # Split data into training and validation sets
            X_train, X_test = X_fold[train_index], X_fold[test_index]
            Y_train, Y_test = Y_fold[train_index], Y_fold[test_index]

            if augmented_data:
                # If augmented data is provided, append it to the training set
                X_train = np.append(X_train, augmented_data[0], axis=0)
                Y_train = np.append(Y_train, augmented_data[1])

            # Train the model and evaluate on the validation set
            predictions, probability_predictions = train_model((X_train, Y_train), (X_test, Y_test), model, deep=deep, stacking=stacking)
            metrics = evaluation(Y_test, predictions, probability_predictions, mean_fpr)

            # Collect metrics for each fold
            accuracy.append(metrics['accuracy'])
            precision.append(metrics['precision'])
            recall.append(metrics['recall'])
            F1.append(metrics['F1'])
            specificity.append(metrics['specificity'])
            tprs.append(metrics['tprs'])
            aucs.append(metrics['roc_auc'])

        # Compute mean TPR and AUC across folds
        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = AUC(mean_fpr, mean_tpr)
        return np.array(
            [
                np.array(accuracy).mean(),
                np.array(recall).mean(),
                np.array(specificity).mean(),
                np.array(precision).mean(),
                np.array(F1).mean(),
                mean_auc,
                mean_tpr,
                mean_fpr,
            ], dtype=object
        )



def k_fold_results(x, y, models=models, test_samples=None, n_splits=10, deep_name='Deep', stacking_name='Stacking', augmented_data=None):
    """
    Runs K-Fold cross-validation on multiple models and summarizes results.

    Parameters:
    x (pd.DataFrame): Feature dataset.
    y (pd.Series): Label dataset.
    models (dict): Dictionary of models to evaluate.
    test_samples (tuple): Optional test samples for evaluation.
    n_splits (int): Number of folds for cross-validation. Default is 10.
    deep_name (str): Keyword to identify deep learning models. Default is 'Deep'.
    stacking_name (str): Keyword to identify stacking models. Default is 'Stacking'.
    augmented_data (tuple): Optional augmented training data.

    Returns:
    pd.DataFrame: Summary of evaluation metrics for all models.
    """
    # Initialize a DataFrame to store evaluation metrics for each model
    df = pd.DataFrame(index=['accuracy', 'recall', 'specificity', 'precision', 'F1', 'AUC', 'tpr', 'fpr'])

    # Convert input data to NumPy arrays for processing
    X_fold = x.to_numpy()
    Y_fold = y.to_numpy()

    for item in models.items():
        model_name, model = item
        if deep_name in model_name:
            # Perform K-Fold evaluation for deep learning models
            df[model_name] = k_fold(X_fold, Y_fold, model, test_samples=test_samples, n_splits=n_splits, deep=True, augmented_data=augmented_data)
        elif stacking_name in model_name:
            # Perform K-Fold evaluation for stacking models
            df[model_name] = k_fold(X_fold, Y_fold, model, test_samples=test_samples, n_splits=n_splits, stacking=True, augmented_data=augmented_data)
        else:
            # Perform K-Fold evaluation for other models
            df[model_name] = k_fold(X_fold, Y_fold, model, test_samples=test_samples, n_splits=n_splits, augmented_data=augmented_data)

    # Visualize the ROC AUC curves for all models
    plot_roc_auc(df)

    # Drop 'tpr' and 'fpr' rows as they are not part of the evaluation metrics
    df.drop(index=['tpr', 'fpr'], axis=0, inplace=True)

    return df




def framingham_dataset():
    """
    Downloads the Framingham Heart Study dataset using the Kaggle API.

    Precondition:
    - The Kaggle API key (kaggle.json) must be available in the specified Google Drive path.
    
    Postcondition:
    - The dataset is downloaded to the current working directory as a zip file.
    """
    # Copy Kaggle API key to the appropriate directory
    !cp /content/gdrive/MyDrive/.kaggle/kaggle.json ~/.kaggle/kaggle.json
    # Set correct permissions for the Kaggle API key
    !chmod 600 ~/.kaggle/kaggle.json
    # Download the dataset using the Kaggle API
    !kaggle datasets download -d aasheesh200/framingham-heart-study-dataset


def cleveland_dataset():
    """
    Downloads the Cleveland Heart Disease dataset using the Kaggle API.

    Precondition:
    - The Kaggle API key (kaggle.json) must be available in the specified Google Drive path.
    
    Postcondition:
    - The dataset is downloaded to the current working directory as a zip file.
    """
    # Copy Kaggle API key to the appropriate directory
    !cp /content/gdrive/MyDrive/.kaggle/kaggle.json ~/.kaggle/kaggle.json
    # Set correct permissions for the Kaggle API key
    !chmod 600 ~/.kaggle/kaggle.json
    # Download the dataset using the Kaggle API
    !kaggle datasets download -d ritwikb3/heart-disease-cleveland


def bmc_dataset():
    """
    Downloads the BMC Heart Failure Clinical dataset using the Kaggle API.

    Precondition:
    - The Kaggle API key (kaggle.json) must be available in the specified Google Drive path.
    
    Postcondition:
    - The dataset is downloaded to the current working directory as a zip file.
    """
    # Copy Kaggle API key to the appropriate directory
    !cp /content/gdrive/MyDrive/.kaggle/kaggle.json ~/.kaggle/kaggle.json
    # Set correct permissions for the Kaggle API key
    !chmod 600 ~/.kaggle/kaggle.json
    # Download the dataset using the Kaggle API
    !kaggle datasets download -d andrewmvd/heart-failure-clinical-data


def extract_zipFiles(src, dest):
    """
    Extracts the contents of a zip file to a specified destination folder.

    Parameters:
    src (str): Path to the zip file.
    dest (str): Directory where the contents of the zip file should be extracted.

    Postcondition:
    - The contents of the zip file are extracted to the specified destination.
    """
    try:
        # Open and extract all files from the zip archive
        with zipfile.ZipFile(src, "r") as zip_:
            zip_.extractall(dest)
    except zipfile.BadZipFile:
        # Handle cases where the file is not a valid zip file
        print(f"Error: The file {src} is not a valid zip file.")
    except Exception as e:
        # Catch and report any other exceptions
        print(f"An error occurred while extracting {src}: {str(e)}")



def normalizer(df, method='min-max'):
    """
    Normalizes a DataFrame using the specified method.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be normalized. Assumes the last column is the target.
    method (str): Normalization method ('min-max', 'z-score', or 'log'). Default is 'min-max'.

    Returns:
    pd.DataFrame: Normalized DataFrame with the target column preserved.
    """
    # Select the appropriate normalization method
    if method == 'min-max':
        normalizer = MinMaxScaler()
    elif method == 'z-score':
        normalizer = StandardScaler()
    elif method == 'log':
        normalizer = PowerTransformer(method='box-cox')
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    # Apply normalization to all columns except the target
    array_norm = normalizer.fit_transform(df[df.columns[:-1]])
    df_norm = pd.DataFrame(array_norm, columns=df.columns[:-1])
    # Preserve the target column
    df_norm['target'] = df.target

    return df_norm


def get_describeData(df, dataTypeDict):
    """
    Describes dataset features based on their data type.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing features and a target column.
    dataTypeDict (dict): A dictionary mapping feature names to their data types 
                         ('binary', 'category', or 'continuous').

    Returns:
    tuple: 
      - describeData (dict): A dictionary containing feature summaries:
        - For categorical features: Lists of unique values.
        - For continuous features: Min, max, mean, and standard deviation.
      - features_mask (np.array): Boolean array indicating categorical features (True) or numerical (False).
    """
    describeData = {}
    features_mask = []

    for f in df.columns[:-1]:  # Exclude the target column
        if dataTypeDict[f] == 'binary' or dataTypeDict[f] == 'category':
            # Summarize categorical or binary features with unique values
            describeData[f] = df[f].unique()
            features_mask.append(True)
        elif dataTypeDict[f] == 'continuous':
            # Summarize continuous features with basic statistics
            describeData[f] = {
                'min': df[f].min(),
                'max': df[f].max(),
                'mean': df[f].mean(),  # Fixed typo: .max() -> .mean()
                'std': df[f].std()
            }
            features_mask.append(False)

    features_mask = np.array(features_mask)

    return describeData, features_mask



from sklearn.neighbors import NearestNeighbors
class UnderSampling():
  def __init__(self, df, features_mask, majority_sign, k=2):
    self.df = df
    self.features_mask = features_mask
    self.majority_sign = majority_sign
    self.data = self.df.to_numpy()
    self.features = self.data[:, :-1]
    self.target = self.data[:, -1]
    self.k = k

  def calculate_total_distance(self, point: np.array) -> np.array:
      # return HEEM(point, self.features, self.features_mask)
      return np.linalg.norm(self.features - point, axis=1)

  def find_neighbors(self, i):
    distances = self.calculate_total_distance(self.features[i])
    return np.argsort(distances)[0:self.k]

  def tomek_links(self):
    # Fit a k-NN model on the data
    pairs = []
    for i in range(len(self.features)):
      pair = self.find_neighbors(i)
      pairs.append(pair)

    # Identify pairs with different class labels
    tomek_pairs = [i for i, j in pairs if (self.target[i] == self.majority_sign) and (self.target[i] != self.target[j])]
    tomek_pairs = list(set(tomek_pairs))

    # Remove majority class instances from identified pairs
    return self.df.drop(tomek_pairs, axis=0)

  def edited_nearest_neighbors(self):
    # Fit a k-NN model on the data
    sets = []
    for i in range(len(self.features)):
      set_ = self.find_neighbors(i)
      sets.append(set_)

    enn_removeable = []
    for set_ in sets:
      if len(set(self.target[set_])) > 1:
        for i in set_:
          if self.target[i] == self.majority_sign:
            enn_removeable.append(i)

    # Remove majority class instances from identified pairs
    return self.df.drop(enn_removeable, axis=0)