[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

# DAVE3625 - Lab4

<p align="center">
  <a href="https://github.com/DAVE3625/DAVE3625-24H/tree/main/Lab4">
    <img src="img/header.png" alt="Classification Algorithms" width="auto" height="auto">
  </a>
</p>

<p align="center">
  KNN, SVM, and Random Forest Classification on Wine Quality Dataset<br>
  <br />
  ·
  <a href="https://github.com/DAVE3625/DAVE3625-24H/issues">Report Bug</a>
  ·
  <a href="https://github.com/DAVE3625/DAVE3625-24H/issues">Request Feature</a>
</p>

## Table of Contents

1. [About The Lab](#about-the-lab)
2. [Imports](#imports)
3. [Tasks](#tasks)
    - [Task 1: Load and Explore the Dataset](#task-1-load-and-explore-the-dataset)
    - [Task 2: Preprocess the Data](#task-2-preprocess-the-data)
    - [Task 3: Feature Scaling - Data Standardization](#task-3-feature-scaling---data-standardization)
    - [Task 4: Split the Dataset](#task-4-split-the-dataset)
    - [Understanding Accuracy and Confusion Matrix](#understanding-accuracy-and-confusion-matrix)
    - [Task 5: Apply K-Nearest Neighbors (KNN) Classifier](#task-5-apply-k-nearest-neighbors-knn-classifier)
    - [Task 6: Apply Support Vector Machine (SVM) Classifier](#task-6-apply-support-vector-machine-svm-classifier)
    - [Task 7: Compare the Performance of Different Classifiers and General Reflection](#task-7-compare-the-performance-of-different-classifiers-and-general-reflection)


</details>

## About The Lab

In this lab, you will gain hands-on experience with several key concepts and techniques in machine learning:

- **K-Nearest Neighbors (KNN)**: Implement and understand the KNN algorithm for classification tasks.
- **Support Vector Machines (SVM)**: Explore SVMs with different kernels and understand their impact on classification.
- **Data standardization**: Understand the importance of feature scaling and standardization for certain algorithms.
- **Evaluation Metrics**: Learn about **[Accuracy][whatis-accuracy]** and **[Confusion Matrix][whatis-confusion-matrix]** to evaluate classification models.
### Tools and Libraries



### Instructions

In this lab, you will work with the Wine Quality dataset to predict wine quality using various classification algorithms.
- Follow the instructions from the tasks.
- Watch the videos
- Feel free to look at the hints
- Contact the TA's if you have any doubts or questions


**Check the Solution**: Try your best before looking at the solutioons.

Good luck, and enjoy the lab!

## Imports

We will be using the following packages in this lab:

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `matplotlib`: For data visualization.
- `seaborn`: For statistical data visualization.
- `scikit-learn`: For machine learning algorithms and evaluation metrics.

If you don't have them, you can install them in Anaconda Prompt (terminal for Mac).

**Remember to activate your environment first**

With pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

With conda:
```bash
conda install pandas numpy matplotlib seaborn scikit-learn
```



```python
# Import modules
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import modules for machine learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
```


## Tasks
## Task 1: Load and Explore the Datset
The Wine Quality dataset (UCI Machine Learning Repository) contains information about red and white variants of the Portuguese "Vinho Verde" wine.

Instructions:

1. Load the dataset into a pandas DataFrame. You can use either the red wine or white wine dataset, or combine them if you wish.

2. Display the first few rows of the dataset.

3. Get a summary of the dataset, including data types and descriptive statistics.


<details>
  <summary>Hints</summary>

  **Hint 1**: To load the dataset, you can use the `pd.read_csv()` function from the pandas library. Make sure to provide the correct path to the dataset file.

  **Hint 2**: To display the first few rows of the dataset, you can use the `.head()` method on the DataFrame.

  **Hint 3**: To get a summary of the dataset, you can use the `.info()` method for data types and the `.describe()` method for descriptive statistics.

</details>

****

## Task 2: Preprocess the Data

Before we can apply machine learning algorithms, we need to preprocess the data.

Instructions:

1. Explore the 'quality' column and consider converting it into a binary classification problem (e.g., wine is 'good' if quality >= 7, otherwise 'not good').
2. Plot the distribution of the new target variable.

<details>
  <summary>Hints</summary>

  
  **Hint 1**: To convert the 'quality' column into a binary classification problem, you can use the `.apply()` method with a lambda function.

  ```Python
  # Convert 'quality' into binary classes
  df['quality_binary'] = np.where(df['quality'] >= 7, 1, 0)
  ```


  **Hint 2**: To plot the distribution of the new target variable, you can use the `sns.countplot()` function from the seaborn library.
  ```Python
  # Plot distribution
  sns.countplot(x='quality_binary', data=df)
  plt.title('Distribution of Wine Quality')
  plt.xlabel('Quality (0 = Not Good, 1 = Good)')
  plt.ylabel('Count')
  plt.show()
  ```

</details>

****


## Task 3: Feature Scaling - Data Standardization

Feature scaling ensures that each feature contributes equally to the result, preventing features with larger scales from dominating the learning process. Standardization transforms features to have a mean of 0 and a standard deviation of 1, which is crucial for algorithms like KNN and SVM.


(For a deeper understanding of feature scaling and its importance in machine learning, you can refer to this [GeeksforGeeks article on feature scaling: normalization vs standardization](https://www.geeksforgeeks.org/normalization-vs-standardization/).)


Instructions:

1. Separate the features and the target variable.
2. Use `StandardScaler` to standardize the feature data.
 (We imported the class from the `sklearn.preprocessing` module)

  ```python
  # Normalize the feature data
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  ```
3. Compare the data before and after scaling.

<details>
  <summary>Hints</summary>

  **Hint 1**: To separate the features and the target variable, you can use the `.drop()` method to drop the target column from the DataFrame and assign it to `X`, and assign the target column to `y`.

  ```python
  # Separate features and target
  X = df.drop(['quality', 'quality_binary'], axis=1)
  y = df['quality_binary']
  ```

  **Hint 2**: To compare the data before and after scaling, you can create a DataFrame from the scaled data and use the `.describe()` method to show descriptive statistics.

  ```python
  # Compare data before and after scaling
  X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
  print("Before Scaling:\n", X['pH'].head(5))
  print("After Scaling:\n", X_scaled_df['pH'].head(5))
  ```

</details>

****

## Task 4: Split the Dataset

We need to split the dataset into training and testing sets to evaluate our models.

Instructions:

1. Use train_test_split to split the data into training and testing sets (e.g., 70% training, 30% testing).

2. Set a random state for reproducibility.

<details>
  <summary>Hint</summary>

  ```python
  # Split the dataset
  X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

  ```
</details>

****





***Read this before doing task 5***

## Understanding Accuracy and Confusion Matrix:

- **Accuracy**: The ratio of correctly predicted observations to the total observations. It answers the question: "How often is the classifier correct?"

  \[
  \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
  \]

- **Confusion Matrix**: A table used to describe the performance of a classification model. It shows the counts of True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).

  |                | Predicted Positive | Predicted Negative |
  |----------------|--------------------|--------------------|
  | **Actual Positive** | True Positive (TP) | False Negative (FN) |
  | **Actual Negative** | False Positive (FP) | True Negative (TN) |

### Interpretation:

- **True Positives (TP)**: Correctly predicted positive observations.
- **True Negatives (TN)**: Correctly predicted negative observations.
- **False Positives (FP)**: Incorrectly predicted positive observations (Type I error).
- **False Negatives (FN)**: Incorrectly predicted negative observations (Type II error).

****
## Task 5: Apply K-Nearest Neighbors (KNN) Classifier
Now, let's build and evaluate a KNN classifier.

Instructions:

1. Initialize a KNN classifier.
2. Use cross-validation to find the optimal value of K (number of neighbors).
3. Train the model with the optimal K.
4. Make predictions on the test set.
5. Evaluate the model using accuracy score and confusion matrix.


<details>
  <summary>Hints</summary>

  **Hint 1**: To initialize a KNN classifier, you can use the `KNeighborsClassifier` class from the `sklearn.neighbors` module.

  ```python
  # Initialize KNN classifier
  knn = KNeighborsClassifier()
  ```

  **Hint 2**: To use cross-validation to find the optimal value of K, you can use the `GridSearchCV` class from the `sklearn.model_selection` module. Define a parameter grid with different values of K and fit the grid search to the training data.

  ```python
  # Define parameter grid
  param_grid = {'n_neighbors': np.arange(1, 31)}

  # Use GridSearchCV to find the optimal K
  knn_gscv = GridSearchCV(knn, param_grid, cv=5)
  knn_gscv.fit(X_train, y_train)

  # Get the optimal K
  optimal_k = knn_gscv.best_params_['n_neighbors']
  print(f"Optimal K: {optimal_k}")
  ```

  **Hint 3**: To train the model with the optimal K, initialize the KNN classifier with the optimal K and fit it to the training data.

  ```python
  # Train the model with the optimal K
  knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)
  knn_optimal.fit(X_train, y_train)
  ```

  **Hint 4**: To make predictions on the test set, use the `predict` method of the trained model.

  ```python
  # Make predictions on the test set
  y_pred = knn_optimal.predict(X_test)
  ```

  **Hint 5**: To evaluate the model using accuracy score and confusion matrix, use the `accuracy_score` and `confusion_matrix` functions from the `sklearn.metrics` module.

  ```python
  # Evaluate the model
  accuracy = accuracy_score(y_test, y_pred)
  cm = confusion_matrix(y_test, y_pred)

  print(f"Accuracy: {accuracy}")
  print("Confusion Matrix:")
  print(cm)
  ```

</details>

<br>

 **Additional Resources**

For a more in-depth understanding of K-Nearest Neighbors (KNN), check out this excellent video from StatQuest:

[![StatQuest: K-Nearest Neighbors (KNN)](https://img.youtube.com/vi/HVXime0nQeI/0.jpg)](https://www.youtube.com/watch?v=HVXime0nQeI&t=249s&ab_channel=StatQuestwithJoshStarmer)


****


## Task 6: Apply Support Vector Machine (SVM) Classifier

Let's apply SVM with different kernels and evaluate the performance.

Instructions:

1. Initialize SVM classifiers with different kernels ('linear', 'rbf').
2. Train the models on the training data.
3. Make predictions on the test set.
4. Evaluate the models using accuracy score and confusion matrix.

<details>
  <summary>Hints</summary>

  **Hint 1**: To initialize SVM classifiers with different kernels, you can use the `SVC` class from the `sklearn.svm` module.

  ```python
  # Initialize SVM classifiers with different kernels
  svm_linear = SVC(kernel='linear')
  svm_rbf = SVC(kernel='rbf')
  ```

  **Hint 2**: To train the models on the training data, use the `fit` method of the SVM classifiers.

  ```python
  # Train the models
  svm_linear.fit(X_train, y_train)
  svm_rbf.fit(X_train, y_train)
  ```

  **Hint 3**: To make predictions on the test set, use the `predict` method of the trained models.

  ```python
  # Make predictions on the test set
  y_pred_linear = svm_linear.predict(X_test)
  y_pred_rbf = svm_rbf.predict(X_test)
  ```

  **Hint 4**: To evaluate the models using accuracy score and confusion matrix, use the `accuracy_score` and `confusion_matrix` functions from the `sklearn.metrics` module.

  ```python
  # Evaluate the models
  accuracy_linear = accuracy_score(y_test, y_pred_linear)
  cm_linear = confusion_matrix(y_test, y_pred_linear)

  accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
  cm_rbf = confusion_matrix(y_test, y_pred_rbf)

  print(f"Linear Kernel - Accuracy: {accuracy_linear}")
  print("Linear Kernel - Confusion Matrix:")
  print(cm_linear)

  print(f"RBF Kernel - Accuracy: {accuracy_rbf}")
  print("RBF Kernel - Confusion Matrix:")
  print(cm_rbf)
  ```

</details>

<br>

**Additional Resources**

For a more in-depth understanding of Support Vector Machines (SVM), check out this excellent video from StatQuest:
[![Visually Explained: Support Vector Machines (SVM)](https://img.youtube.com/vi/_YPScrckx28/0.jpg)](https://www.youtube.com/watch?v=_YPScrckx28&ab_channel=VisuallyExplained)

For a more in-depth understanding of Support Vector Machines (SVM), check out these additional resources:

- [(3 Part video series on SVM) StatQuest: Support Vector Machines (SVM)](https://www.youtube.com/watch?v=efR1C6CvhmE&ab_channel=StatQuestwithJoshStarmer)
- [(Article) GeeksforGeeks: Support Vector Machine Algorithm](https://www.geeksforgeeks.org/support-vector-machine-algorithm/)

****



## Task 7: Compare the Performance of Different Classifiers and General Reflection

Let's compare the performance of the KNN and SVM classifiers that we have trained.

### Instructions:

1. **Create a summary table of the accuracy scores of each classifier.**

```python
# Summary table of accuracy scores
summary = {
  'Classifier': ['KNN', 'SVM (Linear Kernel)', 'SVM (RBF Kernel)'],
  'Accuracy': [accuracy, accuracy_linear, accuracy_rbf]
}
summary_df = pd.DataFrame(summary)
print(summary_df)
```
<details>

**<summary>2. Plot ROC curves for each classifier (optional).</summary>**

```python
from sklearn.metrics import roc_curve, auc

# Compute ROC curve and ROC area for each classifier
fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_optimal.predict_proba(X_test)[:,1])
roc_auc_knn = auc(fpr_knn, tpr_knn)

fpr_svm_linear, tpr_svm_linear, _ = roc_curve(y_test, svm_linear.decision_function(X_test))
roc_auc_svm_linear = auc(fpr_svm_linear, tpr_svm_linear)

fpr_svm_rbf, tpr_svm_rbf, _ = roc_curve(y_test, svm_rbf.decision_function(X_test))
roc_auc_svm_rbf = auc(fpr_svm_rbf, tpr_svm_rbf)

# Plot ROC curves
plt.figure()
plt.plot(fpr_knn, tpr_knn, color='blue', lw=2, label='KNN (area = %0.2f)' % roc_auc_knn)
plt.plot(fpr_svm_linear, tpr_svm_linear, color='green', lw=2, label='SVM Linear (area = %0.2f)' % roc_auc_svm_linear)
plt.plot(fpr_svm_rbf, tpr_svm_rbf, color='red', lw=2, label='SVM RBF (area = %0.2f)' % roc_auc_svm_rbf)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.show()
```
</details>


****

### Reflection

**1. Which classifier performed the best based on accuracy?**

**2. Considering both accuracy and confusion matrices, which model would you choose and why?**

**3. Could you use the models trained on red wine to predict the quality if the different white wines?**

**4. Reflect on how you might apply the techniques learned in this lab to other datasets and machine learning problems.**





## Great Job!

Congratulations on completing the lab! You've done an excellent job working through the tasks and applying various machine learning algorithms to the Wine Quality dataset. Here's a quick recap of what you've accomplished:

- **Loaded and explored the dataset**: You successfully loaded the dataset and performed initial exploration to understand its structure and contents.
- **Preprocessed the data**: You converted the 'quality' column into a binary classification problem and visualized the distribution of the new target variable.
- **Standardized the features**: You applied feature scaling to ensure that each feature contributed equally to the learning process.
- **Split the dataset**: You split the data into training and testing sets to evaluate the performance of your models.
- **Applied K-Nearest Neighbors (KNN)**: You found the optimal value of K using cross-validation, trained the KNN classifier, and evaluated its performance.
- **Applied Support Vector Machine (SVM)**: You trained SVM classifiers with different kernels and evaluated their performance.
- **Compared classifiers**: You compared the performance of KNN and SVM classifiers and reflected on their results.


👏 **Well done!** 👏

## Usefull links

[pandas cheatsheet][pandas-cheatsheet]

[matplotlib cheatsheet][matplotlib-cheatsheet]

[seaborn cheatsheet][seaborn-cheatsheet]

[sklearn cheatsheet][sklearn-cheatsheet]

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
<!-- shields -->
[issues-shield]: https://img.shields.io/github/issues/umaimehm/Intro_to_AI_2021.svg?style=for-the-badge
[issues-url]: https://github.com/DAVE3625/DAVE3625-24H/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/DAVE3625/DAVE3625-24H/blob/main/Lab1/LICENSE

<!-- images -->


<!-- documentation -->
[pandas-doc]: https://pandas.pydata.org/docs/
[numpy-doc]: https://numpy.org/doc/stable/
[seaborn-doc]: https://seaborn.pydata.org/
[sklearn-doc]: https://scikit-learn.org/0.21/documentation.html

<!-- tutorials -->
[pandas-cheatsheet]: https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf
[matplotlib-cheatsheet]: https://matplotlib.org/cheatsheets/
[sklearn-cheatsheet]: https://www.datacamp.com/cheat-sheet/scikit-learn-cheat-sheet-python-machine-learning
[seaborn-cheatsheet]: https://www.kaggle.com/code/themlphdstudent/cheat-sheet-seaborn-charts


<!-- links -->
[regex]: https://www.geeksforgeeks.org/python-regex-cheat-sheet/
[solution]: solution.ipynb
[whatis-accuracy]: https://developers.google.com/machine-learning/crash-course/classification/accuracy
[whatis-confusion-matrix]: https://developers.google.com/machine-learning/crash-course/classification/true-false-positive-negative