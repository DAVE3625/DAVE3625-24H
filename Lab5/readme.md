# DAVE3625 - Lab5

<p align="center">
  <a href="https://github.com/DAVE3625/DAVE3625-24H/tree/main/Lab4">
    <img src="img/header.png" alt="Classification Algorithms" width="auto" height="auto">
  </a>
</p>

<p align="center">
  Decision Trees | Random Forest | Naive Bayes<br>
  <br />
  ·
  <a href="https://github.com/DAVE3625/DAVE3625-24H/issues">Report Bug</a>
  ·
  <a href="https://github.com/DAVE3625/DAVE3625-24H/issues">Request Feature</a>
</p>

## Table of Contents
- [About The Lab](#about-the-lab)
- [Instructions](#instructions)
- [Imports](#imports)
- [Tasks](#tasks)

  - [Task 1: Load and Explore the Student Performance Dataset](#task-1-load-and-explore-the-student-performance-dataset)
  - [Task 2: Prepare the Target Variable](#task-2-prepare-the-target-variable)
  - [Task 3: Data Preprocessing](#task-3-data-preprocessing)
  - [Task 4: Split the Dataset](#task-4-split-the-dataset)
  - [Task 5: Apply Decision Tree Classifier](#task-5-apply-decision-tree-classifier)
  - [Task 6: Apply Random Forest Classifier](#task-6-apply-random-forest-classifier)
  - [Task 7: Apply Naive Bayes Classifier](#task-7-apply-naive-bayes-classifier)
  - [Task 8: Reflection](#task-8-reflection)
- [Great Job!](#great-job)

## About The Lab

In this lab, the goal is to uderstand and apply **Decision Trees**, **Random Forest**, and **Naive Bayes** classifiers on binary and multiclass classification problems.



### Instructions

In this lab, you will work with the following dataset:

**Student Performance Dataset:** Data for different students with a many variables. We will use it for classification to predict whether a given student gets a pass or fail grade.

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


**Import modules:**
```python
# Import modules
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import modules for machine learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
```


## Tasks

## Task 1: Load and Explore the Student Performance Datset
The Student Performance Dataset contains student achievement data, including various attributes related to their academic and personal backgrounds. The goal is to predict whether a student needs intervention based on their performance.

You can read more about the dataset and the meaning of the columns/variables here: https://www.kaggle.com/datasets/larsen0966/student-performance-data-set

Instructions:

1. Load the Student_Performance dataset into a pandas DataFrame.

2. Display the first few rows of the dataset.

3. Get a summary of the dataset, including data types and descriptive statistics.

4. Visualize the distribution of the target variable (G3).

5. (Bonus) Have a look at the documentation for the dataset to understand the columns of the dataset better. 


<details>
  <summary>Hints</summary>

  **Hint 1**: To load the dataset, you can use the `pd.read_csv()` function from the pandas library. Make sure to provide the correct path to the dataset file.

  **Hint 2**: To display the first few rows of the dataset, you can use the `.head()` method on the DataFrame.

  **Hint 3**: To get a summary of the dataset, you can use the `.info()` method for data types and the `.describe()` method for descriptive statistics.


  **Hint 4**: To visualize the distribution of the target variable, you can use the `sns.countplot()` function from the seaborn library.


  ```python
  # Visualize the distribution of the target variable
  sns.countplot(x='target_column', data=df)
  plt.title('Distribution of Target Variable')
  plt.xlabel('Target')
  plt.ylabel('Count')
  plt.show()
  ```

  **Hint 5**: https://www.kaggle.com/datasets/larsen0966/student-performance-data-set


</details>

****

## Task 2: Prepare the target variable

First we should rename the columns G1, G2 and G3 for conveniance. Then we will create it a binary class for Pass/Fail so that we can use certain machine learning algorithms on the data.

Instructions:

1. Rename the columns G1, G2, G3 in the dataframe:
- G1 ➔ period_1_grades
- G2 ➔ period_2_grades
- G3 ➔ final_grade
2. Create a new binary target variable passed based on the final_grade column:

- If final_grade is greater than or equal to a certain threshold (e.g., 10), then passed is True.
Otherwise, passed is False.

**QUESTION:** What percentage of students passed based on the threshold?

<details>
  <summary>Hints</summary>


  **Hint 1**: To rename the columns, you can use the `rename()` method of the DataFrame.

  ```python
  # Rename columns
  df.rename(columns={'G1': 'period_1_grades', 'G2': 'period_2_grades', 'G3': 'final_grade'}, inplace=True)
  ```

  **Hint 2**: To create a new binary target variable, you can use the `apply()` method with a lambda function.

  ```python
  # Create binary target variable
  df['passed'] = df['final_grade'].apply(lambda x: True if x >= 10 else False)
  ```


</details>

****


## Task 3: Data Preprocessing

Now that we have our target variable passed, we need to preprocess the data before applying machine learning algorithms.

Instructions:

1. Identify the numerical columns in the DataFrame df.
2. Drop the columns which are not numerical, except the "passed" column.
3. Check for missing values in the numerical data and handle them if necessary.

<details>
  <summary>Hints</summary>


  **Hint 1**: To identify the numerical columns, you can create a list of column names that correspond to numerical data.

  ```python
  # List of numerical columns
  numerical_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
            'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
            'absences', 'period_1_grades', 'period_2_grades']
  ```

  **Hint 2**: You dont actually have to drop all the columns, you can create a new dataframe with only the numerical columns and with the "passed" column. 

  ```python
  # Create a new dataset with numerical columns and the 'passed' column
  df_numeric = df[numerical_cols + ['passed']]
  ```



  **Hint 3**: To check for missing values in the numerical data, you can use the `isnull()` method combined with `sum()`.

  ```python
  # Check for missing values
  missing_values = df_numerical.isnull().sum()
  print(missing_values)
  ```



</details>

****

## Task 4: Split the Dataset

We need to split the dataset into training and testing sets.

Instructions:



1. Separate features and target variable

2. Use train_test_split to split the data (e.g., 70% training, 30% testing).

<details>
  <summary>Hint</summary>


  **Hint 1**: To separate features and target variable, you can create two separate DataFrames: one for the features (X) and one for the target variable (y).

  ```python
  # Separate features and target variable
  X = df_numeric.drop(columns=['passed'])
  y = df_numeric['passed']
  ```

  **Hint 2**: To split the data into training and testing sets, you can use the `train_test_split` function from the `sklearn.model_selection` module. Specify the test size (e.g., 0.3 for 30% testing data) and a random state for reproducibility.

  ```python
  # Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  ```
  

</details>



****







## Task 5: Apply Decision Tree Classifier

Now, let's build and evaluate a Decision Tree classifier.

Instructions:

1. Initialize a Decision Tree classifier.
2. Train the model on the training data.
3. Make predictions on the test set.
5. Evaluate the model using accuracy score and confusion matrix.

<details>
  <summary>Hints</summary>

  **Hint 1**: To initialize a Decision Tree classifier, you can use the `DecisionTreeClassifier` class from the `sklearn.tree` module.

  ```python
  # Initialize Decision Tree classifier
  dt = DecisionTreeClassifier()
  ```

  **Hint 2**: To train the model, just use fit()

  ```python
  # Train the model
  dt.fit(X_train, y_train)
  ```

  **Hint 3**: To make predictions on the test set, use the `predict` method of the trained model.

  ```python
  # Make predictions
  y_pred_dt = dt.predict(X_test)
  ```

  **Hint 4**: To evaluate the model using accuracy score and confusion matrix, use the `accuracy_score` and `confusion_matrix` functions from the `sklearn.metrics` module.

  ```python
  # Evaluate the model
  accuracy = accuracy_score(y_test, y_pred)
  cm = confusion_matrix(y_test, y_pred)

  print(f"Accuracy: {accuracy}")
  print("Confusion Matrix:")
  print(cm)
  ```

</details>

**Additional Resources**

For a more in-depth understanding of Decision Trees, check out this excellent video from StatQuest:

[![StatQuest: Decision Trees](https://img.youtube.com/vi/_L39rN6gz7Y/0.jpg)](https://www.youtube.com/watch?v=_L39rN6gz7Y&ab_channel=StatQuestwithJoshStarmer)

Readings: 
https://www.geeksforgeeks.org/decision-tree/


****

## Task 6: Apply Random Forest Classifier

Let's apply a Random Forest classifier and evaluate its performance.

### Instructions:

1. Initialize a Random Forest classifier.
2. Train the model on the training data.
3. Make predictions on the test set.
4. Evaluate the model using accuracy score and confusion matrix.

<details>
  <summary>Hints</summary>

  **Hint 1**: To initialize a Random Forest classifier, you can use the `RandomForestClassifier` class from the `sklearn.ensemble` module.

  ```python
  # Initialize Random Forest classifier
  rf = RandomForestClassifier()
  ```

  **Hint 2**: To train the model, use the `fit` method of the Random Forest classifier.

  ```python
  # Train the model
  rf.fit(X_train, y_train)
  ```

  **Hint 3**: To make predictions on the test set, use the `predict` method of the trained model.

  ```python
  # Make predictions
  y_pred_rf = rf.predict(X_test)
  ```

  **Hint 4**: To evaluate the model using accuracy score and confusion matrix, use the `accuracy_score` and `confusion_matrix` functions from the `sklearn.metrics` module.

  ```python
  # Evaluate the model
  accuracy_rf = accuracy_score(y_test, y_pred_rf)
  cm_rf = confusion_matrix(y_test, y_pred_rf)

  print(f"Accuracy: {accuracy_rf}")
  print("Confusion Matrix:")
  print(cm_rf)
  ```

</details>

**Additional Resources**

For a more in-depth understanding of Random Forests, check out this excellent 2 part video from StatQuest:

[![StatQuest: Random Forests](https://img.youtube.com/vi/J4Wdy0Wc_xQ/0.jpg)](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ&ab_channel=StatQuestwithJoshStarmer)


[![StatQuest: Random Forests Part 2](https://img.youtube.com/vi/sQ870aTKqiM/0.jpg)](https://www.youtube.com/watch?v=sQ870aTKqiM&ab_channel=StatQuestwithJoshStarmer)

Readings:
- [Random Forest Algorithm - GeeksforGeeks](https://www.geeksforgeeks.org/random-forest-algorithm/)

****



## Task 7: Apply Naive Bayes Classifier

Let's apply a Naive Bayes classifier and evaluate its performance.

### Instructions:

1. Initialize a Naive Bayes classifier.
2. Train the model on the training data.
3. Make predictions on the test set.
4. Evaluate the model using accuracy score and confusion matrix.

<details>
  <summary>Hints</summary>

  **Hint 1**: To initialize a Naive Bayes classifier, you can use the `GaussianNB` class from the `sklearn.naive_bayes` module.

  ```python
  # Initialize Naive Bayes classifier
  nb = GaussianNB()
  ```

  **Hint 2**: To train the model, use the `fit` method of the Naive Bayes classifier.

  ```python
  # Train the model
  nb.fit(X_train, y_train)
  ```

  **Hint 3**: To make predictions on the test set, use the `predict` method of the trained model.

  ```python
  # Make predictions
  y_pred_nb = nb.predict(X_test)
  ```

  **Hint 4**: To evaluate the model using accuracy score and confusion matrix, use the `accuracy_score` and `confusion_matrix` functions from the `sklearn.metrics` module.

  ```python
  # Evaluate the model
  accuracy_nb = accuracy_score(y_test, y_pred_nb)
  cm_nb = confusion_matrix(y_test, y_pred_nb)

  print(f"Accuracy: {accuracy_nb}")
  print("Confusion Matrix:")
  print(cm_nb)
  ```

</details>

**Additional Resources**

For a more in-depth understanding of Naive Bayes, check out this excellent video from StatQuest:

[![StatQuest: Naive Bayes](https://img.youtube.com/vi/O2L2Uv9pdDA/0.jpg)](https://www.youtube.com/watch?v=O2L2Uv9pdDA&ab_channel=StatQuestwithJoshStarmer)
[![StatQuest: Gaussian Naive Bayes](https://img.youtube.com/vi/H3EjCKtlVog/0.jpg)](https://www.youtube.com/watch?v=H3EjCKtlVog)

Readings:
- [Naive Bayes Algorithm - GeeksforGeeks](https://www.geeksforgeeks.org/naive-bayes-classifiers/)

****

## Task 8: Reflection

In this section, you will reflect on the tasks you have completed and consider additional challenges to deepen your understanding of the concepts. Answer the following questions and try out the suggested challenges:

### Questions for Reflection

1. **Model Performance Comparison**: Which models performed better in terms of accuracy and confusion matrix? Why do you think that is the case?
2. **Feature Importance**: How important were the features G1, G2, and G3 in predicting the target variable? How would the model performance change if you removed these features from the dataset?
3. **Data Preprocessing**: How did data preprocessing steps like feature scaling and handling missing values impact the performance of the models?
4. **Model Selection**: If you had to choose one model for deployment, which one would it be and why? Consider factors like accuracy, interpretability, and computational efficiency.
5. **Hyperparameter Tuning**: How did hyperparameter tuning (e.g., choosing the optimal value of K for KNN) affect the performance of the models? What other hyperparameters could you tune to potentially improve the models?
6. **Overfitting and Underfitting**: Did you observe any signs of overfitting or underfitting in your models? How did you address these issues? or how could you address these issues in the future?

### Additional Challenges

1. **Feature Engineering**: Try creating new features from the existing ones. For example, you could create an average grade feature from G1, G2, and G3. How does this new feature impact model performance?
2. **Cross-Validation**: Implement cross-validation to get a more robust estimate of model performance. How do the results compare to the train-test split method?
3. **Ensemble Methods**: Experiment with other ensemble methods like Gradient Boosting or AdaBoost. How do these methods compare to the Random Forest classifier?
4. **Dimensionality Reduction**: Apply dimensionality reduction techniques like PCA (Principal Component Analysis) to the dataset. How does this affect model performance?
5. **Different Datasets**: Try applying the same models and preprocessing steps to a different dataset. How do the results compare?

Reflecting on these questions and trying out the additional challenges will help you gain a deeper understanding of the machine learning concepts and improve your problem-solving skills.


## Great Job!

Congratulations on completing the lab! 🎉 


👏 **Well done!** 👏


