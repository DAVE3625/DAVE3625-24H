<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]




<!-- PROJECT LOGO -->
<br />
<h3 align="center">Dave3625 - Lab3</h3>
<p align="center">
  <a href="https://github.com/DAVE3625/DAVE3625-24H/tree/main/Lab3">
    <img src="img/header.png" alt="Data wrangling" width="auto" height="auto">
  </a>

  

  <p align="center">
    Logestic Regression<br>
    <br />
    ·
    <a href="https://github.com/DAVE3625/DAVE3625-24H/issues">Report Bug</a>
    ·
    <a href="https://github.com/DAVE3625/DAVE3625-24H/issues">Request Feature</a>
  </p>
</p>


<!-- ABOUT THE LAB -->

<details>
  <summary>Table of Contents</summary>

1. [About The Lab](#about-the-lab)
   - [Tools and Libraries](#tools-and-libraries)
   - [Instructions](#instructions)
2. [New Imports](#new-imports)
3. [Tasks](#tasks)
4. [Conclusion](#conclusion)
5. [Useful Links](#useful-links)
6. [License](#license)

</details>

## About The Lab

In this lab, you will gain hands-on experience with several key concepts and techniques in machine learning:

- **[Logistic Regression][whatis-logisticregression]**: Understand and implement logistic regression models for binary classification tasks.
- **[Correlation][whatis-correlation]**: Learn how to compute and interpret correlation matrices to understand relationships between features.
- **[Feature Selection][whatis-featureselection]**: Discover methods to select the most relevant features for your supervised learning models.
- **Model Evaluation**: Evaluate your models using performance metrics such as [ROC][whatis-roc] (Receiver Operating Characteristic) curves and [AUC][whatis-auc] (Area Under the Curve).

### Tools and Libraries

We will be using the following Python libraries:

- **[pandas][pandas-doc]**: For data manipulation and analysis.
- **[numpy][numpy-doc]**: For numerical computations.
- **[seaborn][seaborn-doc]**: For data visualization.
- **[sklearn][sklearn-doc]**: For machine learning algorithms and tools.

### Instructions

In this lab, you will work with the Titanic dataset to predict passenger survival using logistic regression. You will:

- Preprocess and split the data into training and testing sets.
- Engineer new features.
- Perform exploratory data analysis.
- Select important features using Recursive Feature Elimination (RFE).
- Build and evaluate a logistic regression model.
- Understand evaluation metrics such as confusion matrix, accuracy, log loss, and AUC.
- Perform cross-validation.

1. **Check the Solution**: Try your best before looking at the **[solution][solution]**.

Good luck, and enjoy the lab!

## New imports

We will use a new package in this lab. Install it in anaconda prompt (terminal for mac) with:

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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix
```


Load the titanic dataset that we cleaned in the last lab, it's found under /data/Titanic_Cleaned.csv.


## Tasks
**Task 1: Split the Dataset**
We need to split the dataset into two parts:
- Training Set (df_train): Contains passengers with known 'Survived' values.
- Test Set (df_test): Contains passengers with missing 'Survived' values.

**Instructions:**

- Create the test set by selecting rows where 'Survived' is NaN.
- Create the training set by dropping rows where 'Survived' is NaN.
- Save both datasets as CSV files for future use.

****

**Task 2: Create a New Feature 'TravelAlone'**

According to the kaggle data dictionary the columns **SibSp** and **Parch** are related to traveling with family. Create a new categorical variable: wether or not the individual was travling alone.

**Instructions:**

1. Create the 'TravelAlone' feature:
   - Set to 1 if the passenger was traveling alone.
   - Set to 0 if the passenger was traveling with family.
2. Drop the 'SibSp' and 'Parch' columns, as they are no longer needed.

Code: 

```python
# using np.where()
df_train.loc[:, 'TravelAlone'] = np.where((df_train["SibSp"] + df_train["Parch"]) > 0, 0, 1) 

# or using boolean check
df['TravelAlone'] = (df['SibSp'] + df['Parch'] == 0).astype(int)
```

****

**Task 3: Explore the Data**
Let's explore the data to understand how different features affect survival.

**Instructions:**
1. Visualize the impact of age categories on survival using a density plot.
2. Analyze other features such as 'Fare', 'Pclass', 'Sex', and 'TravelAlone'.
3. Interpret the plots and write down your observations.


**Example for 'CatAge' column:**
```python
plt.figure(figsize=(15,8))
ax = sns.kdeplot(df_train["CatAge"][df_train.Survived == 1], color="darkturquoise", fill=True)
sns.kdeplot(df_train["CatAge"][df_train.Survived == 0], color="lightcoral", fill=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Age Category for Surviving Population and Deceased Population')
ax.set(xlabel='Age')
plt.xlim(-1,4)
plt.show()
```

Repeat similar plots for other features like 'CatFare', 'Pclass', 'TravelAlone', etc...

**Questions:**

- What do the plots tell you about who survived?
- Are there any surprising findings?
- How might this information influence your model?

****

**Task 4: Compute the Correlation Matrix**

Understanding the correlation between features can help in feature selection.

**Instructions:**

1. Compute the correlation matrix for df_train.
2. Visualize the correlation matrix using a heatmap.
3. Analyze the correlations and identify any strong relationships.

<details>
<summary>What is correlation?</summary>

Correlation is a way to measure how two things are related. If one thing changes, does the other thing change in a similar way?

- Positive correlation: When one goes up, the other goes up too (e.g., the more you study, the better your grades).
- Negative correlation: When one goes up, the other goes down (e.g., the more you exercise, the lower your weight might be).
- No correlation: No clear relationship between the two (e.g., shoe size and intelligence).

It tells us how strongly two things are connected!

(More about correlation: https://medium.com/@abdallahashraf90x/all-you-need-to-know-about-correlation-for-machine-learning-e249fec292e9)
</details>

Code:

```python
# Compute correlation matrix
correlation_matrix = train_df.corr()

#Visualize the Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Training Dataset Features')
plt.show()
```

**Questions:**

1. Which features are most strongly correlated with 'Survived'?
2. Are there features that are highly correlated with each other?

****

**Task 5: Perform Feature Selection with RFE**

Use Recursive Feature Elimination (RFE) to select the most important features.

https://medium.com/@rithpansanga/logistic-regression-for-feature-selection-selecting-the-right-features-for-your-model-410ca093c5e0

**Instructions:**

1. Prepare the data:
   - Drop irrelevant features (e.g., 'PassengerId', 'Survived', etc.).
2. Initialize the logistic regression model.
3. Perform RFE to select a subset of features.
4. List the selected features.

**Questions:**

1. Do the selected features make sense based on your earlier analysis?
2. Are there any features you expected to be important that were not selected?

**** 

**Task 6: Build and Evaluate the Logistic Regression Model**

Now, use the selected features to build and evaluate a logistic regression model.

**Instructions:**

1. Split the data into training and testing sets.
2. Train the logistic regression model using the selected features.
3. Make predictions on the test set.
4. Evaluate the model using a confusion matrix and classification report.

Hint:
Look at the tutorial in the above task.

**Questions:**

1. How well does your model perform?
2. Are there any improvements you can make? 
   - What happens if you remove features and predict again?

****

**Task 7: Understand Evaluation Metrics**

It's important to understand what the evaluation metrics mean.

**Instructions:**

1. Accuracy: Calculate the accuracy of your model.
2. Log Loss: Understand what log loss represents.
3. AUC: Compute the Area Under the ROC Curve (AUC).

Hint:

accuracy_score(y_test, y_pred)

log_loss(y_test, y_pred_probability)

roc_auc_score(y_test, y_pred_probablity)

**Questions:**

1. What do these metrics tell you about your model?
2. Is there a trade-off between precision and recall?


****

**Task 8: Cross-Validation**

Perform cross-validation to assess the model's performance more robustly.

**Instructions:**

1. Use K-Fold Cross-Validation with ```cv=10```.
2. Evaluate using different scoring metrics: accuracy, log loss, and AUC.
3. Report the mean scores.

**Questions:**

1. Does cross-validation provide similar results to your initial evaluation?
2. Why is cross-validation important?

****

# Conclusion
In this lab, you have:

- Preprocessed and split the Titanic dataset.
- Created new features and performed exploratory data analysis.
- Used Recursive Feature Elimination to select important features.
- Built and evaluated a logistic regression model.
- Understood and interpreted various evaluation metrics.
- Performed cross-validation to assess model performance.

Understanding these steps is crucial in building robust predictive models and can be applied to various datasets and machine learning algorithms.



## Usefull links
You can find usefull information about feature engeneering [here][feature-eng-tutorial].

You can find useful information about logistic regression [here][sklearn-logreg].

[pandas cheatsheet][pandas-cheatsheet].
[matplotlib cheatsheet][matplotlib-cheatsheet].
[seaborn cheatsheet][seaborn-cheatsheet].
[sklearn cheatsheet][sklearn-cheatsheet].


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.






<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
<!-- shields -->
[issues-shield]: https://img.shields.io/github/issues/umaimehm/Intro_to_AI_2021.svg?style=for-the-badge
[issues-url]: https://github.com/DAVE3625/DAVE3625-24H/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/DAVE3625/DAVE3625-24H/blob/main/Lab1/LICENSE

<!-- images -->
[names]: img/names.png
[cC]: img/columnsCount.png
[cC2]: img/columnsCount2.png
[pl1]: img/plot1.PNG
[table-task4]: img/table4.png
[table-task4-m]: img/table4-marked.png
[final-df]: img/finalDf.png
[nan]: img/nan.png
[skewed-age]: img/skewed_age.png

<!-- documentation -->
[pandas-doc]: https://pandas.pydata.org/docs/
[numpy-doc]: https://numpy.org/doc/stable/
[seaborn-doc]: https://seaborn.pydata.org/
[sklearn-doc]: https://scikit-learn.org/0.21/documentation.html

<!-- tutorials -->
[feature-eng-tutorial]: https://github.com/PacktPublishing/Python-Feature-Engineering-Cookbook
[pandas-cheatsheet]: https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf
[matplotlib-cheatsheet]: https://matplotlib.org/cheatsheets/
[sklearn-cheatsheet]: https://www.datacamp.com/cheat-sheet/scikit-learn-cheat-sheet-python-machine-learning
[seaborn-cheatsheet]: https://www.kaggle.com/code/themlphdstudent/cheat-sheet-seaborn-charts
[sklearn-logreg]: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

<!-- links -->
[flip-bool]: https://www.kite.com/python/answers/how-to-invert-a-pandas-boolean-series-in-python
[lakeforest.edu]: http://campus.lakeforest.edu/frank/FILES/MLFfiles/Bio150/Titanic/TitanicMETA.pdf
[get-dummies]: https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
[get-dummies-vs-onehot]: https://albertum.medium.com/preprocessing-onehotencoder-vs-pandas-get-dummies-3de1f3d77dcc
[regex]: https://www.geeksforgeeks.org/python-regex-cheat-sheet/
[solution]: solution.ipynb
[whatis-roc]: https://www.geeksforgeeks.org/how-to-plot-roc-curve-in-python/
[whatis-auc]: https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
[whatis-featureselection]: http://scikit-learn.org/stable/modules/feature_selection.html
[whatis-correlation]: https://medium.com/@abdallahashraf90x/all-you-need-to-know-about-correlation-for-machine-learning-e249fec292e9
[whatis-logisticregression]: https://developers.google.com/machine-learning/crash-course/logistic-regression/
