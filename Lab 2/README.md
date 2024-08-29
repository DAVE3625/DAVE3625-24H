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
<h3 align="center">Dave3625 - Lab2</h3>
<p align="center">
  <a href="https://github.com/umaimehm/Intro_to_AI_2021/tree/main/Lab2">
    <img src="img/header.png" alt="Data wrangling" width="auto" height="auto">
  </a>

  

  <p align="center">
    Feature engeneering - on the Titanic dataset <br \>This is a classic dataset used in many data mining tutorials and demos -- perfect for getting started with exploratory analysis and building binary classification models to predict survival.
    <br />
    ·
    <a href="https://github.com/umaimehm/Intro_to_AI_2021/issues">Report Bug</a>
    ·
    <a href="https://github.com/umaimehm/Intro_to_AI_2021/issues">Request Feature</a>
  </p>
</p>


<!-- ABOUT THE LAB -->
## About The Lab

In this lab, we will start to look at feature engineering on the Titanic dataset.

*The titanic and titanic2 data frames describe the survival status of individual passengers 
on the Titanic. The titanic data frame does not contain information from the crew, but it 
does contain actual ages of half of the passengers.* - *[LakeForest.edu][lakeforest.edu]*

We will be using [pandas][pandas-doc], [numpy][numpy-doc] and [seaborn][seaborn-doc].

**[Solution added][solution]**

## New imports

We will use a new package in this lab. Don't worry, it's a python standard package called re, so remember to add
```python
#Import modules
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
```
to your imports.

Load the titanic set found under /data/Titanic.csv as we did in Lab1

Hint:
Click view as "Raw" and copy the url

## Tasks
**1. Check for null and nan values**

In lab 1 we used df.isna().sum() to check for nan values. Since we didn’t find many, we converted blankspace into np.nan to help us procide. Lets try it again on this dataset.
```python
df.isna().sum()
# You can also use df.isnull().sum()
```
![nan][nan]
Fill Age, Fare and Embarked with sensible values. (Embarked could be filled with "S")
Since the nan values are defined differently in this dataset, we can use the function right out of the box.

During the last lab people asked how to fill nan values with meaningful values. What a meaningful value is differ from dataset to dataset, but lets just add a median value for all numeric columns.
Hint: Filling columns with median values
```python
df["column"] = df["column"].fillna(df["column"].median())
# To fill with a set value or a char, change .fillna("desired value")
``` 
We can also see that many people has a NaN for Cabin. It’s not as easy as just fill a dummy value here. We could fill with “no cabin”, but for machine learning, we like to have numerical or bool values. To achieve this, lets make a new bool column:

Cabin = True / False
And set all NaN values = False, all other = True
```python
df["HasCabin"] = df.Cabin.isnull()
```
do a df.head() and you can see we have a new column, but there is an error. 

Hint: 

[Try to find the error before checking the hint][flip-bool]

Adding a new column based on data available is considered creating a new feature.


**2. Adding a feature**

Lets extract the title for each person on the boat, and make a new column called «Title»
As we can see from the data set, the syntax for names is
LastName, Title. RestOfName

![names][names]
A easy way to extract a sertan string is to use 
```python
lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1)
```

*What is this syntax? It's called regex, and a [explanation can be found here][regex]*

And in our case we would like to put this data in a new column, so we can run 
```python
df["Title"] = df.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1)) 
```

Check with df.head() that you now have acolumn called Title.
We can now see how many has each title. This can be done in many ways, but calling 
```python
df["column"].value_counts() // you need to replace "column" 
# whit the name of the column you want to count
```
![count][cC]

As we can see from the count, we have 18 titles, some of them with only one person. 
Replace Mlle and Ms with "Miss", and Mme with "Mr" using:
```python
df["column"] = df["column"].replace({'xxx':'yyy', 'jjj':'iiii', … 'uuu':'iii'})
```


We can also package all titles with few persons into a unique category
```python
df["column"] = df["column"].replace(["x","y", … , "n"], "Unique")
```
And do a new count of titels and see if you get something simellare to this:
![recount][cC2]

You can also produce a plot with
```python
sns.countplot(x='Title', data=df); //Seaborn countplot
plt.xticks(rotation=45);
```
![plot][pl1]


**3. Convert Age and Fare into categorical data.**

 This can be done using pandas qcut function
```python
df['CatAge'] = pd.qcut(df["Age"], q=4, labels=False )
```
do this for both Age and Fare.

**4. Convert dataframe to binary data**

To train a dataset easily, we want all data to be numerical. To achieve this, we need to drop columns that don’t make sense converting to a numerical value. At this point, your dataframe should look something like this:

![dataframe][table-task4]

Identify columns that we need to drop to convert to a numerical dataset.

[Solution][table-task4-m]

Drop the tabels you identified with

```python
df = df.drop(["column1", ... , "columnN"], axis=1)
```
Converting to binary  data is a trivial task in pandas. Try using [pd.get_dummies][get-dummies]
This works well for analytic tasks, but you could also use [OneHotEncoder() for machine learning tasks.][get-dummies-vs-onehot]

**All done**

At the end of the lab you should have a table looking something like 
![finalTable][final-df]

In this lab you have:
* engineered some new features such as 'Title' and 'Has_Cabin'
* dealt with missing values, binned your numerical data and transformed all features into numeric variables


## More hints

This section will be updated after the first lab session

## Usefull links
You can find usefull information about feature engeneering [here][feature-eng-tutorial]

[pandas cheatsheet][pandas-cheatsheet]

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.






<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
<!-- shields -->
[issues-shield]: https://img.shields.io/github/issues/umaimehm/Intro_to_AI_2021.svg?style=for-the-badge
[issues-url]: https://github.com/umaimehm/Intro_to_AI_2021/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/umaimehm/Intro_to_AI_2021/blob/main/Lab1/LICENSE

<!-- images -->
[names]: img/names.png
[cC]: img/columnsCount.png
[cC2]: img/columnsCount2.png
[pl1]: img/plot1.PNG
[table-task4]: img/table4.png
[table-task4-m]: img/table4-marked.png
[final-df]: img/finalDf.png
[nan]: img/nan.png

<!-- documentation -->
[pandas-doc]: https://pandas.pydata.org/docs/reference/index.html#api
[numpy-doc]: https://numpy.org/doc/stable/
[seaborn-doc]: https://seaborn.pydata.org/api.html

<!-- tutorials -->
[feature-eng-tutorial]: https://github.com/PacktPublishing/Python-Feature-Engineering-Cookbook
[pandas-cheatsheet]: https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf

<!-- links -->
[flip-bool]: https://www.kite.com/python/answers/how-to-invert-a-pandas-boolean-series-in-python
[lakeforest.edu]: http://campus.lakeforest.edu/frank/FILES/MLFfiles/Bio150/Titanic/TitanicMETA.pdf
[get-dummies]: https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
[get-dummies-vs-onehot]: https://albertum.medium.com/preprocessing-onehotencoder-vs-pandas-get-dummies-3de1f3d77dcc
[regex]: https://www.geeksforgeeks.org/python-regex-cheat-sheet/
[solution]: solution.ipynb



