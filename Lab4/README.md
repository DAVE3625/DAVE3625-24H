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
<h3 align="center">Dave3625 - Lab4</h3>
<p align="center">
  <a href="https://github.com/umaimehm/Intro_to_AI_2021/tree/main/Lab4">
    <img src="img/header.jpg" alt="Linear Regression" width="auto" height="auto">
  </a>

  

  <p align="center">
    Training your first model<br \>
    <br />
    ·
    <a href="https://github.com/umaimehm/Intro_to_AI_2021/issues">Report Bug</a>
    ·
    <a href="https://github.com/umaimehm/Intro_to_AI_2021/issues">Request Feature</a>
  </p>



<!-- ABOUT THE LAB -->
## About The Lab

In this lab, we will train our first model using linear regression.
The dataset used is created for this lab, and we will be able to identify a linear relationship between some variables and the result.

We will be using [pandas][pandas-doc], [sklearn][sklearn-doc]



## New imports

```python
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

```

<details>
  <summary>Click to show all imports</summary>

```python
%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Yes, I like to sort imports based on length

```

</details>


## Tasks
**1. Read in data.csv, and look for correlation between rows**


<details>
  <summary>How to read a csv from github</summary>

```python
url = "TEXT"
#Find the raw url from the github repo
df = pd.read_csv(url)
```

</details>

Use pandas .corr() to create a correlation matrix.
You can then plot the matrix to get a better understanding for the data

<details>
  <summary>Hint</summary>

```python
#corrMatrix is the variable where you saved the correlation matrix

#Standard corr plot
plt.matshow(corrMatrix)
plt.show()

#Another style of corr plot
corrMatrix.style.background_gradient(cmap='coolwarm')
```
*Check out [this page][cmap] for other cmap (color maps) for the plot.*

From the matrix, we can see that Var1 and Result is highly correlated. We can also see that Var3 might have a correlation.

![corrplot][corr]


</details>

We can also look at the correlation with scatter plots:

```python
df.plot.scatter(x = 'Var1', y = 'Result')
```

This is the column with the highest correlation with Result plotted against it:

![scatter-plot][scatter1]

We can also plot more columns together, like this:

```python
ax1 = df.plot(kind='scatter', x='Var1', y='Result', color='r')    
ax2 = df.plot(kind='scatter', x='Var2', y='Result', color='g', ax=ax1)    
ax3 = df.plot(kind='scatter', x='Var3', y='Result', color='b', ax=ax1)
```

![scatter-combo][scatter2]

As we can see from this plot, different columns have different min/max values. To solve this we need to scale the columns to a given interval.
This can be done by importing sklearn  and use the preprocessing.MinMaxScaler().


<details>
  <summary>Scaling</summary>

To scale a dataset, you can run:

```python
x = df2.values #returns a numpy array
scaler = preprocessing.MinMaxScaler().fit(x)
x = scaler.transform(x)
df = pd.DataFrame(x)
#To keep column names do
#df[list(df.columns)] = scaler.transform(df)
#instead of line 3 and 4
#But we want to just have a numeric id for now, since it will help us later.
```

Output
  
![scaling][scale1]

If you want to unscale, do:

```python
x = df2.values #returns a numpy array
x = scaler.inverse_transform(x)
df = pd.DataFrame(x)
df.head()
```

![scaling][scale2]

Tip: If you put the scaled dataset in df2, you can compare them easy.

</details>

After scaling the data, we can plot them together in a new scatter plot

![scatter-scaled][scatter3]

Lets make an another scatter plot where we plot each column against "Result" (or column 6)



```python
#Scatterplot all columns against last column
fig, ax = plt.subplots(df.shape[1]-1, figsize=(15, 15)) #Figsize ( length, height )
for i in range(df.shape[1]-1):   #This loop is why we wanted to keep column name numeric, and not keep original names
    
    ax[i].scatter(x = df[i], y = df[6])
    ax[i].set_xlabel("Column " + str(i))
    ax[i].set_ylabel("Y")
fig.tight_layout()
plt.show()

```


<details>
  <summary>Show output</summary>
  
![scatter-combo][scatterall]

If you compare this to the correlation plot, you'll identify the same columns having a relation

![corrplot][corr]
  
  </details>
  
## Let's train a model 

In the task above, we found that Var1 and Var3 had a high correlation with the output. Let's try to train a model on Var1 and evaluate the result.

```python
x = df2.values #returns a numpy array
x = scaler.inverse_transform(x)
df = pd.DataFrame(x)
df.head()
```

First let's split the set in a training and a testing set

```python
X = pd.DataFrame(df[0]) #Var1
y = pd.DataFrame(df[6]) #Result

#Now, split the set in training and testing set
#test_size = 0.33 tell the function that 1/3 of values should be put in test arrat
#Random state is a variable that seeds the random generator. In that way
#you'll get the same training and testing set each run
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```
More on [train-test split][traintest]

We now have 2/3 of the values in Var1 stored in X_train and 1/3 stored in X_test.
You can now create a linear regressor model:

```python
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X_train, y_train)  # perform linear regression
Y_pred = linear_regressor.predict(X_train)  # make predictions
```
To see the result, you can plot the prediction against the result column

```python
plt.scatter(X_train, y_train)             #Plot blue dots with real data
plt.plot(X_train, Y_pred, color='red')    #Plot red line with prediction
plt.show()                                #Show the plot
print( "MSE = "+str(metrics.mean_squared_error(y_train,Y_pred))) #Calculate MSE
```
<details>
  <summary>Show output</summary>

![model 1][mod1]

</details>

Let's check how good our model works on the test data

```python
Y_pred = linear_regressor.predict(X_test)  # Predict the model on X_test
plt.scatter(X_test, y_test)
plt.plot(X_test, Y_pred, color='red')
plt.show()
print( "MSE = "+str(metrics.mean_squared_error(y_test,Y_pred)))
```
<details>
  <summary>Show output</summary>

![model 2][mod2]

</details>

By chance we got a lower error on the test set. This is good, because the lower MSE is, the more accurate is the model.

### Can we improve the output by creating a feature?

As we discovered from the correlation plot, Var1 and Var3 was correlated with output. What happens if we create: Var7 = Var1 x Var3 

*Remember Var1 = column 0, Var 3 = column 2*

```python
df[7] = df[0]*df[2]  #Create a new var, based on Var1 and Var3

X = pd.DataFrame(df[7])  #Lets skip making train test set for now, and just
Y = pd.DataFrame(df[6])  #load the entire dataset
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()
print( "MSE = "+str(metrics.mean_squared_error(Y,Y_pred)))
```
<details>
  <summary>Show output</summary>

![model 3][mod3]

MSE = 0.4! That's impressive.

</details>

Now, lets manually check with the first row from data_test.csv

```python
                       # Var1 * Var3
linear_regressor.predict([[95*63]]) #Expected output = 62.46
```
Seems like the model is working good with new data as well.
We can now import data_test.csv and try the model on the entire set:
```python
df2 = pd.read_csv(...) #Read test-set into df
#Make a new column for Var1 * Var3
df2["Combined"] = df2["Var1"]*df2["Var3"]
```
First, create a new column with Var1 and Var3 *(call it Combines to use the code that follow)*

Extract:
	The combined value from df2 and put it in X
	The result from df2 and put it in Y
	df2 Var1 and put it in X1
	df2 Var3 and put it in X2

	column 7 from df and put it in Xt
	column 0 from df and put it in X1t
	column 2 from df and put it in X2t
	column 6 from df and put it in Yt
	and set:
	dataSet = [X,X1,X2]
	trainingSet = [Xt,X1t,X2t]

<details>
  <summary>Click to show code snippet</summary>

```python
#Set X to the combined set
X = pd.DataFrame(df2["Combined"])
#Y to result
Y = pd.DataFrame(df2["Result"])
#And make sets for Var1 and Var2
X1 = pd.DataFrame(df2["Var1"])
X2 = pd.DataFrame(df2["Var3"])

#And lets do the same for the test set
Xt = pd.DataFrame(df[7])
X1t = pd.DataFrame(df[0])
X2t = pd.DataFrame(df[2])
Yt = pd.DataFrame(df[6])
#Put the data in list, so we can test the different sets in a for loop
dataSet = [X,X1,X2]
trainingSet = [Xt,X1t,X2t]
```


</details>

Now let's try to train the a model on Var1, Var3 and on Var1 x Var3

```python
fig, ax = plt.subplots(3, figsize=(15, 15)) #Figsize ( length, height )
models = [] #List to save the different model for later use
caps = ["Var1*Var2", "Var1", "Var2"] #Caption for the plots
MSE = [] #list to save mean square error
for i in range(3):  
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(trainingSet[i], Yt)  # perform linear regression
    Y_pred = linear_regressor.predict(dataSet[i])  # make predictions
    ax[i].scatter(dataSet[i], Y) #Plot blue dots from test set
    ax[i].plot(dataSet[i], Y_pred, color='red') #Plot predicted from train
    err=metrics.mean_squared_error(Y,Y_pred)
    ax[i].set_xlabel(caps[i] + " MSE = " +str(err )) #Set caption
    ax[i].set_ylabel("Y") #Set y lable
    MSE.append(err) #Calculate and save mse for model
    models.append(linear_regressor) #Save
fig.tight_layout()
plt.show()
```
We can now see how the choice of variable is closely connected with the models accuracy.

![model 5][mod5]

We can also check the models with some data not included in any of the sets:

```python
#Var1 = 79
#Var2 = 135
#Var3 = 36
#Var4 = 62
#Var5 = 6
#Var6 = 14
#Result = 31.61

#Var1 * Var3
v1=models[0].predict([[79*36]]) #Make a prediction on and store the result in V1
#Var 1
v2=models[1].predict([[79]])
#Var3
v3=models[1].predict([[36]])
print(f"Prediction with:\nVar1*Var3 = {v1[0][0]}\nVar1 = {v2[0][0]}\nVar3 = {v3[0][0]}")
print("Correct value = 31.61")
```

<details>
  <summary>Output</summary>

```
Prediction with:
Var1*Var3 = 30.950911117701498
Var1 = 53.80770756065712
Var3 = 25.79824010435377
Correct value = 31.61
```

</details>


## So we got a model, what now?

We can now save the model. That way we can use it later without doing all the training of it again. 
By using pickle we can easy save data to a file:

```python
import pickle
# save the model to disk
filename = 'my_model.sav' #Give it whatever name + extension you want
pickle.dump(models[0], open(filename, 'wb'))

##############################################################
#Load the model again, and test to see if it works.
loaded_model = pickle.load(open(filename, 'rb'))
loaded_model.predict([[79*36]])
```

## More hints

This section will be updated after the first lab session

## Useful links
You can find useful information about feature engineering [here][feature-eng-tutorial]

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

[corr]: img/corr.png
[scatter1]: img/scatter1.png
[scatter2]: img/scatter2.png
[scatter3]: img/scatter3.png

[scatterall]: img/scatterall.PNG
[scale1]: img/scale1.png
[scale2]: img/scale2.png
[mod1]: img/mod1.png
[mod2]: img/mod2.png
[mod3]: img/mod3.png
[mod4]: img/mod4.png
[mod5]: img/threeMod.png

<!-- documentation -->
[pandas-doc]: https://pandas.pydata.org/docs/reference/index.html#api
[numpy-doc]: https://numpy.org/doc/stable/
[seaborn-doc]: https://seaborn.pydata.org/api.html
[sklearn-doc]: https://scikit-learn.org/stable/modules/classes.html


<!-- tutorials -->
[feature-eng-tutorial]: https://github.com/PacktPublishing/Python-Feature-Engineering-Cookbook
[pandas-cheatsheet]: https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf
[for-loop]: https://www.w3schools.com/python/python_for_loops.asp
[traintest]: https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/

<!-- links -->
[api-key]: https://frost.met.no/auth/requestCredentials.html
[regex]: https://www.geeksforgeeks.org/python-regex-cheat-sheet/
[solution]: solution.ipynb
[faker]: https://github.com/joke2k/faker
[laundromat]: https://github.com/navikt/laundromat
[frost]: https://frost.met.no/python_example.html
[cmap]: https://matplotlib.org/stable/tutorials/colors/colormaps.html


