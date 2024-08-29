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
  <a href="https://github.com/umaimehm/Intro_to_AI_2021/tree/main/Lab3">
    <img src="img/header.jpeg" alt="Data anonymization" width="auto" height="auto">
  </a>

  

  <p align="center">
    Data Creation - Data anonymization - Timeseries<br \>
    <br />
    ·
    <a href="https://github.com/umaimehm/Intro_to_AI_2021/issues">Report Bug</a>
    ·
    <a href="https://github.com/umaimehm/Intro_to_AI_2021/issues">Request Feature</a>
  </p>



<!-- ABOUT THE LAB -->
## About The Lab

In this lab, we will have a look at two python packages. One for creating fake data and one on how to anonymize data 

We will also take a look on how to work with datetime objects



We will be using [pandas][pandas-doc], [laundromat][laundromat] and [faker][faker]

## Solution
[Task 1 - 2][solution-fak]

[Task 3][solution-fly]



## New imports

Faker and laudromat is new packages and we need to install them. Go to your anaconda prompt, make sure the right enviromet is active (in most of your cases dave3625) and write:

```python
pip install Faker
# Since people have problems with laundromat, we will drop this part of the task
#pip install spacy==2.3.2
#pip install laundromat
```
 After the install, you should be able to use
 ```python
from faker import Faker
import pandas as pd
#from laundromat.spacy.spacy_model import SpacyModel
from faker.providers.credit_card import Provider as CreditCardProvider #Add creditcards to faker
```


## Tasks
**1. Faker**
Create a dataframe with 100 fake persons. Let the dataframe look like this

![t1][table1]

Hints:
How to use faker:
```python
fake = Faker(['no_NO']) #Create a faker with norwegian Names and structures

#Faker calls you will need:
fake.name(),fake.address(),fake.ssn(),fake.credit_card_number(),fake.ipv4()
```
[Faker documentation][faker-doc]

How to create a empty df:
```python
df = pd.DataFrame(columns=['c1','c2', ... ,'cn']) # Create a empty df

```

How to add data to a df, in a for loop:
```python
row = "Tekst1", "Tekst2", .... , "TekstN"
df.loc[i]=row
```

[How to do a for loop][for-loop]

**2. Anonymization**
When a customer send your company an email that contain personal information, you could train a model to remove that data Laundromat is one package that can do that. Lets just try to remove Name and Address, since laundromat can do that "out of the box"

Let's try Laundromat on the 10 first rows in the dataframe, using a for loop.

```python
textArray = [] #Create a empty array
# Do a for loop, and assign values to name and adress
    row=df.iloc[i]
    name = row[0] 
    adress = row[1]
    ssn = row[2]
    cc = row[3]
    #Using f string we can now create a new string with the values, and
    #append (add) that string to the text array
    textArray.append(f'Hi, my name is {name}. I wonder if you deliver to {adress}. My credi card nr is {cc} and my ssn is {ssn}')
```

TIP:
If you want to look at the entire table, the for loop can be slow, and best practice is to use df.iterrows()
<details>
  <summary>Click to show code</summary>
  
```python
for index, row in df.iterrows():

    name = row[0] 
    adress = row[1]
    ssn = row[2]
    cc = row[3]
    #Using f string we can now create a new string with the values, and
    #append (add) that string to the text array
    textArray.append(f'Hi, my name is {name}. I wonder if you deliver to {adress}')
```
</details>

*Due to issues with spacy (a package that laundromat need to operate correctly), we removed this part of the lab. You can still see how it works in the [solution][solution-fak]*

<details>
  <summary>Removed laundromat because of run issues!</summary>

We now have a array with 10 text strings. Using a for-loop we will now see if laundromat can catch all 
Names and addresses
First we need to create a SpacyModel, this is done by:
```python
nlp = SpacyModel()
```

```python
for line in textArray:
    entities = nlp.predict(line)
    replaced_text_1 = nlp.replace(line, replacement="entity", replacement_char=":^)")
    print(line)
    print(replaced_text_1+'\n')
```

As you can see, we didn't hide ssn and credtit card. To do this we need to add regext to laundromat



```python
nlp.add_patterns(lookup=True)
nlp.print_regex_labels()
```

Try to run the above for-loop again to see if we catched credit card and ssn this time.

You will most likely see that we don't have a 100% success rate, since we didn't spend alot of time setting up
the function. 

</details>

Feel free to play with both [Faker][faker] and [Laundromat][laundromat] on your own time.

**3. Time series**

Consider opening up a new notebook for this task.

Imports:
```python
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
```

Read in [this][flightdata] file as we did in lab1 and 2
<details>
  <summary>Click to show code</summary>

```python
url = "TEXT"
#Find the raw url from the github repo
df = pd.read_csv(url)
```

</details>
Convert the columns: datetime_val, dep_time, arr_time and sched_arr_time to datetime type within the dataframe.

**Hint:**
```python
#Convert one column to datetime
df['column'] = pd.to_datetime(df['column'])
```

Calculate the following:
1. air_time dep_time - arr_time
2. Delay arr_time – sched_arr_time
3. Delay as a percent of air time (I have yet to produce a good solution for this)

Hint:
Directly converting to datetime will produce a error because some of the timestamps are < 24:00
You can fix this by ignoring that pic of data

```python
df['column'] = pd.to_datetime(df['column'], errors='coerce') 
#errors='coerce'  puts a NaT value (NaT = dummy var for missing value)
```
Some flights go past midnight as well, but the dataset don’t handle this. So a plain that took off 2013-01-01 23:57:00  and landed 2013-01-02 02:57:00 will come up in the dataset as 01-01 23:57:00  -> 01-01 02:57:00 (negative flight time)

This task take more effort to correct, but it’s a nice task for those of you who feel this lab was to easy. We will later on fix negative flight time in a faster and hacky way.

<details>
  <summary>A proposed solution to the problem</summary>

```python
for index, row in df.iterrows():
    #if arr_time is less then dep_time
    if (row['arr_time']<row['dep_time']):
        #add one day to arr_time
        df.loc[index, 'arr_time'] = (row['arr_time'])+ datetime.timedelta(days=1)
    if (row['sched_arr_time']<row['dep_time']):
        df.loc[index, 'sched_arr_time'] = (row['sched_arr_time'])+ datetime.timedelta(days=1)
      
```
</details>

When the columns are converted to datetime, you can add and substract
```python
df['newCol'] = df["Col1"]-df["Col2"]
```

On the column air_time, lets fix the negative values:
To do this we need to use df.iterrows(), do a check if row['air_time'].days is less then 0
If so, replace that value with 24 hours - air_time + abs(days)
It can be a challenge to write this code, so for those of you who are not to familiar with python, check the hint. (But try to google a bit and figure it out)

<details>
  <summary>Try to understand this code, because it covers many concepts of pandas and timeseries</summary>

  *Disclamber: This code is not 100% correct, but is written to show how you can work with timeseries. In some cases it will provide the wrong result.*

```python
#For every row in df
for index, row in df.iterrows():
    #if air_time is negative
    if (row['air_time'].days < 0):
        #Find the row with df.loc                      Take 24 hrs, - air_time + negative days (could be replaced with 1)
        df.loc[index, 'air_time'] = datetime.timedelta(hours=24)-(row['air_time'] + datetime.timedelta(abs(row['air_time'].days)))
```
</details>

Find the delay as a percent of air_time:
```python
psudo code
percent_delay = (100 * delay)/airtime
```

As you can see by doing a boxplot, we have some big outliers. 

![outliers][otli]

By analyzing the row with minimum value, we can conclude that these values most probably are a dataset error. We can try to fix the values, but lets just remove outliers.

**Finding outliers**
This is a huge dataset, and the hacky way to fix negative airtime, don't work 100%, so lets remove some outliers.
Do a df["percent_delay"].plot.box() to check outliers, and a df["percent_delay"].describe() to see the min and max values
To make it easy to do remove outliers later on, lets make it a function:


```python
from pandas.api.types import is_numeric_dtype
def remove_outlier(df):
    low = .05
    high = .95
    quant_df = df.quantile([low, high])
    if is_numeric_dtype(df):
        df = df[(df > quant_df.loc[low]) & (df < quant_df.loc[high])]
    return df
  ```

and call the function

```python
df["percent_delay"] = remove_outlier(df["percent_delay"])
```

Check the box plot and the describe again to check if the outliers are removed.

**Extra**

See if you can create some nice looking charts (plot) with this data.

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

[table1]: img/table1.png
[dfinfo]: img/dfinfo.png
[df2info]: img/df2info.png
[flightdata]: data/flight.csv
[otli]: img/sched_arr_time.png

<!-- documentation -->
[pandas-doc]: https://pandas.pydata.org/docs/reference/index.html#api
[numpy-doc]: https://numpy.org/doc/stable/
[seaborn-doc]: https://seaborn.pydata.org/api.html
[faker-doc]: https://faker.readthedocs.io/en/master/

<!-- tutorials -->
[feature-eng-tutorial]: https://github.com/PacktPublishing/Python-Feature-Engineering-Cookbook
[pandas-cheatsheet]: https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf
[for-loop]: https://www.w3schools.com/python/python_for_loops.asp

<!-- links -->
[api-key]: https://frost.met.no/auth/requestCredentials.html
[regex]: https://www.geeksforgeeks.org/python-regex-cheat-sheet/
[solution]: solution.ipynb
[faker]: https://github.com/joke2k/faker
[laundromat]: https://github.com/navikt/laundromat
[frost]: https://frost.met.no/python_example.html
[solution-fak]: faker.ipynb
[solution-fly]: flight.ipynb



