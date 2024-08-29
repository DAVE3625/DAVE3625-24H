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
<h3 align="center">Dave3625 - Lab5</h3>
<p align="center">
  <a href="https://github.com/umaimehm/Intro_to_AI_2021/tree/main/Lab5">
    <img src="img/head.jpg" alt="Linear Regression" width="auto" height="auto">
  </a>

  

  <p align="center">
    Looking into clustering<br \>
    <br />
    ·
    <a href="https://github.com/umaimehm/Intro_to_AI_2021/issues">Report Bug</a>
    ·
    <a href="https://github.com/umaimehm/Intro_to_AI_2021/issues">Request Feature</a>
  </p>



<!-- ABOUT THE LAB -->
## About The Lab

In this lab, we will look at unsupervised learning - clustering.
We will first do clustering on seed data, then for clarity we will use a real world example, using geodata from OsloBysykkel

We will be using [pandas][pandas-doc], [sklearn][sklearn-doc]



## New imports

```python
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans2, whiten

```

<details>
  <summary>Click to show all imports</summary>

```python
%matplotlib inline

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans2, whiten
```

</details>


## Tasks

**Seed data**

**1. Read the data**

Read the dataset found in ./data/seeds_dataset.txt
You can still use pd.read_csv, but as sep you should use "\t"

Lets then see if we can spot some corelations in the set

```python
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')
```
<details>
  <summary>Output</summary>

![corr plot][corr]

The correlation is not as clear as last lab, but let's see what the algorithm can do non the less
  
</details>

Drop the result column from the dataset, and store it in a variable.

<details>
  <summary>Hint</summary>

```python
result = df["result"]
df.drop(columns="result", inplace=True)
df.head()
```

</details>

Prepare the data for the model

```python
x = np.array(df)
clusters = 3
```

and train the model

```python
kmeans = KMeans(n_clusters=clusters, random_state=0, max_iter=300).fit(x)
cluster=kmeans.labels_
```

copy the labels given into the dataframe, so we can use seaborn pairplot

```python
df["y"]=cluster
df.head()
```

We can now do a pairplot to see how it turned out

```python
sns.pairplot(df, vars=df.columns[:-1], hue = "y", markers=["o", "s", "D"])
```

and we can also compare it to the original labels

```python
df["result"]=result
sns.pairplot(df, vars=df.columns[:-2], hue = "result", markers=["o", "s", "D"])
```

<details>
  <summary>Output</summary>

![pair plot][pair]

</details>

**OsloBysykkel data**

Feel free to start a new notebook for this task.

**1. Read in the dataset from oslobysykkel, and look for correlation between rows**



```python
# Load Data Set
year  = "2021"
month = "10"
URL = f"https://data.urbansharing.com/oslobysykkel.no/trips/v1/{year}/{month}.csv"
df = pd.read_csv(URL)
df.head()
```

**2. Preparing the data**

Since we have covered data preparation, you will be provided working code snippets.
```python
#Getting only uniqe station ID's
df = df.sort_values('start_station_id', ascending=False)
sdf = df.drop_duplicates(subset='start_station_name', keep='first')
sdf = sdf.sort_values('start_station_id', ascending=True)
sdf.head()
```

And lets define a class to store stations in:

```python
#A class to keep station data
class station:
    def __init__(self, id,longitude,latitude,name, change = 0, zone = 0):
        self.id = id
        self.long = longitude
        self.lat = latitude
        self.name = name
        self.change = change
        self.zone = zone
    
    def updateChange(self, tick):
        self.change += tick
    def setZone(self, zone):
        self.zone = zone
    def getZone(self):
        return self.zone        
    def getId(self):
        return self.id
    def getName(self):
        return self.name
    def getLongLat(self):
        return [self.long,self.lat]
    def getChange(self):
        return self.change
    def export(self):
        return [self.id,self.name,self.long,self.lat,self.change,self.zone]
```

Then, put each station in a class object
```python
#Assign each station to it¨s own station object
listOfStations=[]
for index, row in sdf.iterrows():
                  #id,latitude,longitude
    tmp = station(row[3],row[6],row[7],row[4])
    listOfStations.append(tmp)
```

**3. Train some clusters**

First, lets define how may clusters we want. 
```python
clusters = ? #input number here
```

Then, export coordinates from each station to a coordinate list

```python
cor = []
for station in listOfStations:
    cor.append(station.getLongLat())
coordinates= np.array(cor)
```

We will now check two kmeans libraries

***1: scipy.cluster.vq.kmeans2***

```python
#Run the algorithm
#x = center of cluster
#clust1 = what item belong to what cluster
x, clust1 = kmeans2(whiten(coordinates), clusters, iter = 150)  
```

***2: sklearn.cluster.KMeans***

```python
kmeans = KMeans(n_clusters=clusters, random_state=0, max_iter=150).fit(coordinates)
clust2=kmeans.labels_
```

So whats the difference? Let's try to plot them to have a look:
First we define a way to plot 
```python
def plot_cluster(cluster):
    sns.set(rc={'figure.figsize':(15,10)})
    sns.scatterplot(data=coordinates, x=coordinates[:,0], y=coordinates[:,1], hue=cluster, palette="deep")
    plt.title("Stations clustered by kmeans", fontsize=20)
    plt.xlabel('Latitude', fontsize=12)
    plt.ylabel('Longitude', fontsize=12)
```
Now, lets plot the clusters:
```python
plot_cluster(clust1)
```
and
```python
plot_cluster(clust2)
```
This will produce something like:

![cluster plot][clustplot]

The two plots might be identical, but you can play around with iter / max_iter to see how this affect the output.

Why did we do this with two algorithms?
Well, to show you that it's easy to switch between algorithms once the data is prepared. We can also check with other clustering methods by changing one line

```python
#Different linking methods 'ward', 'average', 'complete', 'single'   
from sklearn.cluster import AgglomerativeClustering 
newAlg = AgglomerativeClustering(linkage = 'average', n_clusters=clusters)
newAlg.fit(coordinates)
clust3=newAlg.labels_

plot_cluster(clust3)
```

This ends the machine learning part of the lab, but we will look at how we can represent the data we trained on, in a interactive map.

**3 Visualize the data**


This is for those of you who want to dig deeper into data visualization, and not important for the course itself.



Before continuing: You will need a API key for the rest of this lab to work.
Go to [mapbox.com][api-key] and sign up and create your API key. Copy it and do

```python
token = " key "
```

Lets download some new packages. Go to your Anaconda prompt, activate the correct env and write these lines

```python
pip3 install mapboxgl
pip3 install coloraide
conda install requests
```
Within the same notebook we worked on in task 1 and 2, lets continue by importing the new packages and assigning each station the cluster it belongs to. First choose what algorithm you want to plot from:

```python
from mapboxgl.viz import *
from mapboxgl.utils import *
from coloraide import Color
```

```python
#Cluster we want to go forward with
clust = clust1
for i in range(len(listOfStations)):
    listOfStations[i].setZone(clust[i]+1)
newDf = pd.DataFrame(columns = ["Id", "Name", "Longitude", "Latitude", "Change", "Zone"])
```

So we filled the stations with the cluster they belong to, and made a new dataframe.
Now, lets fill that frame

```python
#Fill newDf with data
for s in listOfStations:
    series = pd.Series(s.export(), index = newDf.columns)

    newDf = newDf.append(series, ignore_index=True)
```

Since every station has a latitude and longitude, we can create a geojson file and plot them onto a map.

```python
#Create a geojson object for mapbox plot
df_to_geojson(newDf, filename='points.geojson',
              properties=['Id', 'Name','Change','Zone'],
              lon='Latitude', lat='Longitude', precision=3)
geoFile ='points.geojson'
````

We also want different colors on the different clusters, so lets make a list of colors:

```python
nrClust = clust.max()+1  #Find number of clusters
#Create the colors
color_stops = Color("red").steps(["yellow", "purple","green", "blue" ], steps=nrClust)
c = ([x.to_string() for x in color_stops])
l = []
#Converting the colors to the right format
for i in range(nrClust):
    c[i]=c[i].replace(" ", ",")
    l.append([i+1,c[i]])
```

And finaly, lets make the map

```python

#Plot clustered zones on the map
center =(10.77837,59.928349)  #Center the map on Oslo
zoom = 10
viz = CircleViz(geoFile,
                access_token=token,
                height='500px',
                radius=3,
                color_property = "Zone",
                color_stops = l,
                center = center,
                zoom = zoom,
                below_layer = 'waterway-label'
              )
viz.show(True)
```

and the result will be something like this

![map plot][mapplot]

## More hints

This section will be updated after the first lab session

## Useful links




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
[clustplot]: img/clustplot.png
[mapplot]: img/mapplot.png
[pair]: img/pair.png

<!-- documentation -->
[pandas-doc]: https://pandas.pydata.org/docs/reference/index.html#api
[numpy-doc]: https://numpy.org/doc/stable/
[seaborn-doc]: https://seaborn.pydata.org/api.html
[sklearn-doc]: https://scikit-learn.org/stable/modules/classes.html


<!-- tutorials -->
[skclust]: https://scikit-learn.org/stable/modules/clustering.html


<!-- links -->
[api-key]: https://account.mapbox.com/access-tokens/
[solution]: solution.ipynb



