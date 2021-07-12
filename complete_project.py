# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 15:34:56 2021

@author: niklas
"""

# Datenaufbereitung

## Import der benötigten Bibliotheken und Datensets

import pandas as pd

ratings = pd.read_csv("C:\\Users\\nikla\\Desktop\\Explo\\ratings.csv")
ratings = ratings.drop(columns = ["timestamp"])
ratings.movieId = ratings.movieId.astype(float)

## Bearbeitung des Datensets und Aufstellen verschiedenster Kennzahlen zu dem Datenset

print("Anzahl der User:")
print(len(ratings["userId"].unique().tolist()))
user_count = len(ratings["userId"].unique().tolist())

movies = pd.read_csv("C:\\Users\\nikla\\Desktop\\Explo\\movies_metadata.csv")
movies = movies[["genres", "id", "original_title"]]
movies = movies.rename(columns = {"id" : "movieId", "original_title" : "title"})

movies = movies[~(movies.movieId.isin(["1997-08-20", "2012-09-29", "2014-01-01"]))]

movies.movieId = movies.movieId.astype(int)

### Erstellen des zusammengeführten Datensets

ratings = pd.merge(ratings, movies,on='movieId',how='left')

## Weitere Bearbeitung und Kennzahlen

print("Anzahl der NA-Zeilen:")
print(ratings['title'].isna().sum())

ratings = ratings.dropna()

print("Zeilenanzahl vor Filter:")
print(len(ratings["userId"].tolist()))

groups = ratings.groupby(by = ["movieId"]).count()
groups["movieId"] = groups.index
groups = groups.drop(columns = ["rating", "genres", "title"])
groups = groups.rename(columns = {"userId" : "count"})
print("Anzahl aller Filme:")
print(len(groups["movieId"].tolist()))

small_movies = groups.loc[groups["count"] < 0.02 * user_count]
small_movieId = small_movies.movieId.tolist()
print("Anzahl der entfernten Filme:")
print(len(small_movieId))

ratings = ratings.loc[~ratings["movieId"].isin(small_movieId)]
print("Zeilenanzahl nach Filter:")
print(len(ratings["userId"].tolist()))

print("Durch Filter entfernte User:")
print(user_count - len(ratings["userId"].unique().tolist()))

## Erstellung des "breiten" Datensets

ratings = ratings.drop(columns = ["genres", "movieId"])

ratings_wide = ratings.pivot_table(index = ["userId"],
                             columns = ["title"],
                             values = ["rating"])

#ratings_wide.to_csv(r"C:\\Users\\nikla\\Desktop\\ratings_wide.csv")

## Aufgrund Perfromanceprobleme Kürzung des finalen Datensets auf 50.000 Zeilen

short = ratings_wide.head(50000)

#short.to_csv(r"C:\\Users\\nikla\\Desktop\\short.csv")

print("data preperation done")

# Algorithmus

## Import der benötigten Bibliotheken

import random
#import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

## Seed-Festlegung für Reproduzierbarkeit

random.seed(100)

## Import des aufbereiteten Datensets und finale Bearbeitung dieses

#ratings = pd.read_csv(r"C:\\Users\\nikla\\Desktop\\ratings_wide.csv")
#ratings = pd.read_csv("C:\\Users\\nikla\\Desktop\\short.csv")
ratings = short

headers = ratings.iloc[0]
headers[0] = "userId"

ratings  = pd.DataFrame(ratings.values[2:], columns=headers)

ratings = ratings.rename(columns = {300.0 : "300"})

ratings = ratings.fillna(0.0)

ratings = ratings.apply(pd.to_numeric)

# Machinen Learning
## Clustering

kmeans = KMeans(n_clusters=50,n_init=5, random_state=10)
kmeans.fit(ratings.drop(columns = ["userId"]))
ratings.loc[:,'cluster'] = kmeans.labels_

headers = ratings.columns.tolist()
headers = headers[-1:] + headers[:-1]

ratings = ratings[headers]

ratings_short = ratings[["cluster", "userId"]]
cluster_deviation = ratings_short.groupby(["cluster"]).count()

## Recommender System
### Datensatzaufteilung in train und test

trenner = int(0.8 * len(ratings["userId"]))

ratings_train = ratings[:trenner]
ratings_test = ratings[trenner:]
ratings_test_prediction = ratings_test[["userId", "cluster"]]

### KNN

k = 10
knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(ratings_train.drop(columns = ["cluster", "userId"]), ratings_train["cluster"])

prediction = knn.predict(ratings_test.drop(columns = ["cluster", "userId"]))

ratings_test_prediction["prediction_knn"] = prediction

errors = 0

for i in range(0, len(ratings_test_prediction["userId"])):
               if ratings_test_prediction["cluster"].tolist()[i] != ratings_test_prediction["prediction_knn"].tolist()[i]:
                   errors += 1

print("Anzahl aller Fehler der Clusterprediction durch KNN:")
print(errors)

### Random Forest

forest = RandomForestClassifier(n_jobs=2, random_state=10)
forest.fit(ratings_train.drop(columns = ["cluster", "userId"]), ratings_train["cluster"])

prediction = forest.predict(ratings_test.drop(columns = ["cluster", "userId"]))

ratings_test_prediction["prediction_randomforest"] = prediction

errors = 0

for i in range(0, len(ratings_test_prediction["userId"])):
               if ratings_test_prediction["cluster"].tolist()[i] != ratings_test_prediction["prediction_randomforest"].tolist()[i]:
                   errors += 1

print("Anzahl aller Fehler der Clusterprediction durch Random Forest:")
print(errors)

### Naive Bayes

bayes = GaussianNB()
bayes.fit(ratings_train.drop(columns = ["cluster", "userId"]), ratings_train["cluster"])

prediction = bayes.predict(ratings_test.drop(columns = ["cluster", "userId"]))

ratings_test_prediction["prediction_bayes"] = prediction

errors = 0

for i in range(0, len(ratings_test_prediction["userId"])):
               if ratings_test_prediction["cluster"].tolist()[i] != ratings_test_prediction["prediction_bayes"].tolist()[i]:
                   errors += 1

print("Anzahl aller Fehler der Clusterprediction durch Naive Bayes:")
print(errors)

## Ausarbeitung der Top 10 Filme je Cluster

cluster_movies = ratings.drop(columns = ["userId"]).groupby(["cluster"]).sum()
cluster_movies["cluster"] = cluster_movies.index

clusters = []
cluster_movie_1 = []
cluster_movie_2 = []
cluster_movie_3 = []
cluster_movie_4 = []
cluster_movie_5 = []
cluster_movie_6 = []
cluster_movie_7 = []
cluster_movie_8 = []
cluster_movie_9 = []
cluster_movie_10 = []

for i in (cluster_movies["cluster"]):
    clusters.append(i)
    cluster_row = cluster_movies.loc[cluster_movies["cluster"] == clusters[i]].values.flatten().tolist()
    top_movies = []
    for k in range(0, 10):
        column_index = cluster_row.index(max(cluster_row))
        top_movies.append(cluster_movies.columns[column_index])
        cluster_row[column_index] = -1
    cluster_movie_1.append(top_movies[0])
    cluster_movie_2.append(top_movies[1])
    cluster_movie_3.append(top_movies[2])
    cluster_movie_4.append(top_movies[3])
    cluster_movie_5.append(top_movies[4])
    cluster_movie_6.append(top_movies[5])
    cluster_movie_7.append(top_movies[6])
    cluster_movie_8.append(top_movies[7])
    cluster_movie_9.append(top_movies[8])
    cluster_movie_10.append(top_movies[9])

cluster_top_movies = pd.DataFrame({"Cluster" : clusters, "ClusterMovie1" : cluster_movie_1, "ClusterMovie2" : cluster_movie_2, "ClusterMovie3" : cluster_movie_3, "ClusterMovie4" : cluster_movie_4, "ClusterMovie5" : cluster_movie_5, "ClusterMovie6" : cluster_movie_6, "ClusterMovie7" : cluster_movie_7, "ClusterMovie8" : cluster_movie_8, "ClusterMovie9" : cluster_movie_9, "ClusterMovie10" : cluster_movie_10})


# Beispiel Vorhersage einer Recommandation eines "neuen" Users

user_id = 8

user = ratings[user_id:user_id + 1]
user["cluster"] = forest.predict(user.drop(columns = ["cluster", "userId"]))
user_cluster_movies = cluster_top_movies.loc[cluster_top_movies["Cluster"] == user.iloc[0]["cluster"]].drop(columns = ["Cluster"]).values.flatten().tolist()
print("User and Cluster:")
print(user["cluster"])

user_movies = user.loc[:, (user != 0).any(axis=0)]
user_movies = user_movies.drop(columns = ["userId", "cluster"]).columns
recommandation = list(set(user_cluster_movies) - set(user_movies))
print("User movie recommandation:")
print(recommandation)

print("algorithm done")