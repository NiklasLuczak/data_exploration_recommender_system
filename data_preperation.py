# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 15:34:56 2021

@author: niklas
"""

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

short.to_csv(r"C:\\Users\\nikla\\Desktop\\short.csv")

print("done")
