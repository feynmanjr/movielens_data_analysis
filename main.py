"""
This is a practice work on how to analyse data for a movie recommendation system.
"""

import os
import re
import pandas as pd
import numpy as np 
from ipywidgets import widgets
from IPython.display import display, HTML
import matplotlib.pyplot as plt 

## The first step is to import the .dat file into pandas dataframe.

## The rating table consists of 4 columns:
## - 'user_id' : the id of the user who put the comment.
## - 'movie_id' : the id of the movie that the user commented on it.
## - 'rating' : the rating for that movie from that user.
## - 'timestamp' : the time at which the rating was given.
## We read it to a dataframe called "df_rating".
ratings_header = ['user_id', 'movie_id', 'rating', 'timestamp']
df_rating = pd.read_table('ratings.dat', sep='::', header= None, names= ratings_header, engine='python')


## The users table consists of 5 columns:
## - 'user_id' : the id of the user.
## - 'gender' : the gender of the user which is either 'male' or 'female'.
## - 'age' : the age range of this user.
## - 'occupation' : the job of this user.
## - 'zipcode' : the zipcode that this user lives.
## We read it to a dataframe called "df_users".
users_header = ['user_id', 'gender', 'age', 'occupation', 'zipcode']
df_users = pd.read_csv('users.dat', sep='::', header= None, names= users_header, engine='python')

## The movie table consists of 3 columns:
## - 'movie_id' : the id of the movie.
## - 'title' : the title of the movie followed by the production year.
## - 'genre' : the the genre(s) of the movie.
## We read it to a dataframe called "df_movies".
movies_header = ['movie_id', 'title', 'genres']
df_movies = pd.read_csv('movies.dat', sep='::', header= None, names= movies_header, engine='python')

## To make future work easier, we create a new column in the movies dataframe for the production year.
df_movies['year'] = df_movies.title.str[-5:-1]

## A printing statement to see what the max year is.
print "this is s list of movies untill {} which created at {}".format(
    df_movies.iloc[np.argmax(df_movies['year'])][1][:-7],
    df_movies.iloc[np.argmax(df_movies['year'])][3])
    
    
    
## we need to convert the genre column in the movies dataframe to a logical (or 0 and 1) array. We, then, add a new column to the movies dataframe as 'genre_id'
def genre_id(genre):
    x = np.zeros(17, dtype=np.int)
    for g in genre:
        if g =="Action"     : x[0] = 1
        if g =="Adventure"  : x[1] = 1
        if g =="Animation"  : x[2] = 1
        if g =="Children's" : x[3] = 1
        if g =="Comedy"     : x[4] = 1
        if g =="Crime"      : x[5] = 1
        if g =="Documentary": x[6] = 1
        if g =="Drama"      : x[7] = 1
        if g =="Fantasy"    : x[8] = 1
        if g =="Film-Noir"  : x[9] = 1
        if g =="Musical"    : x[10] = 1
        if g =="Mystery"    : x[11] = 1
        if g =="Romance"    : x[12] = 1
        if g =="Sci-Fi"     : x[13] = 1
        if g =="Thriller"   : x[14] = 1
        if g =="War"        : x[15] = 1
        if g =="Western"    : x[16] = 1
    return x
genre_list = []
for i in range(len(df_movies)):
    d1 = df_movies['genres'][i].split('|')
    d = genre_id(d1)
    genre_list.append(d)
df_movies['genre_id'] = genre_list


## Dropping movies with less than or equal to some specific values.
cutoff_rating = 100

## Check the aggregated values of the rating of each movies with their statistical parameters.
movie_ratings = df_rating.groupby('movie_id').agg({'rating': [np.size, np.mean, np.std]})
movie_ratings['popularity'] = pd.DataFrame(movie_ratings.loc[:,('rating', 'size')]).apply(lambda x: (x - np.min(x))/(np.max(x) - np.min(x)))
movie_ratings_1 = movie_ratings[movie_ratings.loc[:,('rating', 'size')] > cutoff_rating]


## A printing statement to see what the maximum and minimum ratings are.
print "Minimum and maximum number of ratings are {} and {}".format(
    np.min(movie_ratings.loc[:,('rating', 'size')]),
    np.max(movie_ratings.loc[:,('rating', 'size')]))
## Plotting the rating distribution to see how it looks.
number_of_bins = 26
bins_range =np.arange(cutoff_rating, 750, number_of_bins)plt.hist(movie_ratings_1.loc[:,('rating', 'size')], bins=bins_range)
plt.xlim(100,500)
plt.show()

## A printing statement to see how many movies were  dropped based on this cutoff.
print "Out of {} movies, only {} of them have more than {} ratings".format(
    len(movie_ratings), len(movie_ratings_1), cutoff_rating)
f = pd.DataFrame(movie_ratings_1.loc[:, ('popularity', '')].to_frame(name = 'popularity').to_records())
movie_df = pd.merge(df_movies,f, on='movie_id', how='inner')


## How important it is to recommend a popular movie.
weight_of_popularity = 0

def movies_relation(i, j):
    sum_i = sum(movie_df['genre_id'][i])
    sum_j = sum(movie_df['genre_id'][j])
    if (sum_i == 0 or sum_j == 0):
        sim = 0
    else:
        inner_prod = np.inner(movie_df['genre_id'][i]*1.0,movie_df['genre_id'][j]*1.0)
        sim = inner_prod/max(sum_i , sum_j)
    popular = movie_df.loc[i,'popularity'] * movie_df.loc[j,'popularity']
    return [sim, popular]


def similar_to_this_movie(i, n):
    s = pd.Series()
    dic = {}
    for j in range(len(movie_df)):
        if j == i:
            continue
        else:
            value = movies_relation(i,j)[0] + weight_of_popularity * movies_relation(i,j)[1]
            index = movie_df.loc[j,'title']
            s = s.set_value(index, value)    
    return s.nlargest(n)

## Some print statements to check the work.
print movie_df.loc[0]
print movies_relation(2,1)
print similar_to_this_movie(0,6)
