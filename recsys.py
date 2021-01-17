# recsys.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os import path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from ast import literal_eval

movies = pd.read_csv(r'.\data\movies_metadata.csv', low_memory=False)
credits = pd.read_csv(r'.\data\credits.csv').assign(id= lambda x: x['id'].astype('int')).drop_duplicates()
keywords = pd.read_csv(r'.\data\keywords.csv').assign(id= lambda x: x['id'].astype('int')).drop_duplicates()

def plot_score(df):
    fig, ax = plt.subplots(figsize=(10,10))
    df['vote_average'].plot(kind='hist', ax=ax, bins=20)
    ax.set_xticks(np.linspace(0, 10, num=21))
    ax.set_title('Movielens Score Distribution')

def create_index(df):
    return df[['title', 'release_date']].assign(release_date= lambda x: x['release_date'].str.split('-').str[0])

def weighted_rating(x, m, C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

def get_recommendations(movie_name, cosine_sim, data_source, index):
    """
    movie_name :: string of a movie name
    cosine_sim :: similarity matrix
    data_source :: original dataframe with the source data
    index :: a modified version of the data-source with movie names and release year
    """
    # Get the index of the movie that matches the title
    movie_count = index.loc[index['title'] == movie_name, 'title'].count()
    if movie_count > 1:
        movie_year = str(input(f'There are more than one movies named {movie_name}.\n Please provide the year the movie was released >>> '))
        movie_index = index.loc[(index['title'] == movie_name) & (index['release_date'] == movie_year)].index
    else:
        movie_index = index.loc[index['title'] == movie_name].index
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[movie_index[0]]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    sim_scores = [i[1] for i in sim_scores]

    # Return the top 10 most similar movies
    return (data_source.loc[data_source.index.isin(movie_indices), ['title', 'release_date']]
                       .reindex(movie_indices)
                       .set_index('title')
                       .assign(similarity_score=sim_scores))

def transform_metadata(df):
    return (df.drop([19730, 29503, 35587]) # bad id rows
                .assign(id= lambda x: x['id']
                .astype('int'))
                .merge(credits, on='id')
                .merge(keywords, on='id')
                .reset_index(drop=True))

def literal_eval_columns(df):
    df = df.copy()
    features = ['cast', 'crew', 'keywords', 'genres']
    for feature in features:
        df[feature] = df[feature].apply(literal_eval)
    return df 

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names
    #Return empty list in case of missing/malformed data
    return []

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

def create_tfidf_vector(df, columne_name):
    tfidf = TfidfVectorizer(stop_words='english')
    return tfidf.fit_transform(df[columne_name])

def create_count_vector(df, columne_name):
    count = CountVectorizer(stop_words='english')
    return count.fit_transform(df[columne_name])

def calculate_linear_kernel(tfidf_matrix):
    return linear_kernel(tfidf_matrix, tfidf_matrix)

def calculate_similarity_score(count_matrix):
    return cosine_similarity(count_matrix, count_matrix)