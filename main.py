from config import TMDB_API_KEY
import requests
import polars as pl
import numpy as np
from bs4 import BeautifulSoup
from tmdbv3api import TMDb, Discover, Genre

tmdb = TMDb()
tmdb.api_key = TMDB_API_KEY
discover = Discover()
movies = discover.discover_movies({
    'sort_by': 'vote_average.desc',
    'vote_count.gte': 10_000,
})

id_and_genre = Genre().movie_list()['genres']
id_to_genre = {x['id']: x['name'] for x in id_and_genre}

df = pl.from_dicts([dict(movie) for movie in movies])
df = df[['id', 'title', 'overview', 'genre_ids']]
