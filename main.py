from config import TMDB_API_KEY
import requests
import polars as pl
import numpy as np
from bs4 import BeautifulSoup
from tmdbv3api import TMDb, Discover, Genre

tmdb = TMDb()
tmdb.api_key = TMDB_API_KEY
discover = Discover()
movies = discover.discover_movies(
    {
        "sort_by": "vote_average.desc",
        "vote_count.gte": 10_000,
    }
)

id_and_genre = Genre().movie_list()["genres"]
id_to_genre = {x["id"]: x["name"] for x in id_and_genre}

df = pl.from_dicts([dict(movie) for movie in movies])
# Drop unecessary columns
df = df[["id", "title", "overview", "genre_ids"]]
# Convert from tmdb object dtype to list
df = df.with_columns(df["genre_ids"].map_elements(list, return_dtype=list))
# Add genre names
df = df.with_columns(
    df["genre_ids"]
    .map_elements(lambda li: [id_to_genre[id_] for id_ in li], return_dtype=list)
    .alias("genres")
)
