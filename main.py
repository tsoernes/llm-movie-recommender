from config import TMDB_API_KEY
import requests
import polars as pl
from pathlib import Path

movies_path = Path().absolute() / "movies.parquet"


def fetch_movies() -> pl.DataFrame:
    """
    Fetch movies from themoviedb and return them in a DataFrame
    """
    # Set the API endpoint and parameters
    url = "https://api.themoviedb.org/3/discover/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "sort_by": "vote_average.desc",
        "vote_count.gte": 10_000,
        "page": 1,
    }

    # Send a GET request to the API
    response = requests.get(url, params=params)
    response.raise_for_status()

    # Extract the movie data from the response
    movies = response.json()["results"]

    # Create a Polars DataFrame from the movie data
    df = pl.DataFrame(movies)

    # Fetch genre names
    url = "https://api.themoviedb.org/3/genre/movie/list"
    params = {"api_key": TMDB_API_KEY}
    response = requests.get(url, params=params)
    response.raise_for_status()
    id_and_genre = response.json()["genres"]

    # Map genre ids to genre names
    id_to_genre = {x["id"]: x["name"] for x in id_and_genre}

    # Add genre names from genre ids
    df_genres = df["genre_ids"].map_elements(
        lambda li: [id_to_genre[id_] for id_ in li], return_dtype=pl.List(pl.String)
    )
    df = df.with_columns(df_genres.alias("genres"))

    # Drop unecessary columns
    df = df[["id", "title", "overview", "genre_ids"]]
    return df


def main():
    # Fetch movies from TMDb if they are not stored on disk
    if movies_path.exists():
        df = pl.read_parquet(movies_path)
    else:
        df = fetch_movies()
        df.write_parquet(movies_path)
