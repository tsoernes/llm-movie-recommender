from config import TMDB_API_KEY
import requests
import polars as pl
from pathlib import Path
from functools import lru_cache
import chromadb

movies_path = Path().absolute() / "movies.parquet"
db_path = Path().absolute() / "movies.chromadb"
tmdb_url = "https://api.themoviedb.org/3"


def fetch_top_rated_movies(min_votes: int = 1_000) -> list[dict]:
    """
    Fetch top rated movies from themoviedb

    Parameters
    ----------
    min_votes
        Only fetch movies that have this number of votes or more

    Returns
    -------
    A list of movies, each movie in a dictionary
    """
    # Set the API endpoint and parameters
    url = tmdb_url + "/discover/movie"

    # Fetch movies page by page until there are no more
    movies = []
    i = 1
    while True:
        params = {
            "api_key": TMDB_API_KEY,
            "sort_by": "vote_average.desc",
            "vote_count.gte": min_votes,
            "page": i,
        }

        # Send a GET request to the API and raise an exception if it fails
        response = requests.get(url, params=params)
        response.raise_for_status()

        # Extract the movie data from the response
        movies_ = response.json()["results"]
        if not movies_:
            break
        movies.extend(movies_)
        i += 1

    return movies


def search_movie(query: str) -> list[dict]:
    """
    Search for a movie by title from themoviedb

    Parameters
    ----------
    query
        Query search term. Use the title of the movie.

    Returns
    -------
    A list of result movies, each movie in a dictionary
    """
    # Set the API endpoint and parameters
    url = tmdb_url + "/search/movie"

    # Fetch movies page by page until there are no more
    movies = []
    i = 1
    while True:
        params = {
            "api_key": TMDB_API_KEY,
            "query": query,
            "page": i,
        }

        # Send a GET request to the API and raise an exception if it fails
        response = requests.get(url, params=params)
        response.raise_for_status()

        # Extract the movie data from the response
        movies_ = response.json()["results"]
        if not movies_:
            break
        movies.extend(movies_)
        i += 1

    return movies


@lru_cache
def get_id_to_genre() -> dict[int, str]:
    """Return a mapping from genre id to genre name"""
    # Fetch genre names
    url = tmdb_url + "/genre/movie/list"
    params = {"api_key": TMDB_API_KEY}
    response = requests.get(url, params=params)
    response.raise_for_status()
    id_and_genre = response.json()["genres"]

    # Map genre ids to genre names
    id_to_genre = {x["id"]: x["name"] for x in id_and_genre}

    return id_to_genre


def prep_movies(movies: list[dict]) -> pl.DataFrame:
    """
    Prepare movie data for embedding

    Attributes
    ----------
    movies
        List of movies

    Returns
    -------
    polars.DataFrame
    """
    # Create a Polars DataFrame from the movie data
    df = pl.DataFrame(movies)

    id_to_genre = get_id_to_genre()
    # Add genre names from genre ids
    df_genres = df["genre_ids"].map_elements(
        lambda li: [id_to_genre[id_] for id_ in li], return_dtype=pl.List(pl.String)
    )
    df = df.with_columns(df_genres.alias("genres"))

    # Extract year from release date
    df = df.with_columns(df["release_date"].str.head(4).alias("year"))

    # Merge all information into a single text column
    text = (
        "Movie title: "
        + df["title"]
        + "\nYear: "
        + df["year"]
        + "\nOverview: "
        + df["overview"]
        + "\nGenres: "
        + df["genres"].list.join(", ")
    )
    df = df.with_columns(text.alias("text"))

    # Drop unecessary columns
    df = df[["id", "title", "year", "overview", "genres", "text"]]
    return df


def select_query_movie(
    query: str, autoselect_first: bool = False
) -> pl.DataFrame | None:
    """
    Query for movie. Select
    """
    movies = search_movie(query)
    if not movies:
        print(f"Did not find any movies for {query=}. Try again.")
        return
    df = prep_movies(movies)
    print_movies_s = (
        pl.Series(range(len(df))).cast(str)
        + ". "
        + df["title"]
        + " ("
        + df["year"]
        + "): "
        + df["overview"]
    )
    if autoselect_first:
        ix = 0
    else:
        print("Select from the following movies:")
        print(print_movies_s.str.concat("\n")[0])
        ix = 0
        first_inp = True
        while first_inp or ix not in range(len(df)):
            first_inp = False
            try:
                ix = input(f"Select movie (0-{len(df)}): ")
                ix = int(ix)
            except ValueError:
                pass
    df = df[int(ix), :]
    return df


def main():
    # Fetch movies from themoviedb if they are not stored on disk
    if movies_path.exists():
        df = pl.read_parquet(movies_path)
        print(f"Read {len(df)} movies")
    else:
        movies = fetch_top_rated_movies()
        df = prep_movies(movies)
        df.write_parquet(movies_path)
        print(f"Fetched {len(df)} movies")

    client = chromadb.PersistentClient(path=str(db_path))
    # Here you can add a custom embedding function (e.g.OpenAI)
    # using the embedding_function argument. The default is a all-MiniLM-L6-v2
    # Sentence Transformer.
    collection = client.get_or_create_collection("movies")

    if not collection.count():
        print(
            "Adding movies to database. This will download a language model on first initialization"
        )
        # Add docs to the collection.
        # chromadb handles tokenization, embedding, and indexing automatically.
        collection.add(
            documents=df["text"].to_list(),
            ids=df["id"].cast(str).to_list(),
            # metadatas=df[["col1", "col2"]].to_dicts(), # filter on these!
        )

    query_df = None
    while query_df is None:
        query = input("Query for a movie: ")
        query_df = select_query_movie(query)

    # Query/search
    results = collection.query(
        query_texts=query_df["text"].to_list(),
        n_results=10,
        # where={"metadata_field": "is_equal_to_this"}, # optional filter
        # where_document={"$contains":"search_string"}  # optional filter
    )
    if docs := results["documents"]:
        print(*(x for x in docs[0]), sep="\n\n")


if __name__ == "main":
    main()
