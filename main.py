from config import TMDB_API_KEY
import requests
import polars as pl


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

# Check if the request was successful
if response.status_code != 200:
    print("Failed to retrieve movies:", response.status_code)

# Parse the JSON response
data = response.json()

# Extract the movie data from the response
movies = data["results"]

# Create a Polars DataFrame from the movie data
df = pl.DataFrame(movies)

# Fetch genre names
url = "https://api.themoviedb.org/3/genre/movie/list"
params = {"api_key": TMDB_API_KEY}
response = requests.get(url, params=params)
if response.status_code != 200:
    print("Failed to retrieve movies:", response.status_code)
data = response.json()
id_and_genre = data["genres"]
# Map genre ids to genre names
id_to_genre = {x["id"]: x["name"] for x in id_and_genre}

# Drop unecessary columns
df = df[["id", "title", "overview", "genre_ids"]]
# Add genre names
df = df.with_columns(
    df["genre_ids"]
    .map_elements(lambda li: [id_to_genre[id_] for id_ in li], return_dtype=list)
    .alias("genres")
)
