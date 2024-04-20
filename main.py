from config import TMDB_API_KEY
from config import TMDB_API_KEY
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from config import TMDB_API_KEY

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}

min_votes = 10000
base_url = f"https://www.imdb.com/search/title/?title_type=feature&num_votes={min_votes},&sort=user_rating,desc&"

html_elements_list = []
n_movies = 100
for i in range(0, n_movies, 50):
    url = base_url + f"start={i}&ref_=adv_nxt"
    html_file = requests.get(url, headers=headers)
    html_beautiful_file = BeautifulSoup(html_file.content, "html.parser")
    html_elements_list.extend(
        html_beautiful_file.find_all(
            "div", attrs={"class": "lister-item mode-advanced"}
        )
    )
