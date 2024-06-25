# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # LLM Movie Recommender

# %% [markdown]
# Denne notebooken demonstrerer bruk av en språkmodel (Language Model - LM) og en vektordatabase for anbefaling av filmer.
#
# Denne metoden løser det såkalte "cold start" problemet for anbefalingsalgoritmer, hvor det ikke finnes
# data fra brukeren av systemet eller data fra andre brukere av systemet til å gi anbefalinger.
#
# I tilfellet hvor det finnes data på hvilke brukere som foretrekker hvilke filmer, er det enklere å lage en anbefalingsalgoritme.
# Da er det mulig å bruke denne dataen til å gi anbefalinger - for eksempel, blandt alle brukere som har gitt høy vurdering (rating)
# for "Star Wars", er det kanskje gitt høy rating for "Back to The Future", og da kan man gi anbefaling av denne.
#
# I stedet, kan vi bruke en språkmodell til å finne film-anbefalinger basert på hvor likt sammendraget av filmen er (semantisk likhet), og basert på hvor ofte
# navnet på filmen forekommer i samme kontekst som andre filmer.
#
# Denne anbefalingsalgoritmen ligner på clustering. Datasettet består av unlabelled data - informasjon om filmer, uten informasjon
# om hvilke filmer som er like eller hvilke brukere som like hvilke filmer. Algoritmen gir informasjon om hvor like filmer er hverandre,
# og kan dermed brukes til å gi anbefalinger. For eksempel kan vi forvente (håpe) på at filmer som handler å reise i tid ved hjelp av tidsmaskiner
# ligger nær hverandre - det vil si at cluster-algoritmen mener disse filmene har en likhet med hverandre.

# %%

# %% [markdown]
# La oss først begynne med å hente filmdata fra den åpne filmdatabasen TMDb ved å bruke API-et.
#
# Dette er veldig rett fram. Vi henter informasjon om alle tilgjengelige filmer som har 1000 brukervurderinger eller mer. 

# %%
# %autoreload 2
from main import *
from pprint import pprint

# %%
from config import TMDB_API_KEY
import requests

tmdb_url = "https://api.themoviedb.org/3"

# %%
??fetch_popular_movies

# %%
movies = fetch_popular_movies()

# %% [markdown]
# API-et returnerer grunnleggende informasjon om filmene. Av interesse her er tittel, sjanger, år, og sammendrag.
#

# %%
pprint(movies[0])

# %% [markdown]
#
# Her er følgende av interesse:
# - Tittel
# - Filmsjanger
# - Sammendrag
# - År

# %% [markdown]
# Vi samler denne informasjonen i en dataframe for enklere og raskere prosessering.
# Vi bruker `polars` som et raskere og mer moderne alternativ til `pandas` selv om det har lite å bety i denne sammenheng
# siden datasettet er så lite.
#
# Først henter vi filmsjangernavn fra API-et og bruker disse istedet for filmsjanger-ID'er.
#
# Deretter samler vi informasjonen vi ønsker å bruke til å sammenligne i en kolonne ("text") - nemlig  tittel, år, sammendrag og sjangere.

# %%
import polars as pl
from functools import lru_cache

??get_id_to_genre.__wrapped__

# %%
??prep_movies

# %%
df = prep_movies(movies)

# %% [markdown]
# La oss se på innholdet til DataFramen

# %%
len(df)

# %%
df.head(1)

# %% [markdown]
# Dette gir følgende tekst for filmen. Det er denne teksten vi skal bruke i vektordatabasen til å finne lignende filmer.

# %%
print(df[0, "text"])

# %% [markdown]
# Vi instansierer en vektordatabase som vi skal bruke til å finne liknende dokumenter. Her finnes det mange alternativer, og vi har valgt ChromaDB hvor vi kan velge en valgfri embedding-funksjon.

# %%
import chromadb
client = chromadb.Client()

# %% [markdown]
# Lag en collection og legg til dokumenter.
# `chromadb` gjør all jobb  automatisk. Først tokeniseres dokumentteksten til tokens, før den blir embeddet av en innebygd språkmodell og deretter indeksert av databasen slik at den effektivt kan finne nærliggende naboer til dokumentet. Embeddingen fanger essensen av syntaktisk og semantisk mening til hele dokumentet.
#
# Det mulig å bruke en custom funksjon for å embedde dokumenter. Som standard brukes det en LM som heter `all-MiniLM-L6-v2` fra librariet Sentence Transformer. Dette er en språkmodell på 23 millioner paremetere trent på 1 milliard tokens (ord / sub-ord) som outputter 384 dimensionale vektorer. Til sammenligning er GPT-4 fra OpenAI 1.7 billioner paremetre og trent på 13 billioner tokens og har 3072 dimensionale setnings-vektorer.
#

# %%
title_collection = client.get_or_create_collection("movie_titles")

# %%
if not title_collection.count():
    title_collection.add(documents=df["title"].to_list(), ids=df["id"].to_list())

# %%
embedding = title_collection.query(query_texts=["Star Wars"], n_results=1, include=["embeddings"])["embeddings"]
embedding = embedding[0][0]
print(len(embedding))

# %%
print(embedding[:10])

# %% [markdown]
# Databasen tilbyr flere alternativer for å måle likheten mellom dokumenter. Likheten mellom to dokumenter er gitt av distansen mellom de to tilhørende vektor embeddingene. Som default bruker ChromaDB $L^2$ distanse, gitt av:
#
# $ d=∑_i(A_i - B_i)^2 $
#
# Hvor $A_i$ er index $i$ i vektor embeddingen til dokument $A$

# %%
results = title_collection.query(query_texts=["Star Wars"], n_results=20)

# %%
print(results["documents"])

# %% [markdown]
# Språkmodellen har fanget god semantisk betydning kun utifra filmtittelen. "Star Wars" har høy likhet til "Return of the Jedi" da begge er Star Wars filmer men har svak syntaktisk likhet. Vi ser også at LMen henter ut andre Sci-Fi filmer som "Star Trek" og "Interstellar".
#
# For å demonstrere at LMen gir større likhet mellom dokumenter av lik semantisk betydning, legger vi til dokumentet "Star Alliance". I motsetning til tidligere dokumenter, er ikke dette en film, men navn på et selskap som er en en allianse av flyselskaper.

# %%
title_collection.add(documents=["Star Alliance"], ids=["9999"])

# %%
results = title_collection.query(query_texts=["Star Wars"], n_results=20)

# %%
print(results["documents"])

# %% [markdown]
# Databasen gir relativt svak likhet mellom "Star Wars" og "Star Alliance" til tross for at de har høy syntaktisk likhet. 

# %% [markdown]
# La oss lage en collection som bruker hele filmbeskrivelsen

# %%
collection = client.get_or_create_collection("movies")
if not collection.count():
    collection.add(documents=df["text"].to_list(), ids=df["id"].to_list())

# %%
results = collection.query(query_texts=["Back to the future"], n_results=20)

# %%
for d in results["documents"][0]: print(d, '\n')

# %% [markdown]
# De aller fleste anbefalingene treffer ganske bra. De fleste dreier seg om reise i tid eller tidsmaskiner, self om det er noen anbefalinger som ikke virker til å være relevant og som antageligvis har blitt inkludert på grunn av øvrig syntaktisk likhet. Dette kan en håpe forbedrer seg med en større språkmodell hvor en større andel av embeddingen fanger semtantisk informasjon kontra syntaktisk informasjon.

# %% [markdown]
# forklar kontrastiv læring
