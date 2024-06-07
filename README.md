# llm-movie-recommender

Proof of concept, cold start movie recommender. Uses a Language Model to generate embeddings from movie summaries and genres, and find similar movies using a vector database.

To install:

```
conda create --name llm-movie-recommender --file requirements.txt
```

To run:

```
conda activate llm-movie-recommender
python main.py
```

