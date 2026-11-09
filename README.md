# Hybrid movie recommender

A rating-prediction model for 162,000 users and 62,000 films, built for the ALX Data Science movie recommendation hackathon (February 2026). It predicts how a user would rate a film they haven't seen, using collaborative filtering where there is rating history and film content where there isn't.

**Skills:** Python, pandas, NumPy, SciPy sparse matrices, scikit-learn (TruncatedSVD, StandardScaler), recommender systems, feature engineering

## The data

| | |
|---|---|
| Training ratings | 10,000,038 |
| Ratings to predict | 5,000,019 |
| Users | 162,541 |
| Films with tag-genome features | 13,816, each scored on 1,128 tags |
| Films in the test set with no training ratings | 5,044 |

The last row is the hard part. Those 5,044 films had never been rated in the training data, so a pure collaborative model has nothing to work with for them (the cold-start problem).

## The approach

**1. Content features.** I one-hot encoded the 20 genres and joined them to the tag genome, which scores every film on 1,128 descriptive tags such as "dark", "based on a book" or "pixar animation". After scaling, cosine similarity on these 1,148 features finds films that feel alike. Asked for films similar to *Toy Story*, it returns *A Bug's Life*, *Toy Story 2*, *Monsters, Inc.*, *Finding Nemo*, *The Incredibles* and *Ratatouille*.

**2. Bias-corrected SVD.** Some users rate everything high and others rate everything low. I subtracted each user's mean rating before factorising the sparse user-film rating matrix with TruncatedSVD (50 latent factors), then added the user's mean back at prediction time. A sparse matrix kept the 10 million ratings in memory.

**3. Hybrid prediction.**
- Known user and known film: SVD score plus the user's mean rating.
- Film with no ratings: the average rating for its main genre.
- Anything else: the global mean.
- All predictions are clipped to the 0.5–5 rating scale.

## Run it

The competition data (MovieLens-derived files from GroupLens) isn't redistributed here. Download it from the hackathon page, put the files in a folder, and point `DATA_DIR` in the first cell at that folder. The notebook was written for Kaggle, where it finds the data under `/kaggle/input` automatically.

```bash
pip install -r requirements.txt
jupyter notebook notebooks/movie_recommender.ipynb
```

---

Prince Adufah · [adufah.com](https://adufah.com) · [LinkedIn](https://www.linkedin.com/in/princeadufah)
