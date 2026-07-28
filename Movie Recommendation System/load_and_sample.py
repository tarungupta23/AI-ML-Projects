"""
================================================================================
 STEP 0 -- LOAD REAL DATASET & BUILD A WORKING SAMPLE
================================================================================
Source: archive.zip (Kaggle "Netflix Movie Rating Dataset")
  - Netflix_Dataset_Movie.csv   : Movie_ID, Year, Name            (17,770 movies)
  - Netflix_Dataset_Rating.csv  : User_ID, Rating, Movie_ID       (17.3M ratings,
                                                                    143,458 users)

The full user-item matrix (143,458 x 1,350) is too large to build a dense
matrix and run a from-scratch power-iteration SVD on in this environment, so
we take the same approach any real recommender-systems pipeline takes before
prototyping: sample the most active users (people with enough ratings for the
model to actually learn their taste), keep every movie that appears in the
ratings file, and build the dense matrix from THAT.

Everything downstream (recommender.py) works on real Netflix ratings and real
movie titles -- nothing here is synthetic.
"""
import zipfile
import io
import os
import sys
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Sample the Netflix ratings dataset into a working subset.")
parser.add_argument(
    "--zip-path", default="data/archive.zip",
    help="Path to the Kaggle archive.zip (default: data/archive.zip, relative to repo root)"
)
parser.add_argument(
    "--n-users", type=int, default=1500,
    help="Number of most-active users to keep (default: 1500). Raise this if your machine has more RAM."
)
args = parser.parse_args()

ZIP_PATH = args.zip_path
N_TOP_USERS = args.n_users

if not os.path.exists(ZIP_PATH):
    sys.exit(
        f"\nCouldn't find '{ZIP_PATH}'.\n"
        "Download 'Netflix Movie Rating Dataset' from Kaggle and place archive.zip at that path\n"
        "(or point at it with --zip-path /your/path/archive.zip).\n"
    )

# -----------------------------------------------------------------------------
# Load straight out of the zip (no need to unzip 250MB to disk twice)
# -----------------------------------------------------------------------------
with zipfile.ZipFile(ZIP_PATH) as z:
    with z.open("Netflix_Dataset_Movie.csv") as f:
        movies = pd.read_csv(f)
    with z.open("Netflix_Dataset_Rating.csv") as f:
        ratings = pd.read_csv(f)

movies = movies.rename(columns={"Movie_ID": "item_id", "Name": "title", "Year": "year"})
ratings = ratings.rename(columns={"User_ID": "user_id", "Movie_ID": "item_id", "Rating": "rating"})

print("Raw dataset:")
print(" ratings:", ratings.shape, " unique users:", ratings.user_id.nunique(),
      " unique movies:", ratings.item_id.nunique())

# -----------------------------------------------------------------------------
# Keep the N_TOP_USERS most active users (most ratings given) -- this keeps
# real Netflix data, real preferences, just a computationally sane slice.
# -----------------------------------------------------------------------------
top_users = (
    ratings.groupby("user_id").size()
    .sort_values(ascending=False)
    .head(N_TOP_USERS)
    .index
)
sample = ratings[ratings["user_id"].isin(top_users)].copy()

# keep only movies that still have metadata and appear in the sample
sample = sample.merge(movies[["item_id"]], on="item_id", how="inner")
used_movies = movies[movies["item_id"].isin(sample["item_id"].unique())].copy()

print("\nSampled working set:")
print(" ratings:", sample.shape, " users:", sample.user_id.nunique(),
      " movies:", sample.item_id.nunique())
print(" density: {:.2f}%".format(
    100 * len(sample) / (sample.user_id.nunique() * sample.item_id.nunique())
))

sample.to_csv("sampled_ratings.csv", index=False)
used_movies.to_csv("sampled_movies.csv", index=False)
print("\nSaved sampled_ratings.csv and sampled_movies.csv")
