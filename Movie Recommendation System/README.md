# The Screening Room — SVD Movie Recommender (from scratch)

A movie recommendation system built on real Netflix Prize ratings data. The
SVD at the core is implemented by hand — power iteration with deflation —
with no `numpy.linalg.svd` or `scipy.sparse.linalg.svds` anywhere in the code.

## What's in here

| File | What it does |
|---|---|
| `load_and_sample.py` | Loads the raw Kaggle dataset and samples the most active users into a working subset your machine can actually hold in RAM |
| `recommender.py` | The recommender itself — user-item matrix, from-scratch SVD, `recommend_movies()`, exports data for the interface |
| `movie_recommender_interface.html` | Standalone interactive UI — type a viewer ID, get their top 5 picks |
| `sampled_ratings.csv`, `sampled_movies.csv` | A pre-built 1,500-user working sample, checked in so you can run `recommender.py` immediately without re-downloading anything |
| `requirements.txt` | The only two dependencies: `numpy`, `pandas` |

## Quick start

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt

# Option A: just use the sample that's already checked in
python3 recommender.py

# Option B: rebuild the sample yourself (see "Using the full dataset" below)
python3 load_and_sample.py
python3 recommender.py
```

Then open `movie_recommender_interface.html` in any browser — it's fully
self-contained, no server needed.

## Using the full dataset

The raw dataset (`archive.zip`, ~78MB zipped / ~250MB unzipped, "Netflix
Movie Rating Dataset" on Kaggle) is **not included in this repo** — see
"Why the raw data isn't in git" below. To rebuild the sample from scratch:

1. Download `archive.zip` from Kaggle.
2. Place it at `data/archive.zip` in the repo root.
3. Run:
   ```bash
   python3 load_and_sample.py --n-users 1500
   ```
   `--n-users` controls how many of the most active viewers get pulled into
   the dense matrix. Raise it if your machine has the RAM (see table below),
   lower it if `load_and_sample.py` gets killed by the OS.

### How many users can your machine handle?

The dense matrix is `n_users × n_movies (1,350) × 8 bytes`, and building it
via `pivot_table` needs roughly 3–5x that much RAM at peak, plus more during
the SVD step (`AᵀA` is `1,350 × 1,350`, cheap regardless of user count).

| n_users | Matrix size (raw) | Recommended min RAM |
|---|---|---|
| 1,500 (default) | ~16MB | 2GB |
| 4,000 | ~43MB | 4GB |
| 20,000 | ~216MB | 8GB |
| 143,458 (**all** users) | ~1.55GB | **will not work as-is** — see note below |

Loading the *entire* 143,458-user dataset into a dense matrix is not
recommended: it reliably runs out of memory on typical laptops and CI
runners, and it's slow even when it fits. If you actually need that scale,
the real fix is switching `recommender.py` to build `user_item` as a
`scipy.sparse` matrix and rewriting `svd_from_scratch()` to do sparse
matrix-vector products instead of forming a dense `AᵀA` — that's a genuine
rewrite of the SVD step, not a config change.

## Why the raw data isn't in git

Two reasons:
- **Size**: GitHub hard-blocks any single file over 100MB on a normal
  `git push`. `Netflix_Dataset_Rating.csv` alone is ~248MB unzipped.
- **Licensing**: Kaggle datasets generally shouldn't be re-hosted in a public
  repo. Point people at the Kaggle listing instead of committing the CSVs.

`sampled_ratings.csv` (11MB) is small enough and derived, so it's checked in
directly — anyone who forks the repo can run `recommender.py` immediately
without touching Kaggle at all.

## Hardcoded paths — a note if you're editing this yourself

`load_and_sample.py` takes `--zip-path` (default `data/archive.zip`) instead
of a hardcoded absolute path, specifically so it works the same on your
machine, a collaborator's machine, and CI. If you ever hardcode a path like
`/Users/yourname/Downloads/archive.zip` into a script you commit, it will
break for literally everyone else who clones the repo — always use a
relative path (relative to the repo root) or a CLI flag/environment variable
instead.

## Pipeline steps (matches the code's Step 1–11 comments)

1. Load dataset
2. Build user-item matrix (rows = users, columns = movies)
3. Fill missing ratings with 0, convert to NumPy
4. Compute each user's mean rating (over *rated* movies only)
5. Mean-center ("demean") the matrix
6. **Run SVD from scratch** — power iteration + deflation on `AᵀA`
7. Reconstruct the full predicted-rating matrix
8. Fill in **only** the originally-missing cells with predictions — real
   ratings are never overwritten
9. Clip predicted scores to a 1–5 range for display/ranking
10. `recommend_movies(user_id, n=5)` — rank a user's unrated movies by
    predicted score
11. Export recommendations for every user in the sample, which the HTML
    interface reads directly

## Data source

[Netflix Movie Rating Dataset](https://www.kaggle.com/) on Kaggle (derived
from the Netflix Prize dataset). Predicted scores are model output, not
real ratings.
