"""
================================================================================
 MOVIE RECOMMENDATION SYSTEM -- SVD FROM SCRATCH  (real Netflix dataset)
================================================================================
Input: sampled_ratings.csv / sampled_movies.csv, produced by load_and_sample.py
       from the user-supplied archive.zip (Kaggle Netflix ratings dataset).

Same step numbering as the reference notes (Step 1 ... Step 11). SVD is
implemented by hand with power iteration + deflation -- no np.linalg.svd,
no scipy.sparse.linalg.svds anywhere in this file.
"""

import numpy as np
import pandas as pd
import json

pd.set_option("display.width", 120)

# -----------------------------------------------------------------------------
# Step 1: Load dataset
# -----------------------------------------------------------------------------
ratings = pd.read_csv("sampled_ratings.csv")      # user_id, rating, item_id
movies = pd.read_csv("sampled_movies.csv")         # item_id, year, title

print("Step 1: Loaded dataset")
print(ratings.head())
print(movies.head())
print()

# -----------------------------------------------------------------------------
# Step 2: Create the User-Item matrix
#   rows    = users
#   columns = movies
#   0       = the user has NOT rated that movie (missing value)
# -----------------------------------------------------------------------------
user_item = ratings.pivot_table(
    index="user_id",
    columns="item_id",
    values="rating",
    aggfunc="mean"          # a handful of users rate the same movie twice; average it
).fillna(0)

print("Step 2: User-Item matrix built ->", user_item.shape)
print(user_item.iloc[:5, :5])
print()

# -----------------------------------------------------------------------------
# Step 3: Convert to a NumPy matrix, and remember WHICH entries were originally
#          missing (rating == 0) so we never overwrite real ratings later.
# -----------------------------------------------------------------------------
R = user_item.values.astype(float)                    # shape (n_users, n_movies)
was_missing = (R == 0)                                  # boolean mask of unrated cells
print("Step 3: Converted to NumPy matrix, shape:", R.shape)
print()

# -----------------------------------------------------------------------------
# Step 4: Compute each user's average rating -- using ONLY the movies they
#          actually rated (zeros must not drag the mean down).
# -----------------------------------------------------------------------------
n_rated_per_user = (R != 0).sum(axis=1)
n_rated_per_user[n_rated_per_user == 0] = 1              # avoid /0 for edge case
user_mean = R.sum(axis=1) / n_rated_per_user

print("Step 4: Sample user means:", np.round(user_mean[:5], 2))
print()

# -----------------------------------------------------------------------------
# Step 5: Mean-center (demean) the matrix so SVD focuses on relative taste,
#          not on how generous/harsh a user is.
# -----------------------------------------------------------------------------
R_demeaned = R - user_mean.reshape(-1, 1)
print("Step 5: Ratings demeaned")
print()

# -----------------------------------------------------------------------------
# Step 6: SVD IMPLEMENTED FROM SCRATCH
#          (power iteration + deflation -- no np.linalg.svd / scipy.svds)
#
#   For an m x n matrix A, we find the top-k singular triplets (u, sigma, v) by
#   repeatedly finding the dominant eigenvector of the small n x n matrix
#   B = A^T A via power iteration, then deflating B to remove that
#   component before finding the next one.
# -----------------------------------------------------------------------------
def svd_from_scratch(A, k, n_iter=300, tol=1e-9, seed=0):
    """
    Truncated SVD via power iteration + Hotelling deflation.
    Returns U (m x k), sigma (k,), VT (k x n) such that A ~= U @ diag(sigma) @ VT
    """
    rng = np.random.default_rng(seed)
    m, n = A.shape
    B = A.T @ A                      # n x n, symmetric positive semi-definite

    V_list = []
    sigma_list = []

    for comp in range(k):
        v = rng.normal(size=n)
        v /= np.linalg.norm(v)

        prev_eigval = 0.0
        eigval = 0.0
        for _ in range(n_iter):
            v_new = B @ v
            norm = np.linalg.norm(v_new)
            if norm < 1e-12:
                break
            v_new /= norm

            eigval = float(v_new @ B @ v_new)          # Rayleigh quotient
            if abs(eigval - prev_eigval) < tol:
                v = v_new
                break
            v = v_new
            prev_eigval = eigval

        eigval = max(eigval, 0.0)                        # numerical safety
        sigma = np.sqrt(eigval)

        V_list.append(v)
        sigma_list.append(sigma)

        # Deflation: remove this component from B so the next power
        # iteration converges to the NEXT largest eigenvector.
        B = B - eigval * np.outer(v, v)

        if (comp + 1) % 10 == 0:
            print(f"   ...component {comp + 1}/{k} done (sigma={sigma:.3f})")

    V = np.array(V_list).T                 # n x k
    sigma = np.array(sigma_list)           # k,

    # Recover U from A, V, sigma :  A v_i = sigma_i * u_i
    U = np.zeros((m, k))
    for i in range(k):
        if sigma[i] > 1e-10:
            U[:, i] = (A @ V[:, i]) / sigma[i]
        else:
            U[:, i] = 0.0

    return U, sigma, V.T                    # U, sigma, VT


K_LATENT = 50   # number of latent factors, same as the reference notes (k=50)

print(f"Step 6: Running from-scratch SVD with k={K_LATENT} latent factors ...")
U, sigma, VT = svd_from_scratch(R_demeaned, k=K_LATENT)
print("U shape:", U.shape, " sigma shape:", sigma.shape, " VT shape:", VT.shape)
print()

# -----------------------------------------------------------------------------
# Step 7: Reconstruct the full (dense) predicted-rating matrix
# -----------------------------------------------------------------------------
predicted = U @ np.diag(sigma) @ VT
predicted += user_mean.reshape(-1, 1)          # add the per-user mean back

predicted_df = pd.DataFrame(
    predicted,
    columns=user_item.columns,
    index=user_item.index
)
print("Step 7: Reconstructed predicted-rating matrix")
print(predicted_df.iloc[:5, :5].round(2))
print()

# -----------------------------------------------------------------------------
# Step 8: IMPORTANT -- only fill in the entries that were originally missing.
#          Every rating the user actually gave is kept EXACTLY as-is; we do
#          NOT overwrite real ratings with the model's prediction.
# -----------------------------------------------------------------------------
final_ratings = np.where(was_missing, predicted, R)
final_df = pd.DataFrame(final_ratings, columns=user_item.columns, index=user_item.index)

print("Step 8: Final matrix = original ratings kept, missing cells filled with predictions")
print(final_df.iloc[:5, :5].round(2))
print()

# -----------------------------------------------------------------------------
# Step 9: Clip the predicted scores used for RANKING/DISPLAY to the valid
#          1-5 star range (does not touch stored real ratings from Step 8).
# -----------------------------------------------------------------------------
predicted_display = predicted_df.clip(lower=1, upper=5)

# -----------------------------------------------------------------------------
# Step 10: recommend_movies() -- ranks the movies a user has NOT already
#           rated by their PREDICTED rating and returns the top n.
# -----------------------------------------------------------------------------
movies_by_id = movies.set_index("item_id")

def recommend_movies(user_id, n=5):
    if user_id not in user_item.index:
        return pd.DataFrame(columns=["item_id", "title", "year", "predicted_rating"])

    user_row_missing = was_missing[user_item.index.get_loc(user_id)]
    candidate_items = user_item.columns[user_row_missing]         # unrated movies only

    scores = predicted_display.loc[user_id, candidate_items].sort_values(ascending=False)
    top = scores.head(n)

    result = pd.DataFrame({
        "item_id": top.index,
        "predicted_rating": top.values.round(2)
    }).merge(movies, on="item_id", how="left")
    return result[["item_id", "title", "year", "predicted_rating"]]


# -----------------------------------------------------------------------------
# Step 11: Sample output
# -----------------------------------------------------------------------------
sample_user = int(user_item.index[0])
recommendations = recommend_movies(sample_user, n=5)
print(f"Step 11: Top-5 recommendations for user {sample_user}")
print(recommendations.to_string(index=False))
print()

# -----------------------------------------------------------------------------
# Export precomputed recommendations for ALL sampled users -> powers the
# interface (movie_recommender_interface.html)
# -----------------------------------------------------------------------------
all_recs = {}
for uid in user_item.index:
    uid_int = int(uid)
    recs = recommend_movies(uid_int, n=5)

    user_actual = ratings[ratings["user_id"] == uid_int].sort_values("rating", ascending=False)
    liked = []
    for i in user_actual.head(3)["item_id"]:
        if i in movies_by_id.index:
            row = movies_by_id.loc[i]
            liked.append(f"{row['title']} ({int(row['year'])})")

    all_recs[str(uid_int)] = {
        "already_rated": int((~was_missing[user_item.index.get_loc(uid_int)]).sum()),
        "liked_examples": liked,
        "recommendations": [
            {"movie": f"{row['title']} ({int(row['year'])})" if not pd.isna(row['year']) else row['title'],
             "rating": round(float(row["predicted_rating"]), 2)}
            for _, row in recs.iterrows()
        ]
    }

with open("recommendations.json", "w") as f:
    json.dump(all_recs, f, indent=2)

print("Exported recommendations.json for", len(all_recs), "users")
