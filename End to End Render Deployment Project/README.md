# Car Price Prediction — Resale Value Estimator

A complete, self-contained car price prediction website. 

## What it does

You enter three details about a used car:

| Column          | Meaning                                   |
|-----------------|--------------------------------------------|
| `Year`          | Manufacturing year                         |
| `Present_Price` | Original showroom price (in lakhs ₹)       |
| `Kms_Driven`    | Total kilometers driven                    |

...and it predicts `Selling_Price` — the estimated current resale value —
using a `RandomForestRegressor` trained with scikit-learn.

## Project structure

```
car-price-prediction/
├── train.py              # builds the dataset + trains + saves model.pkl
├── app.py                # Flask backend (serves the site + /predict API)
├── model.pkl             # trained model (already included, ready to use)
├── car_data.csv          # the dataset used for training
├── requirements.txt
├── templates/
│   └── index.html        # frontend page
└── static/
    ├── style.css          # dashboard / gauge styling
    └── app.js             # form handling + animated gauge
```

## How to run it on your machine

1. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate      # on Windows: venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Retrain the model**

   A trained `model.pkl` is already included, so you can skip this step.
   If you want to regenerate the dataset/model yourself:

   ```bash
   python train.py
   ```

   This creates `car_data.csv` and `model.pkl` in the project folder.

4. **Run the website**

   ```bash
   python app.py
   ```

5. **Open your browser** and go to:

   ```
   http://127.0.0.1:5000
   ```

   Fill in the year, showroom price, and kilometers driven, then click
   **Estimate Value** — the gauge on the right animates to show the
   predicted resale price, along with depreciation % and retained value %.

## Deploying on Render

The project is already set up for Render — it includes:

- `Procfile` → tells Render to run `gunicorn app:app --bind 0.0.0.0:$PORT`
- `render.yaml` → optional infra-as-code config (auto-detected by Render)
- `gunicorn` in `requirements.txt`
- `app.py` reads the `PORT` env var (Render sets this automatically) and
  binds to `0.0.0.0`, and defaults `debug` to off unless `FLASK_DEBUG=true`

Steps:

1. **Push this project to a GitHub repo.**
   ```bash
   git init
   git add .
   git commit -m "Car price prediction app"
   git branch -M main
   git remote add origin <your-repo-url>
   git push -u origin main
   ```
   Make sure `model.pkl` is committed too (it's small, ~9 MB, and is not
   in `.gitignore`) — Render needs it at runtime since `train.py` is not
   run automatically during deploy.

2. **On Render:** New → Web Service → connect your GitHub repo.

3. Render will read `render.yaml` automatically. If it asks manually instead, set:
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`

4. Click **Create Web Service**. Render installs dependencies, starts
   gunicorn, and gives you a public URL like
   `https://car-price-prediction.onrender.com`.

5. Every future `git push` to `main` auto-redeploys.

**Note on the free plan:** free Render web services spin down after
periods of inactivity, so the first request after idling can take
10–30 seconds to wake up — that's normal, not a bug.

## Notes

- The dataset is generated synthetically inside `train.py` using a
  realistic depreciation formula (price falls with age and kms driven),
  so the whole project works completely offline.
- Swap in your own real-world dataset any time: just make sure it has
  columns named `Year`, `Present_Price`, `Kms_Driven`, `Selling_Price`
  and replace the `build_dataset()` step in `train.py` with
  `pd.read_csv("your_file.csv")`.
- This uses Flask's built-in development server, which is fine for
  local/demo use. For production, run it behind a WSGI server like
  gunicorn.
