# MovieLens100k - Based Personalized Movie Recommendation System

This project implements a personalized movie recommendation system using collaborative filtering techniques, primarily leveraging matrix factorization via **Singular Value Decomposition (SVD)** from the [`Surprise`](http://surpriselib.com/) library. 

Built on the **MovieLens dataset**, this system predicts how a user might rate unseen movies based on their past preferences and recommends the top N movies accordingly.

It also includes detailed data preprocessing, model evaluation using cross-validation, and clustering of preferences to help understand and improve recommendations.

The design is modular, with clean separation between data loading, preprocessing, modeling, and evaluation components.

---

## Objectives

- Build a **collaborative filtering-based recommendation engine**
- Evaluate model performance using **RMSE** and **MAE**
- Predict user ratings for unseen movies
- Provide **top-N personalized recommendations**
- Evaluate model performance using **RMSE** and **MAE**
- Use clean code practices with modular structure for scalability

---

## Features

- 📥 Load and preprocess the MovieLens dataset
- ⚙️ Format data for use with the Surprise library
- 📊 Perform 3-fold and 5-fold cross-validation
- 🔍 Display top-rated movies by the user
- 🎯 Generate high-confidence movie recommendations
- 🧠 Modular structure: easy to extend, debug, or scale

---

## Evaluation Metrics

The recommendation engine is evaluated using:

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Squared Error — penalizes large prediction errors more harshly |
| **MAE** | Mean Absolute Error — average absolute difference between predicted and actual ratings |

Both metrics are computed using **k-fold cross-validation** (3-fold and 5-fold).

---

## How to Use
```bash
git clone https://github.com/brendanros31/ML-Recommendation-Engine-MovieLens100k.git

cd ML-Recommendation-Engine-MovieLens100k

pip install -r requirements.txt
jupyter notebook main.ipynb
```

---

## Project Structure
```
data/
    MovieLens100k/
        links.csv
        movies.csv
        ratings.csv
        tags.csv

src/
    data_loader.py
    evaluate.py
    features.py
    model.py

main.ipynb
config/config.yaml
```