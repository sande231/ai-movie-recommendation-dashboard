from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os


API_KEY = "ae9a2b98"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)

data = pd.read_csv("dataset.csv")

vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(data["Genre"])
similarity = cosine_similarity(genre_matrix)


def recommend(movie_name):
    if movie_name not in data["Movie"].values:
        return []

    movie_index = data[data["Movie"] == movie_name].index[0]
    scores = list(enumerate(similarity[movie_index]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommended = []
    for i in sorted_scores[1:6]:
        index = i[0]
        score = round(i[1] * 100, 2)

        recommended.append({
            "movie": data.iloc[index]["Movie"],
            "genre": data.iloc[index]["Genre"],
            "score": score,
            "poster": data.iloc[index]["Poster"]
        })

    return recommended


def recommend_by_genre(genres, selected_movie):
    recommendations = []

    for _, row in data.iterrows():
        if row["Genre"] in genres and row["Movie"].lower() != selected_movie.lower():
            score = 95 - len(recommendations) * 5

            recommendations.append({
                "movie": row["Movie"],
                "genre": row["Genre"],
                "score": score,
                "poster": row["Poster"]
            })

    return recommendations[:5]


def search_movie(movie_name):
    url = f"http://www.omdbapi.com/?t={movie_name}&apikey={API_KEY}"
    response = requests.get(url)
    movie_data = response.json()

    if movie_data["Response"] == "True":
        genres = [g.strip() for g in movie_data["Genre"].split(",")]

        return {
            "title": movie_data["Title"],
            "genres": genres,
            "poster": movie_data["Poster"],
            "imdb_rating": movie_data.get("imdbRating", "N/A"),
            "year": movie_data.get("Year", "N/A"),
            "director": movie_data.get("Director", "N/A"),
            "plot": movie_data.get("Plot", "N/A")
        }

    return None


def create_chart():
    genre_counts = data["Genre"].value_counts()

    plt.figure()
    genre_counts.plot(kind="bar")
    plt.title("Movies per Genre")
    plt.xlabel("Genre")
    plt.ylabel("Count")
    plt.tight_layout()

    plt.savefig("static/genre_chart.png")
    plt.close()


@app.route("/", methods=["GET", "POST"])
def home():
    create_chart()

    recommendations = []
    selected_movie = None
    selected_genre = None
    selected_rating = None
    searched_poster = None
    error_message = None
    selected_year = None
    selected_director = None
    selected_plot = None
    top_action = data[data["Genre"] == "Action"]["Movie"].head(3).tolist()

    stats = {
        "total_movies": len(data),
        "total_genres": data["Genre"].nunique()
    }

    if request.method == "POST":
        movie_name = request.form["movie"]

        movie_data = search_movie(movie_name)

        if movie_data:
            selected_movie = movie_data["title"]
            selected_genre = ", ".join(movie_data["genres"])
            searched_poster = movie_data["poster"]
            selected_rating = movie_data["imdb_rating"]
            selected_year = movie_data["year"]
            selected_director = movie_data["director"]
            selected_plot = movie_data["plot"]

            recommendations = recommend_by_genre(movie_data["genres"], selected_movie)
        else:
            error_message = "Movie not found. Please try another title."

    return render_template(
        "index.html",
        recommendations=recommendations,
        stats=stats,
        selected_movie=selected_movie,
        selected_genre=selected_genre,
        selected_rating=selected_rating,
        top_action=top_action,
        searched_poster=searched_poster,
        selected_year=selected_year,
        selected_director=selected_director,
        selected_plot=selected_plot,
        error_message=error_message
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)