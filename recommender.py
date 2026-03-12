import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data = pd.read_csv("dataset.csv")

# Convert genres to numeric vectors
vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(data["Genre"])

# Compute similarity
similarity = cosine_similarity(genre_matrix)

def show_movies():
    print("\nAvailable Movies:\n")
    for movie in data["Movie"]:
        print("-", movie)

def recommend(movie_name):
    if movie_name not in data["Movie"].values:
        print("\nMovie not found. Please choose from the list.")
        return

    movie_index = data[data["Movie"] == movie_name].index[0]
    scores = list(enumerate(similarity[movie_index]))

    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print("\nRecommended Movies:\n")

    for i in sorted_scores[1:6]:
        print(data.iloc[i[0]]["Movie"])


show_movies()

movie = input("\nEnter a movie you like: ")
recommend(movie)