import argparse
import pandas as pd
import numpy as np
import itertools
import sys
import threading
import time
from termcolor import colored
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set the dataset path
DATASET_PATH = "data/wiki_movie_plots_deduped.csv"

# Simple Loading Animation
def loading_animation(stop_event):
    for symbol in itertools.cycle(['|', '/', '-', '\\']):
        if stop_event.is_set():
            break
        sys.stdout.write(f'\r⏳ Finding the best matches {symbol}')
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\r' + ' ' * 50 + '\r')  # Clear the line when done

# Load Movie Dataset
def load_data():
    try:
        df = pd.read_csv(DATASET_PATH)
        df.columns = df.columns.str.strip()  # Remove extra spaces in column names
        return df
    except Exception as e:
        print(colored(f"❌ Error loading dataset: {e}", "red"))
        sys.exit(1)

# Convert Text into TF-IDF Vectors (Optimized)
def preprocess_text(df, text_column):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=20000)  # Limit to 20k features for speed
    tfidf_matrix = vectorizer.fit_transform(df[text_column])  # Keeps it sparse (memory efficient)
    return vectorizer, tfidf_matrix

# Find the Most Similar Movies
def get_recommendations(query, vectorizer, tfidf_matrix, df, top_n=5):
    query_vec = vectorizer.transform([query])  # Convert user input to a vector
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    if np.all(similarity_scores == 0):
        return None, None  # No relevant match found

    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    return df.iloc[top_indices][['Title', 'Plot']], similarity_scores[top_indices]

# Display Recommendations in a Friendly Format
def display_recommendations(recommendations, scores):
    print("\n🎬 " + colored("Top Movie Recommendations:", "cyan", attrs=["bold"]) + "\n")
    for idx, (row, score) in enumerate(zip(recommendations.iterrows(), scores), start=1):
        title = colored(f"{idx}. {row[1]['Title']}", "yellow", attrs=["bold"])
        plot = row[1]['Plot'][:250] + ("..." if len(row[1]['Plot']) > 250 else "")
        similarity = colored(f"({score:.2%} match)", "green")
        print(f"🎥 {title} {similarity}\n   📝 {plot}\n" + "-" * 80)

# Main Function (CLI Handling)
def main():
    parser = argparse.ArgumentParser(description="🎥 Simple Movie Recommendation System")
    parser.add_argument("query", nargs="?", type=str, help="Describe the kind of movie you're looking for")
    parser.add_argument("--top_n", type=int, default=5, help="Number of recommendations to return")
    args = parser.parse_args()

    if not args.query:
        args.query = input("🔍 Tell me what kind of movie you're in the mood for: ")

    # Start the loading animation in a separate thread
    stop_event = threading.Event()
    loading_thread = threading.Thread(target=loading_animation, args=(stop_event,))
    loading_thread.start()

    # Process the dataset
    df = load_data()
    vectorizer, tfidf_matrix = preprocess_text(df, 'Plot')
    recommendations, scores = get_recommendations(args.query, vectorizer, tfidf_matrix, df, args.top_n)

    stop_event.set()  # Stop the loading animation
    loading_thread.join()

    # Show results
    if recommendations is None:
        print("\n❌ Sorry, I couldn't find any good matches. Try a different description!")
    else:
        display_recommendations(recommendations, scores)

# Run the script
if __name__ == "__main__":
    main()
