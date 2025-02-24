#Movie Recomendation System
---
This Movie Recommendation System uses TF-IDF and cosine similarity to find movies that match a given text description. By analyzing movie plots, it suggests the top N most similar films based on user input. It's fast, efficient, and easy to use via the command line. The program returns a certain number of movies that match the given text description along with the similarity score.

#Dataset
---
The dataset, "Wikipedia Movie Plots Deduped," contains over 34,000 movies from various origins, including Hollywood, Bollywood, and other global film industries. Each entry includes details like title, release year, genre, director, cast, and a plot summary. The plot summaries are used to find movie recommendations based on textual similarity.

#Installation
---
Make sure Python 3.8+ is installed. Install required libraries by using the following command
`pip install requirements.txt`

#Usage
---
`pip install rec.py "enter your decription here` --top_n "number of movies"
Example usage:
`python rec.py "I like romantic comedies" --top_n 3`

Output:
---
![image](https://github.com/user-attachments/assets/d01b1e04-4ce1-4dd1-b425-f89e14e0c493)






