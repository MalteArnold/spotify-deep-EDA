import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

# Set the file path
path = "Spotify_Songs_Dataset_exported.csv"

# Load the Dataset into a DataFrame
df = pd.read_csv(path)

# Understanding the Data
# Show basic informations and statistics of the data
df.head()
df.describe()
df.info()
df.nunique()
df.isnull().sum()

# Display the artists with the most songs
artists_most_songs = df["artist"].value_counts().head(15)
plt.figure(figsize=(16, 8))
sns.barplot(x=artists_most_songs.index, y=artists_most_songs.values, palette="viridis")
plt.xticks(rotation=45, ha="right")
plt.xlabel("Artist")
plt.ylabel("Number of Songs")
plt.title("Artists with the most songs")
plt.show()

# Display the artists with the fewest songs
artists_fewest_songs = df["artist"].value_counts().tail(15)
plt.figure(figsize=(16, 8))
sns.barplot(
    x=artists_fewest_songs.index, y=artists_fewest_songs.values, palette="viridis"
)
plt.xticks(rotation=45, ha="right")
plt.xlabel("Artist")
plt.ylabel("Number of Songs")
plt.title("Artists with the fewest songs")
plt.show()

# Distribution of word count in texts
df["Text Count"] = df["text"].apply(lambda x: len(x.split()))
sns.histplot(df["Text Count"], bins=20, kde=True)
plt.title("Distribution of Word Count in Texts")
plt.xlabel("Number of Words in Text")
plt.ylabel("Frequency")
plt.show()

# Display a wordcloud of the lyrics
from wordcloud import WordCloud

text = " ".join(df["text"])
wordcloud = WordCloud(width=1000, height=500, background_color="white").generate(text)

plt.figure(figsize=(16, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of the Lyrics")
plt.show()

# Display the most frequent words
from sklearn.feature_extraction.text import CountVectorizer

corpus = df["text"].astype(str).tolist()
vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(corpus)
word_frequencies = X.sum(axis=0)
word_frequencies = [
    (word, word_frequencies[0, idx]) for word, idx in vectorizer.vocabulary_.items()
]
word_frequencies.sort(key=lambda x: x[1], reverse=True)
top_10_words = word_frequencies[:10]
top_10_words_df = pd.DataFrame(top_10_words, columns=["Word", "Count"])

plt.figure(figsize=(10, 6))
sns.barplot(x="Word", y="Count", data=top_10_words_df, palette="viridis")
plt.title("Top 10 Words")
plt.xlabel("Word")
plt.ylabel("Count")
plt.show()
