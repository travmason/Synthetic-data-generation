import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import re
import nltk

import os
import json
import matplotlib.pyplot as plt

# Create empty lists to store the x and y values for each iteration
x_values = []
y_values = []

def process_file(file_path):
    # Process the contents of the file
    with open(file_path) as file:
        df = json.load(file)
        df = pd.DataFrame(df)

        # Merge the 'Prompt', 'Topic' and 'Utterance' columns into a single column
        df['Preamble'] = df['Prompt'] + ' ' + df['Topic'] + ' ' + df['Utterance']

        df['Response_length'] = df['Response'].apply(lambda x: len(str(x)))

        vectorizer = TfidfVectorizer()

        # Define a helper function to compute cosine similarity
        def compute_similarity(text1, text2):
            corpus = [text1, text2]
            vectors = vectorizer.fit_transform(corpus).toarray()
            return cosine_similarity(vectors)[0,1]

        df['CosineSimilarity'] = df.apply(lambda row: compute_similarity(row['Preamble'], row['Response']), axis=1)


        return df


# Directory path containing the JSON files
directory_path = 'gpt3_logs'

# Iterate over each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.json'):
        # Construct the full file path
        file_path = os.path.join(directory_path, filename)
        print(f"Processing {file_path}")
        # Process the file
        df = process_file(file_path)
        
        # Append the index values and cosine similarity values to the lists
        x_values.extend(df.index)
        y_values.extend(df['CosineSimilarity'])

# Visualize the results using a bar plot
plt.bar(x_values, y_values)
plt.xlabel('Index')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarity Results')
plt.show()


def clean_df(df):
    nltk.download('stopwords')
    from nltk.corpus import stopwords

    stop_words = stopwords.words('english')

    # Merge the 'Prompt', 'Topic' and 'Utterance' columns into a single column
    df['Preamble'] = df['Prompt'] + ' ' + df['Topic'] + ' ' + df['Utterance']

    # Function to clean text
    def clean_text(text):
        text = re.sub(r'[^\w\s]','',text)  # remove punctuation
        text = text.lower()  # convert to lower case
        text = ' '.join(word for word in text.split() if word not in stop_words)  # remove stopwords
        return text

    df['clean_text'] = df['text'].apply(clean_text)