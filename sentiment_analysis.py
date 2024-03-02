import pandas as pd
import numpy as np
import spacy
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df=pd.read_csv(r"c:\Users\yonas\OneDrive\Desktop\projects\task\TASK21N\Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def preprocess_and_sentiment_by_index(df):
    # Check if the DataFrame has at least 6 rows
    if len(df) < 6:
        raise ValueError("DataFrame must have at least 6 rows.")
    
    
    # Select the first 6 rows
    bb = df.head(6).copy()
    
    # Remove rows with missing values in the 'reviews.text' column
    bb.dropna(subset=['reviews.text'], inplace=True)
    
    # Initialize an empty list to store tokenized texts
    tokens = []
    for text in bb['reviews.text']:
        doc = nlp(text)
        tokens.append([token.text for token in doc])
    
    # Apply tokenization to the DataFrame
    outputs = pd.DataFrame({'tokens': tokens, 'reviews.text': bb['reviews.text']})
    
    # Define a function to clean stopwords
    def clean_stopwords(text):
        text = ' '.join(text)
        doc = nlp(text)
        tokens_without_stopwords = [token.text for token in doc if not token.is_stop]
        return tokens_without_stopwords
    
    # Apply stopword removal to the DataFrame
    outputs['cleaned_text'] = outputs['tokens'].apply(clean_stopwords)
    
    # Define a function to extract POS tags
    def extract_pos_tags(text):
        text = ' '.join(text)
        doc = nlp(text)
        pos_tags = [[token.text, token.pos_] for token in doc]
        return pos_tags
    
    # Apply POS tagging to the DataFrame
    outputs['pos_tags'] = outputs['cleaned_text'].apply(extract_pos_tags)
    
    # Define a function for sentiment analysis
    def sentiment_analysis(text):
        text = ' '.join([token for token, pos in text])
        doc = nlp(text)
        blob = TextBlob(doc.text)
        sentiment = blob.sentiment
        polarity = sentiment.polarity
        if polarity > 0:
            sentiment_label = 'Positive'
        elif polarity < 0:
            sentiment_label = 'Negative'
        else:
            sentiment_label = 'Neutral'
        return sentiment_label  
    
    # Apply sentiment analysis to the DataFrame
    outputs['sentiment'] = outputs['pos_tags'].apply(sentiment_analysis)
    
    # Return the DataFrame with sentiment analysis result
    return outputs

# Function to calculate text similarity based on sentiment label
def text_similarity(text1, text2, sentiment_label):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    # Determine similarity threshold based on sentiment label
    if sentiment_label == 'Positive':
        similarity_threshold = 0.7
    elif sentiment_label == 'Negative':
        similarity_threshold = 0.5
    else:
        similarity_threshold = 0.6
    
    if similarity_score > similarity_threshold:
        sim_scor = "similar"
    else:
        sim_scor = "not similar"
    return sim_scor

result = preprocess_and_sentiment_by_index(df)
print(result)

a = int(input("Enter the first number index you want to check the similarity: "))
b = int(input("Enter the second number index you want to check the similarity: "))

x = df['reviews.text'][a]
y = df['reviews.text'][b]
sentiment_label = result['sentiment'][a]  # Sentiment label for text x

similarity_score = text_similarity(x, y, sentiment_label)
print("Cosine similarity between x and y:", similarity_score)
