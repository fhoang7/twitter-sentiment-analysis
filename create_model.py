#%% imports
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from text_preprocessor import TextPreprocessor
from sklearn.model_selection import train_test_split
os.chdir('C:/Users/frank/OneDrive/Documents/Data Projects/twitter-sentiment-analysis/')
df = pd.read_csv('data/Corona_NLP_train.csv', encoding = 'latin-1').head(10)
#%% Determine Unique Labels
labels = ['Extremely Negative', 'Negative', 'Neutral', 'Positive', 'Extremely Positive']
label_encoding = {label: i for i, label in enumerate(labels)}
df['Sentiment_numeric'] = df['Sentiment'].map(label_encoding)
# %% Pull Tweets for Embeddings
tweets = list(df['OriginalTweet'])
#%% Preprocess pipeline
cleaned_tweets = []
for tweet in tweets:
    tp = TextPreprocessor(text = tweet)
    cleaned_tweet = tp.preprocess()
    cleaned_tweets.append(cleaned_tweet)
# %%
model = SentenceTransformer("all-mpnet-base-v2")
# %%
embeddings = model.encode(tweets, normalize_embeddings= True)
# %%
embedded = pd.DataFrame(data = embeddings)
embedded['user'] = list(df['UserName'])
embedded['originalTweet'] = list(df['OriginalTweet'])
embedded['Sentiment_numeric'] = list(df['Sentiment_numeric'])
# %%
X = embedded.drop(columns = ['user', 'originalTweet', 'Sentiment_numeric'])
y = embedded['Sentiment_numeric']
# %%
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size = 0.2)
