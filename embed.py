#%% imports
import pandas as pd
from sentence_transformers import SentenceTransformer
df = pd.read_csv('data/Corona_NLP_train.csv', encoding = 'latin-1').head(10)
# %%
tweets = list(df['OriginalTweet'])
# %%
model = SentenceTransformer("all-mpnet-base-v2")
# %%
embeddings = model.encode(tweets, normalize_embeddings= True)
# %%
first10 = pd.DataFrame(data = embeddings)
first10['user'] = list(df['UserName'])
first10['originalTweet'] = list(df['OriginalTweet'])
# %%
