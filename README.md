# twitter-sentiment-analysis
Sentiment Analysis by classifying different tweets from this labeled dataset. Experimenting with base embeddings vs. Matryoshka truncated embeddings. 
- Data: https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification?select=Corona_NLP_test.csv
- Matryoshka Embeddings: https://huggingface.co/blog/matryoshka#in-sentence-transformers-1

# Primary Goals: 
I will do sentiment analysis and quantify performance of untruncated MPNET text embeddings. I will then baseline that against different truncated embeddings using Matryoshka embedding models to see what the tradeoffs in performance vs speed would be. 

# Secondary Goals:
- Embedding clustering (speed vs. performance)
- Storing vectors in Milvus

