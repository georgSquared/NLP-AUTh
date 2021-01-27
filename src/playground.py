"""
Playground to test stuff, and leave irrelevant code
"""
import pandas as pd

test_df = pd.read_csv('../input/test.csv')

from nltk import word_tokenize
test_df['processed'] = test_df['question_text'].apply(lambda row: word_tokenize(row))
test_df['processed'] = test_df['processed'].apply(lambda row: [token for token in row if token.isalpha()])

from nltk.corpus import stopwords
eng_stopwords = stopwords.words('english')
test_df['processed'] = test_df['processed'].apply(lambda row: [token for token in row if token not in eng_stopwords])

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
test_df['stemmed'] = test_df['processed'].apply(lambda row: [stemmer.stem(token) for token in row])

from wordcloud import WordCloud, STOPWORDS

test_df['wordcloud'] = test_df['processed'].apply(lambda row: [token for token in row if token not in STOPWORDS])
test_df['wordcloud'] = test_df['processed'].apply(lambda row: ' '.join(map(str, row)))

from textblob import TextBlob

test_df["polarity"] = test_df['wordcloud'].apply(lambda x: TextBlob(x).sentiment.polarity)
test_df["subjectivity"] = test_df['wordcloud'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

import textstat

test_df['readability'] = test_df['question_text'].apply(lambda x: textstat.text_standard(x, float_output=True))
test_df.to_csv("test_exotic_features.csv")

# #%% md
#
# ## Topic Modeling (LDA)
# https://blog.bitext.com/lemmatization-to-enhance-topic-modeling-results
#
# (Why we lemmatize)
#
# Dumb lematization everything as nouns
#
#
#
#
# #%%
#
# from nltk.stem import WordNetLemmatizer
#
# lemmatizer = WordNetLemmatizer()
# train_df['corpus'] = train_df['processed'].apply(lambda row: [lemmatizer.lemmatize(token) for token in row])
# train_df.head()
#
# #%%
#
# from gensim import corpora
# sincere_dict = corpora.Dictionary(train_df[train_df.target == 0]['corpus'])
# sincere_dict.save('../input/sincere.dict')
# # print(sincere_dict)
#
# #%% md
#
# ## Topic Modeling (LSA)
# We will attempt some basic topic modeling by performing some basic Latent Semantic Analysis (LSA) on our data. To
# achieve this we will create some basic TF-IDF features and then reduce them using SVD
#
# We will perform this for both our sincere and insincere data
#
# #%%
#
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# from sklearn.pipeline import Pipeline
#
# #%%
#
# train_df['clean_text'] = train_df['stemmed'].apply(lambda row: ' '.join(map(str, row)))
#
# vectorizer = TfidfVectorizer(ngram_range=(1,1))
# svd = TruncatedSVD(n_components=4, random_state=0)
# pipe = Pipeline([
#     ('tf-idf', vectorizer),
#     ('svd', svd)
# ])
#
# #%% md
#
# First identify topics for the sincere subset of our data
#
# #%%
#
# pipe.fit_transform(train_df[train_df.target == 0]['clean_text'])
# sincere_topics = pd.DataFrame(svd.components_)
# sincere_topics.columns = pipe['tf-idf'].get_feature_names()
# sincere_topics.head()
#
# #%% md
#
# Repeat the process for the insincere subset of the data
#
# #%%
#
# pipe.fit_transform(train_df[train_df.target == 1]['clean_text'])
# insincere_topics = pd.DataFrame(svd.components_)
# insincere_topics.columns = pipe['tf-idf'].get_feature_names()
# insincere_topics.head()
#
# # PLOT HERE
