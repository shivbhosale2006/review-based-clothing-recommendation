#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 2&3
# #### Student Name: Shiv Mahesh Bhosale
# #### Student ID: S4192953
# 
# 
# Environment: Python 397 and Jupyter notebook
# 
# Libraries used: 
# - pandas
# - numpy
# - regex
# - sklearn
# - gensim
# 
# ## Introduction
# In this task, multiple feature feature representations are generated from the processed clothing reviews.
# 
# The representations include:
# - Bag-of-words
# - Unweighted word embedding vectors
# - TF-IDF weighted embedding vectors
# 
# These representations will later be used for classification experiments.

# ## Importing libraries 

# In[1]:


import pandas as pd
import numpy as np
import regex as re
from collections import Counter
from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score
import gensim
import gensim.downloader as api
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import cross_val_score


# The commands which are commented below are used to install gensim and solve the errors while running the notebook in py397 version.

# In[2]:


# !pip install pandas numpy scikit-learn gensim nltk


# In[3]:


# import sys

# print(sys.executable)
# print(sys.version)


# In[4]:


# import sys
# !{sys.executable} -m pip install pandas numpy scikit-learn gensim nltk


# ## Task 2. Generating Feature Representations for Clothing Items Reviews

# In this task, multiple feature representations are generated from the processed clothing reviews.
# 
# The representations include:
# - Bag-of-Words count vectors
# - Unweighted word embedding vectors
# - TF-IDF weighted embedding vectors

# Loading the 'vocab.txt' file and creating a vocab for further tasks.

# In[5]:


def load_vocab(filepath='vocab'):
    vocab = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                word, idx = line.rsplit(':', 1)
                vocab[word] = int(idx)
    return vocab


# In[6]:


vocab = load_vocab('vocab.txt')
index_to_word = {idx: word for word, idx in vocab.items()}


# In[7]:


print(f"Vocabulary size: {len(vocab)}")
print(f"\nFirst 10 vocab entries:")
for word, idx in list(vocab.items())[:10]:
    print(f"  {word}:{idx}")


# Loading the 'processed.csv' file and creating a dataframe df.

# In[8]:


df = pd.read_csv('./processed.csv')


# In[9]:


df


# In[10]:


df["Review Text"].isnull().sum()


# In[11]:


df['Review Text'] = df['Review Text'].fillna('')


# In[12]:


tk_reviews = [review.split(' ') for review in df['Review Text'].tolist()]


# In[13]:


tk_reviews[1]


# In[14]:


joined_reviews = [' '.join(tokens) for tokens in tk_reviews]


# In[15]:


joined_reviews[1]


# In[16]:


len(tk_reviews)


# ### Bag-of-words Count Vector Representation
# 
# A bag-of-word count vector is created for each review.
# 

# In[17]:


cVectorizer = CountVectorizer(analyzer="word", vocabulary=vocab)
count_features = cVectorizer.fit_transform(joined_reviews)


# In[18]:


count_features.shape


# ### Saving Count Vector Outputs
# 
# The generated sparse count vectors are saved into `count_vectors.txt` using the required assignment format.

# In[19]:


out_file = open('count_vectors.txt', 'w')


# In[20]:


for doc_id in range(count_features.shape[0]):
    row = count_features[doc_id].tocoo()
    pairs = ','.join(
        f"{col}:{val}"
        for col, val in sorted(zip(row.col, row.data))
    )
    out_file.write(f"#{doc_id},{pairs}\n")
out_file.close()


# In[21]:


with open('count_vectors.txt') as f:
    lines = f.read().splitlines()


# In[22]:


len(lines)


# In[23]:


lines[0]


# ### TF-IDF Vectors

# In[24]:


tVectorizer = TfidfVectorizer(analyzer="word", vocabulary=vocab)
tfidf_features = tVectorizer.fit_transform(joined_reviews)


# In[25]:


tfidf_features.shape


# In[26]:


sample = tfidf_features[0].tocoo()
top = sorted(zip(sample.col, sample.data), key=lambda x: -x[1])[:8]

print("Top TF-IDF words in review 0:")
for index, score in top:
    print(f"  '{index_to_word[int(index)]}': {score:.4f}")


# ### Word Embedding Model
# 
# GloVe word embeddings are used to generate dense semantic vector representations for reviews.

# In[27]:


embedding_model = api.load('glove-wiki-gigaword-100')


# In[28]:


embedding_dim = embedding_model.vector_size
print('Embedding dimension:', embedding_dim)


# In[29]:


def getDocVec_unweighted(review, model, vec_size):
    vectors = []
    for w in review:
        if w in model:
            vectors.append(model[w])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vec_size)


# In[30]:


unweighted_doc_vectors = np.array(
    [getDocVec_unweighted(review, embedding_model, embedding_dim) 
     for review in tk_reviews]
)
unweighted_doc_vectors.shape


# In[31]:


unweighted_doc_vectors[0]


# ### TF-IDF Weighted Embeddings

# In[32]:


tfidf_vectorizer = TfidfVectorizer(
    analyzer="word",
    vocabulary=vocab
)

tfidf_features = tfidf_vectorizer.fit_transform(joined_reviews)

print("TF-IDF matrix shape:", tfidf_features.shape)


# In[33]:


def getDocVec_weighted(row_index, review, model, tfidf_matrix, vocab, embedding_dim):

    words = str(review).split()

    weighted_vector = np.zeros(embedding_dim)
    total_weight    = 0

    for word in words:
        if word in model and word in vocab:
            vocab_index = vocab[word]                           # get word index from vocab dict
            weight      = tfidf_matrix[row_index, vocab_index] # TF-IDF score for this word in this doc

            weighted_vector += model[word] * weight
            total_weight    += weight

    if total_weight == 0:
        return np.zeros(embedding_dim)

    return weighted_vector / total_weight


# In[34]:


weighted_doc_vectors = np.array([
    getDocVec_weighted(i, review, embedding_model, tfidf_features, vocab, embedding_dim)
    for i, review in enumerate(df['Review Text'])
])

print("Weighted vectors shape:", weighted_doc_vectors.shape)


# In[35]:


weighted_doc_vectors[0]


# In[36]:


print("Unweighted (review 0):", unweighted_doc_vectors[0][:8])
print("Weighted   (review 0):", weighted_doc_vectors[0][:8])


# ### Saving Embedding Representations
# The generated embedding vectors are saved for future reuse and experimentation.

# In[37]:


np.savetxt("unweighted_vectors.txt", unweighted_doc_vectors)
np.savetxt("weighted_vectors.txt", weighted_doc_vectors)


# ## Task 3. Clothing Review Classification

# In this task, Logistic Regression is used as the selected machine learning model to classify whether a clothing item review is recommended or not. Two experiments are conducted. The first experiment compares the feature representations generated in Task 2. The second experiment checks whether using extra text information, such as review title, improves classification accuracy. A 5-fold cross validation is used for evaluation.

# ### preparing target variable
# The target variable for classification is the Recommended IND column. This column shows whether the clothing item was recommended (1) or not (0).

# df.columns.tolist()

# In[38]:


y = df["Recommended IND"]


# The target variable is checked before modelling because the classes may not be balanced. If one class appears much more than the other, accuracy alone may not fully explain the model performance. Therefore, precision, recall, and F1-score are also used later with accuracy.

# In[39]:


print(y.value_counts())


# ### Machine learning model
# 
# Logistic Regression is used as the classification model. Only one machine learning model is selected, following the tutor's instruction. The same model is used for all feature representations so the comparison is fair.

# In[40]:


model = LogisticRegression(max_iter=1000)
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# In[41]:


scoring_metrics = {
    "accuracy" : "accuracy",
    "precision": "precision",
    "recall"   : "recall",
    "f1"       : "f1"
}


# ## Q1: Language model comparison
# 
# This method compares the three feature representations generated in Task 2: count vectors, unweighted GloVe vectors, and TF-IDF weighted GloVe vectors. The same Logistic Regression model and 5-fold cross validation are used for all three representations.

# #### creating count vectors

# In[42]:


from collections import Counter

count_vectors = []

for review in tk_reviews:
    freq = Counter(review)
    sparse_vector = [(vocab[word], count) 
                     for word, count in freq.items() 
                     if word in vocab]
    count_vectors.append(sparse_vector)


# The count_vectors list is converted into a sparse matrix for use with sklearn.

# In[43]:


row_indices = []
col_indices = []
data_values = []

for row_id, sparse_vector in enumerate(count_vectors):
    for word_index, word_freq in sparse_vector:
        row_indices.append(row_id)
        col_indices.append(word_index)
        data_values.append(word_freq)

X_count = csr_matrix(
    (data_values, (row_indices, col_indices)),
    shape=(len(count_vectors), len(vocab))
)

print("Count vector matrix shape:", X_count.shape)


# In[44]:


print("Count vector matrix shape:", X_count.shape)


# #### To Evaluate count vector representation using 5-fold cross validation

# In[45]:


count_results = cross_validate(
    model,
    X_count,
    y,
    cv=cv,
    scoring=scoring_metrics
)

print("Count vector accuracy scores:", count_results["test_accuracy"])
print("Mean accuracy:"  , count_results["test_accuracy"].mean())
print("Mean precision:" , count_results["test_precision"].mean())
print("Mean recall:"    , count_results["test_recall"].mean())
print("Mean F1-score:"  , count_results["test_f1"].mean())


# ### Unweighted embedding features

# #### To Evaluate unweighted embedding representation using 5-fold cross validation

# In[46]:


unweighted_results = cross_validate(
    model,
    unweighted_doc_vectors,
    y,
    cv=cv,
    scoring=scoring_metrics
)

print("Unweighted embedding accuracy scores:", unweighted_results["test_accuracy"])
print("Mean accuracy:"  , unweighted_results["test_accuracy"].mean())
print("Mean precision:" , unweighted_results["test_precision"].mean())
print("Mean recall:"    , unweighted_results["test_recall"].mean())
print("Mean F1-score:"  , unweighted_results["test_f1"].mean())


# ### TF-IDF weighted embedding features

# #### Evaluate TF-IDF weighted embedding representation using 5-fold cross validation

# In[47]:


weighted_results = cross_validate(
    model,
    weighted_doc_vectors,
    y,
    cv=cv,
    scoring=scoring_metrics
)

print("Weighted embedding accuracy scores:", weighted_results["test_accuracy"])
print("Mean accuracy:"  , weighted_results["test_accuracy"].mean())
print("Mean precision:" , weighted_results["test_precision"].mean())
print("Mean recall:"    , weighted_results["test_recall"].mean())
print("Mean F1-score:"  , weighted_results["test_f1"].mean())


# ### Q1 result comparison

# The average 5-fold cross validation accuracy values are compared to identify which feature representation performs best.

# In[48]:


q1_results = pd.DataFrame({
    "Feature Representation": [
        "Count Vector",
        "Unweighted GloVe",
        "TF-IDF Weighted GloVe"
    ],
    "Mean Accuracy": [
        count_results["test_accuracy"].mean(),
        unweighted_results["test_accuracy"].mean(),
        weighted_results["test_accuracy"].mean()
    ],
    "Mean Precision": [
        count_results["test_precision"].mean(),
        unweighted_results["test_precision"].mean(),
        weighted_results["test_precision"].mean()
    ],
    "Mean Recall": [
        count_results["test_recall"].mean(),
        unweighted_results["test_recall"].mean(),
        weighted_results["test_recall"].mean()
    ],
    "Mean F1-Score": [
        count_results["test_f1"].mean(),
        unweighted_results["test_f1"].mean(),
        weighted_results["test_f1"].mean()
    ]
})

q1_results


# ### Q2: Comparing title and review text information
# 
# This experiment checks whether adding extra text information improves classification accuracy. Three text inputs are compared: title only, review text only, and title combined with review text.

# #### To Prepare title and review text columns

# In[49]:


df["Title"]       = df["Title"].fillna("")
df["Review Text"] = df["Review Text"].fillna("")

title_text        = df["Title"]
review_text       = df["Review Text"]
title_review_text = df["Title"] + " " + df["Review Text"]

print("Title examples:")
print(title_text.head())

print("\nReview Text examples:")
print(review_text.head())

print("\nTitle + Review Text examples:")
print(title_review_text.head())


# #### To Generate count vector features for title only

# In[50]:


title_vectorizer = CountVectorizer(
    lowercase=True,
    token_pattern=r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
)

X_title = title_vectorizer.fit_transform(title_text)

print("Title feature matrix shape:", X_title.shape)


# #### To Generate count vector features for title only

# In[51]:


title_scores = cross_val_score(model, X_title, y, cv=5, scoring="accuracy")

print("Title only accuracy scores:", title_scores)
print("Mean accuracy:", title_scores.mean())


# #### Generate count vector features for review text only

# In[52]:


review_vectorizer = CountVectorizer(
    lowercase=True,
    token_pattern=r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
)

X_review = review_vectorizer.fit_transform(review_text)

print("Review text feature matrix shape:", X_review.shape)


# #### Evaluate review text only features using 5-fold cross validation

# In[53]:


review_scores = cross_val_score(model, X_review, y, cv=5, scoring="accuracy")

print("Review text only accuracy scores:", review_scores)
print("Mean accuracy:", review_scores.mean())


# #### Generate count vector features for title + review text

# In[54]:


combined_vectorizer = CountVectorizer(
    lowercase=True,
    token_pattern=r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
)

X_combined = combined_vectorizer.fit_transform(title_review_text)

print("Combined feature matrix shape:", X_combined.shape)


# #### Evaluate title + review text features using 5-fold cross validation

# In[55]:


combined_scores = cross_val_score(model, X_combined, y, cv=5, scoring="accuracy")

print("Title + review text accuracy scores:", combined_scores)
print("Mean accuracy:", combined_scores.mean())


# ### Q2 result comparison

# In[56]:


q2_results = pd.DataFrame({
    "Text Information Used": [
        "Title only",
        "Review Text only",
        "Title + Review Text"
    ],
    "Mean Accuracy": [
        title_scores.mean(),
        review_scores.mean(),
        combined_scores.mean()
    ]
})

q2_results


# ## Summary
# In Task 3, Logistic Regression was used as the selected classification model.
# - For Q1, three feature representations from Task 2 were compared using 5-fold cross validation: count vectors, unweighted GloVe vectors, and TF-IDF weighted GloVe vectors.
# - For Q2, three text input settings were compared: title only, review text only, and title combined with review text. The results showed that using both title and review text gave the highest accuracy, and the TF-IDF weighted GloVe representation performed best among the three feature representations.
