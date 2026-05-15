#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 1. Basic Text Pre-processing
# #### Student Name: Shiv Mahesh Bhosale
# #### Student ID: s4192953
# 
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used:
# * pandas
# * re
# * numpy
# * matplotlib
# * nltk
# * collections
# * itertools
# ## Introduction
# The purpose of this task is to perform text preprocessing on clothing review data before building machine learning models.
# 
# The main preprocessing steps include:
# * Tokenisation using the required regular expression
# * Lowercasing all text
# * Removing single-character words
# * Removing stopwords
# * Removing rare words based on term frequency
# * Removing the top 20 most frequent words based on document frequency
# * Building a clean vocabulary
# 
# The outputs generated in this task are:
# * 'processed.csv'
# * 'vocab.txt'

# ## Importing libraries 
# 

# In[1]:


import pandas as pd
import numpy as np
import regex as re
import seaborn as sns
import matplotlib.pyplot as plt
from nltk import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from nltk.probability import *
from itertools import chain
from collections import Counter


# ### 1.1 Examining and loading data
# In this section, the clothing review dataset is loaded and explored to understand its structure.
# 
# The following checks are performed:
# - Dataset shape
# - Column names
# - Missing values
# - Example review records
# 
# This helps verify that the dataset is loaded correctly before preprocessing begins.

# In[2]:


filename = './assignment3.csv'
df = pd.read_csv(filename)


# In[3]:


df


# In[4]:


df.shape


# In[5]:


df.dtypes


# In[6]:


df.isnull().sum()


# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


df['Division Name'].value_counts()


# In[10]:


df['Department Name'].value_counts()


# In[11]:


df['Class Name'].value_counts()


# In[12]:


plt.figure(figsize=(6, 4))
sns.boxplot(data=df['Age'],width=.5)
plt.show()


# ### 1.2 Pre-processing data
# This section applies all required preprocessing steps to the 'Review Text'.
# The preprocessing pipeline improves text quality by reducing noise and keeping meaningful terms for later feature extraction and classification tasks.

# In[13]:


df['Review Text']


# In[14]:


df['Review Text'].isna().value_counts()


# Converting the 'Review text' into list for further processing.

# In[15]:


reviews = df['Review Text'].tolist()


# In[16]:


reviews


# In[17]:


print(f'Total reviews: {len(reviews)}')
print(f"\nSample review:\n{reviews[0]}")


# Using provided regex pattern to make the tokens.

# In[18]:


pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
tokenized_reviews = [re.findall(pattern,review) for review in reviews]


# In[19]:


print(f'Sample tokens of the first review:\n{tokenized_reviews[0]}')


# This function prints basic corpus statistics such as vocabulary size, total tokens, and review length distribution.

# In[20]:


def stats_print(tokenized_reviews):
    words = list(chain.from_iterable(tokenized_reviews)) # we put all the tokens in the corpus in a single list
    vocab = set(words) # compute the vocabulary by converting the list of words/tokens to a set, i.e., giving a set of unique words
    lexical_diversity = len(vocab)/len(words)
    print("Vocabulary size: ",len(vocab))
    print("Total number of tokens: ", len(words))
    print("Lexical diversity: ", lexical_diversity)
    print("Total number of reviews:", len(tokenized_reviews))
    lens = [len(article) for article in tokenized_reviews]
    print("Average review length:", np.mean(lens))
    print("Maximun review length:", np.max(lens))
    print("Minimun review length:", np.min(lens))
    print("Standard deviation of review length:", np.std(lens))


# In[21]:


stats_print(tokenized_reviews)


# All review text is converted into lowercase to ensure consistency.

# In[22]:


tokenized_reviews = [
    [token.lower() for token in tokens]
    for tokens in tokenized_reviews
]
print(f'Sample tokens in lowercase:\n{tokenized_reviews[0]}')


# Removing Single Character Words to clean the noise from the tokenized reviews.

# In[23]:


tokenized_reviews = [
    [token for token in tokens if len(token) > 1]
    for tokens in tokenized_reviews
]


# In[24]:


print(f'Sample tokens in lowercase:\n{tokenized_reviews[0]}')


# In[25]:


stats_print(tokenized_reviews)


# 
# Removing the stopwords by using the 'stopwords_en.txt'. 
# Stopwords do not contribute meaningful information to text classification.

# In[26]:


def load_stopwords(filename1):
    with open(filename1,'r') as f:
        return set(word.strip().lower() for word in f.readlines())


# In[27]:


def remove_stopwords(tokenized_reviews, stopwords_set):
    return [
        [token for token in tokens if token not in stopwords_set]
        for tokens in tokenized_reviews
    ]


# In[28]:


stop_words = load_stopwords('stopwords_en.txt')


# In[29]:


len(stop_words)


# In[30]:


print(f"\nSample review before removal of stopwords: \n{tokenized_reviews[0]}")


# In[31]:


tokenized_reviews = remove_stopwords(tokenized_reviews, stop_words)


# In[32]:


print(f"\nSample review after removal of stopwords: \n{tokenized_reviews[0]}")


# In[33]:


stats_print(tokenized_reviews)


# Removing the less frequent words by term frequency. 
# These rare words are unlikely to contribute useful patterns for machine learning models and may increase sparsity.

# In[34]:


words = list(chain.from_iterable(tokenized_reviews))
term_fd = FreqDist(words)


# In[35]:


lessFreqWords = set(term_fd.hapaxes())
lessFreqWords


# In[36]:


def removeLessFreqWords(review):
    return [w for w in review if w not in lessFreqWords]

tokenized_reviews = [removeLessFreqWords(review) for review in tokenized_reviews]


# In[37]:


print(f"\nSample review after removal of less frequent words: \n{tokenized_reviews[0]}")


# In[38]:


stats_print(tokenized_reviews)


# Removing most frequent words by document frequency. 
# very common words appearing in many reviews may not help distinguish between recommended and non-recommended products.
# Removing these frequent words helps improve feature discrimination.

# In[39]:


doc_freq = Counter()


# In[40]:


for review in tokenized_reviews:
    doc_freq.update(set(review))


# In[41]:


most_common_20 =doc_freq.most_common(20)


# In[42]:


most_common_20


# In[43]:


most_frequent_words = [w for w, c in most_common_20]


# In[44]:


most_frequent_words


# In[45]:


def remove_most_freq_words(review):
    return [w for w in review if w not in most_frequent_words]


# In[46]:


tokenized_reviews = [remove_most_freq_words(review) for review in tokenized_reviews]


# In[47]:


stats_print(tokenized_reviews)


# ## Saving required outputs
# Save the requested information as per specification.
# - vocab.txt
# - processed.csv

# In[48]:


df['Review Text'] = [' '.join(tokens) for tokens in tokenized_reviews]


# In[49]:


df.to_csv('processed.csv', index=False)


# In[50]:


check_df = pd.read_csv('processed.csv')


# In[51]:


len(check_df)


# In[52]:


print(f"Columns: {check_df.columns.tolist()}")


# In[53]:


check_df.head()


# In[54]:


check_df.shape


# Building the vocabulary
# 

# In[55]:


def build_vocab(tokenized_reviews):
    words = list(chain.from_iterable(tokenized_reviews))
    vocab_words = sorted(set(words))
    vocab = {word: idx for idx, word in enumerate(vocab_words)}
    with open("vocab.txt", "w") as f:
        for word, idx in vocab.items():
            f.write(f"{word}:{idx}\n")
    print("Vocabulary size:", len(vocab))
    return vocab


# In[56]:


vocab = build_vocab(tokenized_reviews)


# ## Task 1 summary
# In this task, a complete text preprocessing pipeline was implemented for clothing reviews.

# The preprocessing included:
# - Tokenisation
# - Lowercasing
# - Stopwords removal
# - Less frequent words removal
# - Most frequent words removal
# - Vocabulary construction
# 
