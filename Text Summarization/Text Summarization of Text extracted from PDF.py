#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pdfminer')


# In[1]:


import pdfminer
from pdfminer.high_level import extract_text

# Extract text from the PDF file
text = extract_text('Operations Management.pdf')
print(text)

with open('output.txt', 'w') as f:
    f.write(text)


# In[2]:


# Read Text File

with open('output.txt', 'r') as f:
    text = f.read()


# ### Text Cleaning

# In[3]:


import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


# In[4]:


sentences = sent_tokenize(text)
words = word_tokenize(text)


# In[5]:


stop_words = set(stopwords.words('english'))
words = [word for word in words if word not in stop_words]


# ### Frequency Table

# In[10]:


# create a frequency table that keeps track of how often each word appears in the text

from collections import defaultdict
frequency_table = defaultdict(int)
for word in words:
    frequency_table[word] += 1


# ### Score Sentences

# In[9]:


# Score each sentence by adding up the frequencies of its words

sentence_scores = defaultdict(int)
for i in sentences:
    for j in word_tokenize(i):
        if j in frequency_table:
            sentence_scores[i] += frequency_table[j]


# ### Generate Summary

# In[8]:


import heapq
summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)
summary = ' '.join(summary_sentences)
print(summary)


# In[ ]:




