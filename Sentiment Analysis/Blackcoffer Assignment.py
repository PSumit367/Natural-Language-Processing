#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as plt


# In[2]:


df = pd.read_excel('C:/Users/user/OneDrive/Desktop/Assignments/Blackcoffer/Input.xlsx')
df.info()
df.head()


# In[3]:


import newspaper
from newspaper import Article
 
# Assign single url
url ='https://insights.blackcoffer.com/impact-of-covid-19-pandemic-on-office-space-and-co-working-industries/'
 
# Extract web data
url_i = newspaper.Article(url="%s" % (url), language='en')
url_i.download()
url_i.parse()
print(url_i.text)


# In[4]:


articles=[]
for url in df['URL']:
    try:
        url_i = newspaper.Article(url="%s" % (url), language='en')
        url_i.download()
        url_i.parse()
        articles.append(url_i.text)
    except:
        articles.append('')   
df['text']=articles
df.head(10)


# ### Text Cleaning

# In[6]:


import re
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[9]:


# Tokenizer

def transform(text):
    text = re.sub(r'[^a-zA-Z-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    texts =text.split()
    filtered_tokens = [word for word in texts if word.lower() not in stopwords.words('english')]
    return' '.join(filtered_tokens) 

df['t_text']=df['text'].apply(transform)
df.t_text


# ## Average Number of Words Per Sentence

# In[54]:


# Calculate Word Count
df['word_count'] = df['t_text'].apply(lambda x: len(str(x).split()))

def Average_words_per_sentence(df):
    word_count = df['word_count']       
    sentence_count = len(nltk.sent_tokenize(df['text']))
    if sentence_count != 0:
        return word_count / sentence_count
    else:
        return 0

df['Average_number_of_words_per_sentence'] = df.apply(Average_words_per_sentence, axis=1)
df.Average_number_of_words_per_sentence


# ## Average Words Length 

# In[55]:


Character_length = df['t_text'].apply(lambda x: sum(len(word) for word in x))
df['Avg_Word_length'] = np.nan

for i in range(0,len(df)):
    df['Avg_Word_length'] = df['t_text'].apply(lambda x: sum(len(word) for word in x))/df['t_text'].apply(lambda x: len(str(x).split()))
    
df.Avg_Word_length


# ## Complex Word Count

# In[12]:


def complex_word_count(text):
    vowels = 'aeiou'                  # Required syllables
    count = 0
    for i in text.split():
        syllables = 0
        for j in i:
            if j in vowels:
                syllables += 1
        if syllables > 2:
            count += 1
    return count

df['complex_word_count'] = df['t_text'].apply(complex_word_count)
df.complex_word_count


# ## Syllable Count per Word

# In[56]:


def Syllable_Count_Per_Word(text):
    vowels = 'aeiou'
    word_count = len(text.split())
    total_syllables = 0

    for word in text.split():
        if re.search(r'\b\w+es\b|\b\w+ed\b', word):
            continue
        else:
            syllables = 0
            for i in word:
                if i in vowels:
                    syllables += 1
            total_syllables += syllables

    if word_count > 0:
        return total_syllables / word_count
    else:
        return 0

df['Syllable_Count_Per_Word'] = df['t_text'].apply(Syllable_Count_Per_Word)
df.Syllable_Count_Per_Word


# ## Personal Pronoun Count

# In[14]:


df['personal_pronoun_count']=np.nan

for i in range(0,len(df)):
    df['personal_pronoun_count'] = df['text'].apply(lambda x: sum(1 for word in nltk.word_tokenize(x) if word.lower() in ['i', 'we', 'my', 'ours', 'us']))

df.personal_pronoun_count


# ## Analysis of Readability 
#  Analysis of Readability is calculated using the Gunning Fox index formula described below.
#  Average Sentence Length = the number of words / the number of sentences
#  Percentage of Complex words = the number of complex words / the number of words 
#  Fog Index = 0.4 * (Average Sentence Length + Percentage of Complex words)
# In[57]:


def calculation_metrics(df):
    word_count = df['word_count']
    sentence_count = len(nltk.sent_tokenize(df['text']))
    complex_word_count = df['complex_word_count']

    if sentence_count != 0:
        df['Average_Sentence_Length'] = word_count / sentence_count
    else:
        df['Average_Sentence_Length'] = 0

    if word_count != 0:
        df['Percentage_of_Complex_words'] = complex_word_count / word_count
    else:
        df['Percentage_of_Complex_words'] = 0

    df['Fog_Index'] = 0.4 * (df['Average_Sentence_Length'] + df['Percentage_of_Complex_words'])

    return df

df = df.apply(calculation_metrics, axis=1)


# In[58]:


df.Average_Sentence_Length


# In[59]:


df.Percentage_of_Complex_words


# In[60]:


df.Fog_Index


# ## Sentimental Analysis

# In[19]:


file_path="C:/Users/user/OneDrive/Desktop/Assignments/Blackcoffer/StopWords"
def read_text_file(file_path):
    with open(file_path, 'r') as f:
        print(f.read()) 


# In[20]:


stop_words = []
for i in os.listdir(file_path):
    with open(os.path.join(file_path, i), 'r') as f:
        words = f.read().splitlines()
        stop_words.extend(words)


# In[24]:


stop_words = [word.lower() for word in stop_words] 
stopword = nltk.corpus.stopwords.words('english')
stop_words.extend(stopword)     


# In[25]:


# Text cleaning using updated stopwords
def transform(text):
    text = re.sub(r'[^a-zA-Z-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

df['s_text']=df['text'].apply(transform)
df.s_text


# In[26]:


sentiment="C:/Users/user/OneDrive/Desktop/Assignments/Blackcoffer/MasterDictionary"
def read_text_file(sentiment):
    with open(sentiment, 'r') as word:
        print(word.read())


# In[27]:


positive=set()
negative=set()

for i in os.listdir(sentiment):
    if i =='positive-words.txt':
        with open(os.path.join(sentiment,i),'r') as f:
          positive.update(set(f.read().splitlines()))
    else:
        with open(os.path.join(sentiment,i),'r') as f:
          negative.update(set(f.read().splitlines()))


# ### Positive/Negative Words

# In[28]:


Positive_words=[]
Negative_words=[]
for i in range(0,len(df['s_text'])):
  Positive_words.append([word for word in df['s_text'][i] if word.lower() in positive])
  Negative_words.append([word for word in df['s_text'][i] if word.lower() in negative])        


# ### Positive/Negative Score

# In[29]:


Positive_score=[]
Negative_score=[]
for i in range(0,len(df['s_text'])):
    Positive_score.append(len(Positive_words[i]))
    Negative_score.append(len(Negative_words[i]))


# ### Polarity/ Subjectivity Score

# In[30]:


# Polarity Score = (Positive Score – Negative Score)/ ((Positive Score + Negative Score) + 0.000001)
# Subjectivity Score = (Positive Score + Negative Score)/ ((Total Words after cleaning) + 0.000001)

Polarity_score=[]
Subjectivity_score=[]
for i in range(0,len(df['t_text'])):
    Polarity_score.append((Positive_score[i] - Negative_score[i]) / ((Positive_score[i] + Negative_score[i]) + 0.000001))
    Subjectivity_score.append((Positive_score[i] + Negative_score[i]) / ((len(df['t_text'][i])) + 0.000001))
    


# In[31]:


df['Positive_score']= Positive_score
df['Negative_score']= Negative_score
df['Polarity_score']= Polarity_score
df['Subjectivity_score']= Subjectivity_score


# In[32]:


df.Positive_score


# In[33]:


df.Negative_score


# In[34]:


df.Polarity_score


# In[35]:


df.Subjectivity_score


# ## Output Data

# In[90]:


data = ['URL_ID','URL','Positive_score','Negative_score','Polarity_score','Subjectivity_score',
             'Average_Sentence_Length','Percentage_of_Complex_words',
             'Fog_Index','Average_number_of_words_per_sentence','complex_word_count',
             'word_count','Syllable_Count_Per_Word','personal_pronoun_count','Avg_Word_length']


# In[99]:


output_data = df.drop(['text', 't_text','s_text'], axis=1)


# In[100]:


output_data = output_data.reindex(columns=data)


# In[101]:


URL_ID 24,37,108 does not exists i,e. page does not exist, throughs 404 error
# so we are going to drop these rows from the table

output_data.drop([24,37,108],axis= 0, inplace=True)


# In[104]:


output_data.to_excel('C:/Users/user/OneDrive/Desktop/Output_Data.xlsx')


# In[105]:


output_data.head(10)


# In[ ]:




