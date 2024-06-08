#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt   


# In[3]:


#read dataset
file_loc = 'C:\\Users\\Jay Pal\\Downloads\\vaccination_all_tweets.csv'
tweet_data = pd.read_csv(file_loc)
tweet_data.head(2)


# # Information about the dataset

# In[4]:


tweet_data.info()


# # Number of Record and Column in the data

# In[5]:


print("Number of Row in the dataset: ", tweet_data.shape[0])
print('Number of column in the dataset: ',tweet_data.shape[1] )


# # Visualize the null value from the dataset

# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns
fig,ax= plt.subplots(figsize=(16,13))
sns.heatmap(tweet_data.isna(), yticklabels=False, cbar=False, cmap='viridis')
plt.xticks(rotation=90, size=15)


# # Count and calculate the percentage of null value present in the dataset
# 

# In[7]:


na_value = tweet_data.isna().sum()
percent = (tweet_data.isna()/tweet_data.isna().count())*100
na = pd.concat([na_value, percent],axis = 1, keys=["Total_NA_Value", '%_NA_Value'])
na


# # Drop All Na Value from the dataset

# In[8]:


tweet_data.dropna(inplace=True)
tweet_data


# # Number of records after droping the NA Value

# In[9]:


tweet_data['user_location'].value_counts().sum()


# # Find the Number of Unique value in each column

# In[10]:


def unique_values(df):
    tc = {}
    uniques = []  # Initialize uniques as a list
    for col in df.columns:
        unique_count = df[col].nunique()
        uniques.append(unique_count)  # Append unique counts to the list
    tc['unique_values'] = uniques
    return np.transpose(tc)

# Assuming tweet_data is a pandas DataFrame
unique_values(tweet_data)




# In[11]:


print(unique_values(tweet_data))


# # Find the most frequent data or value in each column

# In[12]:


def most_frequent_value(df):
    tt = {}
    items = []
    vals = []
    total = df.shape[0]
    
    for col in df.columns:
        most_freq_item = df[col].mode()[0]
        frequency = df[col].value_counts().iloc[0]
        items.append(most_freq_item)
        vals.append(frequency)
        
    tt['Most Frequent item'] = items
    tt['frequence'] = vals
    tt['Percent from total'] = np.round(np.array(vals) / total * 100, 3)
    
    return pd.DataFrame(tt).transpose()

most_frequent_value(tweet_data)

    


# # Visualizations

# In[13]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Mock DataFrame for demonstration
file_loc = 'C:\\Users\\Jay Pal\\Downloads\\vaccination_all_tweets.csv'
tweet_data = pd.read_csv(file_loc)
tweet_data.head(2)
def plot_count(feature, title, df, size=3, ordered=True):
    sns.set_style("whitegrid")
    f, ax = plt.subplots(1, 1, figsize=(20, 12))
    total = float(len(df))
    
    if ordered:
        g = sns.countplot(x=df[feature], order=df[feature].value_counts().index[:25], palette='Set3')
    else:
        g = sns.countplot(x=df[feature], palette='Set3')
    
    g.set_xlabel(title, fontsize=15)
    g.set_ylabel("Count", fontsize=15)
    g.set_title("Number and percentage of {}".format(title), size=15)
    
    if size > 3:
        plt.xticks(rotation=70, size=15)
        plt.yticks(rotation=0, size=15)
    
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., 
                height, 
                '{:1.2f}%'.format(100 * height / total),
                ha="center", size=15)
    
    plt.show()





# # Number of percentage of Username from all country and visualize the data

# In[14]:


plot_count('user_name', 'User Name', tweet_data, size=4)


# # Number and percentage of user location from all the country

# In[15]:


plot_count('user_location', 'User location', tweet_data, size=4)


# # Number and Percentage of Sourse from the all country

# In[16]:


import matplotlib.pyplot as plt
plot_count('source', 'Source', tweet_data, size=4)


# # Number and Percentage of verified user from all the country

# In[17]:


plot_count('user_verified', 'User_verified', tweet_data, size=4)


# # Create a new dataframe and extract tweet data from only india

# In[18]:


tweet_data['useer_location'] = tweet_data['user_location'].str.lower()


# In[19]:


# Convert user_location column to lowercase, handling NaN values
tweet_data['user_location'] = tweet_data['user_location'].str.lower()

# Define the cities and states in a regex pattern
city_state = 'india|delhi|maharashtra|tamil nadu|bihar|uttar pradesh|gujarat|madhya pradesh|west bengal'

# Filter the tweets based on the user location, handling NaN values
india_tweet_data = tweet_data[tweet_data['user_location'].fillna('').str.contains(city_state)]


# In[20]:


india_tweet_data


# # Find the number of unique value in each column from India

# In[21]:


unique_values(india_tweet_data)


# # Number and percentage of user location from India

# In[22]:


plot_count('user_location', 'User location',india_tweet_data, size=4)


# # Number and percentage of user name from India

# In[23]:


plot_count('user_name', 'User_name',india_tweet_data, size=4)


# # Number and Percentage of Sourse from India

# In[24]:


plot_count('source', 'source',india_tweet_data, size=4)


# # Number and Percentage of hashtags trends in India

# In[25]:


plot_count('hashtags', 'hashtags',india_tweet_data, size=4)


# # Number and Percentage of Verified trends in India

# In[26]:


plot_count('user_verified', 'user_verified',india_tweet_data, size=4)


# # Visualize Word Cloud
Word Clouds are graphical representation of word frequency that give greater prominence to words that appear more frequently in a source text. The larger the word in the visual the more common the word was in the text data

# In[27]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

def plot_wordcloud(data, title=""):
    text = " ".join(i for i in data.dropna())
    stopwords = set(STOPWORDS)
    stopwords.update(["t", "co", "https", "amp", "U"])
    wordcloud = WordCloud(stopwords=stopwords, scale=4, max_font_size=50, max_words=500, background_color="black").generate(text)
    
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.imshow(wordcloud, interpolation='bilinear')


# # The Most appear words tweets from India

# In[28]:


print("Total tweets from India:",india_tweet_data.shape[0])
plot_wordcloud(india_tweet_data['text'],title = 'The most prevalent words tweets in india')


# # The Most appear words tweets from Delhi and Noida City

# In[66]:


delhi_tweet_data = india_tweet_data[india_tweet_data['user_location'].str.contains('new delhi|delhi|noida')]
print("Total tweets from Delhi and Noida:",delhi_tweet_data.shape[0])
plot_wordcloud(delhi_tweet_data['text'],title = 'The most prevalent words tweets in delhi')                                   


# # Hashtag Analysis

# In[33]:


def plot_features_distribution(features, title,df,islog=False):
    plt.figure(figsize=(12,6))
    plt.title(title)
    for feature in features:
        if (islog):
            sns.distplot(np.log1p(df[feature]),kde=True,hist=False, bins=120, label=feature)
    plt.xlabel('')
    plt.legend()
    plt.show()


# # Data Cleaning

# In[34]:


india_tweet_data['hashtags'] = india_tweet_data['hashtags'].replace(np.nan, '[None]', regex=False)
india_tweet_data['hashtags'] = india_tweet_data['hashtags'].apply(lambda x: x.replace('\\N', ''))

india_tweet_data['hashtags_count'] = india_tweet_data['hashtags'].apply(lambda x: len(x.split(',')))


# # Plot the graph hashtag cout per tweet india

# In[35]:


features = ['hashtags_count']
plot_features_distribution(features, 'Hashtags per tweet in India', india_tweet_data, islog=False)


# # Total number and list of all individual hashtags

# In[36]:


india_tweet_data['hashtags_individual'] = india_tweet_data['hashtags'].apply(lambda x: x.split(','))
from itertools import chain
india_hashtags = set( chain.from_iterable(list(india_tweet_data['hashtags_individual'])))
print(f'There are total hashtags from india:{len(india_hashtags)}')
india_hashtags


# # Create a new dataframe of data

# In[37]:


india_tweet_data['datedt'] = pd.to_datetime(india_tweet_data['date'])
india_tweet_data['datedt']


# # Extract year, day, month, dayofweek,hour, minutes and store it in a new column same dataset

# In[38]:


india_tweet_data['year'] = india_tweet_data['datedt'].dt.year
india_tweet_data['month'] = india_tweet_data['datedt'].dt.month
india_tweet_data['day'] = india_tweet_data['datedt'].dt.day
india_tweet_data['dayofweek'] = india_tweet_data['datedt'].dt.dayofweek
india_tweet_data['hour'] = india_tweet_data['datedt'].dt.hour
india_tweet_data['minute'] = india_tweet_data['datedt'].dt.minute
india_tweet_data['dayofyear'] = india_tweet_data['datedt'].dt.dayofyear
india_tweet_data['date_only'] = india_tweet_data['datedt'].dt.date


# # First Three Records

# In[39]:


india_tweet_data.head(3)


# # Number of count of Tweet datawise

# In[40]:


india_tweet_agg_data = india_tweet_data.groupby(['date_only'])['text'].count().reset_index()
india_tweet_agg_data.columns = ['date_only','count']
india_tweet_agg_data.head(3)


# # Plot the Line Graph of count number of tweets per day of year in india

# In[41]:


def plot_time_variation_graph(df, x='data_only',y='count', hue=None, size=1, title='', is_log=False):
    f,ax = plt.subplots(1,1,figsize=(6*size,3*size))
    g = sns.lineplot(x=x,y=y, hue=hue, data=df)
    plt.xticks(rotation=90)
    if hue:
        plt.title(f'{y} grouped by {hue}| {title}')
    else:
        plt.title(f'{y} | {title}')
    if(is_log):
        ax.set(yscale = 'log')
    ax.grid(color ='black', linestyle= 'dotted', linewidth= 0.75)
    plt.show()


# In[42]:


plot_time_variation_graph(india_tweet_agg_data, title='Number of tweet per day of year in india',size=3)


# In[43]:


def plot_time_variation_graph(df, x='data_only', y='count', hue=None, size=1, title='', is_log=False):
    f, ax = plt.subplots(1, 1, figsize=(6 * size, 3 * size))
    g = sns.lineplot(x=x, y=y, hue=hue, data=df)
    plt.xticks(rotation=90)
    if hue:
        plt.title(f'{y} grouped by {hue} | {title}')
    else:
        plt.title(f'{y} | {title}')
    if is_log:
        ax.set(yscale='log')
    ax.grid(color='black', linestyle='dotted', linewidth=0.75)
    plt.show()

# Ensure the column names are correct in the function call
plot_time_variation_graph(india_tweet_agg_data, x='date_only', y='count', title='Number of tweet per day of year in india', size=3)


# # Number and Percentage of tweet per day of week in india

# In[44]:


plot_count("dayofweek" , 'tweets per day of week in india' ,india_tweet_data, size =3, ordered = False)


# # Number and Percentage of Tweet per date in India

# In[45]:


plot_count("date_only" , 'tweets per date in india' ,india_tweet_data, size =3, ordered = False)


# # Number and Percentage of tweet per hour in India

# In[46]:


plot_count("hour" , 'tweets per hour  in india' ,india_tweet_data, size =3, ordered = False)


# # Number and Percentage of tweet per minute in India

# In[47]:


plot_count("minute" , 'tweets per minute in india' ,india_tweet_data, size =3, ordered = False)


# # Apply Sentiment Intensity analyzer
VADER (Valence Aware Dictionary and sentiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media. VADER uses a combination of A sentiment lexicon is a list of lexical features (e g., words) which are generally labeled according to their semantic orientation as either positive or negative. VADER not only tells about the Positivity and Negativity score but also tells us about how positive of negative a sentiment is.
# In[48]:


get_ipython().system('pip install vaderSentiment')

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def sentiment_analysis(post):
    scores = sia.polarity_scores(post)
    if scores["compound"] > 0:
        return "Positive"
    elif scores["compound"] < 0:
        return "Negative"
    else:
        return "Neutral"

# Example usage
post = "I love this product! It works great."
print(sentiment_analysis(post))  # Output should be "Positive"


# In[49]:


india_tweet_data['sentiment'] = india_tweet_data['text'].apply(lambda x: sentiment_analysis(x))


# In[50]:


india_tweet_data.head(3)


# # Visualize the count of sentiment

# In[51]:


def sentiment_graph(df, feature, title):
    sns.set_style('whitegrid')
    counts = df[feature].value_counts()
    fig, ax = plt.subplots(figsize=(15, 5))
    counts.plot(kind="bar", ax=ax, color= 'green'  )
    ax.set_ylabel(f'Counts: {title} sentiments', size=12)
    plt.suptitle(f"Sentiment analysis: {title}",size=12)
    plt.tight_layout()
    plt.show()
sentiment_graph(india_tweet_data, 'sentiment', 'Text')


# # Vizualize the wordcloud from the positive sentiments tweet

# In[68]:


print("Total tweets Positive text:",india_tweet_data.shape[0])
plot_wordcloud(india_tweet_data['text'],title = 'The most prevalent words tweets in Positive sentiment ')                                   


# # Vizualize the wordcloud from the negative sentiments tweet                             

# In[69]:


print("Total tweets negative text:",india_tweet_data.shape[0])
plot_wordcloud(india_tweet_data['text'],title = 'The most prevalent words tweets in negative sentiment ')                                   


# # Vizualize the wordcloud from the netual sentiments tweet                             

# In[70]:


print("Total tweets Netual text:",india_tweet_data.shape[0])
plot_wordcloud(india_tweet_data['text'],title = 'The most prevalent words tweets in Netual sentiment ')                                   


# # Thankyou!
