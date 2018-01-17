
# coding: utf-8

# # Recommender System for the MovieLens dataset
# 
# ## Importing Libraries and dataset

# In[4]:

import numpy as np
import pandas as pd


# In[5]:

column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=column_names)


# In[6]:

df.head()


# Getiing and merging movie titles:

# In[7]:

movie_titles = pd.read_csv("Movie_Id_Titles")
movie_titles.head()


# In[8]:

df = pd.merge(df,movie_titles,on='item_id')
df.head()


# # EDA
# 
# What are some best rated movies
# 
# ## Visualization Imports

# In[9]:

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
get_ipython().magic('matplotlib inline')


# Creating a ratings dataframe with average rating and number of ratings:

# In[10]:

df.groupby('title')['rating'].mean().sort_values(ascending=False).head()


# In[11]:

df.groupby('title')['rating'].count().sort_values(ascending=False).head()


# In[12]:

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()


# Number of ratings column:

# In[13]:

ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.head()


# Now a few histograms:

# In[14]:

plt.figure(figsize=(10,4))
ratings['num of ratings'].hist(bins=70)


# In[15]:

plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=70)


# In[16]:

sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)


# ## Recommending Similar Movies

# Creating a matrix that has the user ids on one axis and the movie titles on another axis. Each cell will then consist of the rating the user gave to that movie. Note there will be a lot of NaN values, because most people have not seen most of the movies.

# In[17]:

moviemat = df.pivot_table(index='user_id',columns='title',values='rating')
moviemat.head()


# Most rated movie:

# In[18]:

ratings.sort_values('num of ratings',ascending=False).head(10)


# Let's consider two movies: starwars, a sci-fi movie. And Liar Liar, a comedy.

# In[19]:

ratings.head()


# Now let's grab the user ratings for those two movies:

# In[20]:

starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']
starwars_user_ratings.head()


# Using corrwith() method to get correlations between two pandas series:

# In[21]:

similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)


# In[22]:

type(similar_to_liarliar)


# Cleaning this by removing NaN values and using a DataFrame instead of a series:

# In[23]:

corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.head()


# Now if we sort the dataframe by correlation, we should get the most similar movies, however we get some results that don't really make sense. This is because there are a lot of movies only watched once by users who also watched star wars (it was the most popular movie). 

# In[24]:

corr_starwars.sort_values('Correlation',ascending=False).head(10)


# We can fix this by filtering out movies that have less than 100 reviews (this value was chosen based off the histogram from earlier).

# In[25]:

corr_starwars = corr_starwars.join(ratings['num of ratings'])
corr_starwars.head()


# Now after sorting the values, the titles make a lot more sense:

# In[26]:

corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head()


# Doing the same for the comedy Liar Liar:

# In[27]:

corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head()


# # Done!
