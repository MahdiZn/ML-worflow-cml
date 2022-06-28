n/env python
# coding: utf-8
Project : Rain in Australia
# ## Description:
# __Context:__
# Predict next-day rain by training classification models on the target variable RainTomorrow.
# 
# __Content:__
# This dataset contains about 10 years of daily weather observations from many locations across Australia.
# 
# __Business Goal:__
# 
# Main objective of the analysis is to focus on prediction of value of the __RainTomorrow__  .
# 
# __RainTomorrow__ is the target variable to predict. It means -- did it rain the next day, Yes or No? This column is Yes if the rain for that day was 1mm or more.
# 
# Source & Acknowledgements
# Observations were drawn from numerous weather stations. The daily observations are available from [link](http://www.bom.gov.au/climate/data).
# An example of latest weather observations in Canberra:[link](http://www.bom.gov.au/climate/dwo/IDCJDW2801.latest.shtml)
# 
# Definitions adapted from http://www.bom.gov.au/climate/dwo/IDCJDW0000.shtml
# Data source:[source](http://www.bom.gov.au/climate/dwo/ and http://www.bom.gov.au/climate/data).
# 
# Copyright Commonwealth of Australia 2010, Bureau of Meteorology.

# ## Introduction
# 
# 
# 
# 
# ![image.png](attachment:image.png)
# 
# 
# 
# * We will be using the wetherAus data from kaggle.
# 
# * We have 23 columns including targer variable .
# 
# * This a problem of clasifiaiton binary .
# 
# * We will use a diferent model of clasification (logistic reg,Knn,svm,boosting) and compar them .
# 
# * we will use oversampling technique .
#  
# 
# 

# # Exploratory Data Analysis
# 
# * Import library and the csv file and examine its contents.
# * Output summary statistics and check variable data types
# * Check missing values 
# * check the correlation between variables one one hand and with the target 

# ### Library

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import os, sys
import seaborn as sns

########

from sklearn.model_selection import StratifiedShuffleSplit ,GridSearchCV
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, f1_score, roc_auc_score
from sklearn.kernel_approximation import Nystroem
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

import warnings
warnings.filterwarnings('ignore')


# #### Preparing data

# In[2]:


data= pd.read_csv('weatherAUS.csv')


# In[3]:


data.head()


# We have 145460 rows and 23 columns

# In[4]:


data.shape


# In[5]:


data.info()


# 
# #### Values of the Target 
# 
# * As we see we have a signicative inbalanced classes
# *We have more chance the not rain then rain . 
# No -> 110316/145460 = 75%
# Yes -> 31877/145460 = 21.91%

# In[3]:


data.RainTomorrow.value_counts()


# In[4]:


plt.figure()
sns.countplot(x=data.RainTomorrow)
plt.show()


# 
# 

# In[5]:


data.describe().T.round(2)
#Rainfall have a outliers 


# In[6]:


data.describe(include='object')

#we can to transform Date to datetime because is not object


# #### Identifying the number of unique values each variable

# In[7]:


a=pd.Series(data.dtypes)
b=pd.Series(data.nunique())


# In[8]:


pd.concat([a,b],axis=1).rename(columns={0:'Type',1:'unique_values'})


# In[9]:


df_unique=pd.concat([a,b],axis=1).rename(columns={0:'Type',1:'unique_values'})


# 
# 
# 
# 
# 
# 
# 

# #### Check for missing values 
# * Evaporation,Sunshine.Cloud3pm  have more the 40% of missing values ,then is good to drop them
# * The target have 2.2% of misisng values the we will  drop rows contains nan i the target

# In[10]:


round(data.isnull().sum()/len(data)*100,2)


# # Feature engineiring 
# * Handle missing values (drop and fill Nan data)
# * Separate the variables into numerical and categorical
# * Calculate the correlation between numerical vars before filling Nan 
# * Detect outliers and fill de Nan values with mean,median,moda

# __Drop the columns where more than 25% of the data is missing.__

# In[11]:


NanDf=(round(data.isnull().sum()/len(data)*100,2)).to_frame().rename(columns={0:'missing'})


# In[12]:


NanDf


# In[15]:


list(NanDf[NanDf['missing']>=25].index)


# In[16]:


list_drop=list(NanDf[NanDf['missing']>=25].index)
data.drop(list_drop,axis=1,inplace=True)


# In[17]:


data.isnull().sum()


# In[18]:


#RainTomorrow drop nan rows
data.dropna(subset=['RainTomorrow'],inplace=True)


# In[20]:


data.isnull().sum()


# #### Now we have to divide our data into categories and numbers
# * first we transform Date to datetime
# * Binary variables and rest of categories , numerical variables
# 

# In[21]:


data.Date.dtype


# In[22]:


data['Date']=pd.to_datetime(data['Date'])


# In[23]:


data['Month']=data['Date'].dt.month
data['Year']=data['Date'].dt.year
data.drop(columns='Date',inplace=True)


# In[24]:


binary_var = list(df_unique[df_unique['unique_values']==2].index)
binary_var


# In[25]:


categorical_variables =list(data.select_dtypes(include='object'))
categorical_variables 


# In[26]:


categorical_variables = list(set(categorical_variables) - set(binary_var))
categorical_variables 


# In[27]:


numerical_vars = list(set(data.columns)-set(categorical_variables)-set(binary_var))
numerical_vars 


# In[ ]:


#numerical_vars.remove('Date')


# In[ ]:





# ### filling missing values
# * Binary variables
# * Categorical variables
# * Numerical variables

# __Binary variables__

# In[28]:


data[binary_var].isnull().sum()


# In[29]:


data.RainToday.value_counts()


# In[33]:


data.RainToday.mode()[0]


# we can see that raintoday and rainfall have some relationship because they contain the same number of nan
# When we scale our data, perhaps we will see the relationship between this two variables

# In[34]:


data.Rainfall.isnull().sum()


# In[32]:


data[data['RainToday'].isnull()]['Rainfall']


# In[35]:


data['RainToday'].fillna(data['RainToday'].mode()[0],inplace=True)


# In[36]:


data[binary_var].isnull().sum()


# __Categorical variables__

# In[37]:


data[categorical_variables].isnull().sum()


# In[42]:


print("Mode of WindDir3pm is ",data.WindDir3pm.mode()[0])
print()
print(data.WindDir3pm.value_counts())


# In[37]:


data.WindGustDir.value_counts()


# In[38]:


data.WindDir9am.value_counts()


# In[43]:


for cat in categorical_variables:
    data[cat].fillna(data[cat].mode()[0],inplace=True)


# In[44]:


data[categorical_variables].isnull().sum()


# __Numerical variables__

# In[45]:


data[numerical_vars].isnull().sum()


# - Before doing any data filling in the numerical values, first we are going to see the correlation between these variables to see if we can precede any one

# In[46]:


# we draw the distributon
# some vars are skewed right and left
data[numerical_vars].hist(color='green', figsize=(12, 9))
plt.show()


# In[51]:


#data[numerical_vars].corr().stack().to_frame().reset_index() \
#.rename(columns={'level_0':'var1','level_1':'var2',0:'correlation'})


# In[ ]:





# In[47]:


corr_values = data[numerical_vars].corr()

# Simplify by emptying all the data below the diagonal

#np.tril_indices_from ->Return the indices for the lower-triangle of arr
tril_index = np.tril_indices_from(corr_values)

# Make the unused values NaNs
for coord in zip(*tril_index):
    corr_values.iloc[coord[0], coord[1]] = np.NaN
    
# Stack the data and convert to a data frame
corr_values = (corr_values
               .stack()
               .to_frame()
               .reset_index()
               .rename(columns={'level_0':'feature1',
                                'level_1':'feature2',
                                0:'correlation'}))

# Get the absolute values for sorting
corr_values['abs_correlation'] = corr_values.correlation.abs()


# Correlation between numerical variable:

# In[54]:


corr_values.sort_values(by='abs_correlation',ascending=False).query('abs_correlation > 0.65')


# As we can see there are highly correlated variables, we will proceed to eliminate Temp3pm, Temp9am, Pressure9am
# 

# In[55]:


data.drop(columns=['Temp3pm','Temp9am','Pressure9am'],inplace=True)


# In[56]:


lista=['Temp3pm','Temp9am','Pressure9am']
for i in lista:
        numerical_vars.remove(i)


# In[57]:


numerical_vars


# In[58]:


data[numerical_vars].isnull().sum()


# Before we see if the data have outliers or not to know if we will fill the Nan with mean or median

# __Outlieres__
# 
# * Except humidity3pm all variables have outliers
# * filling missing values 

# In[59]:


fig,ax=plt.subplots(figsize=(15,19),nrows=len(numerical_vars),ncols=1)
i=0
for col in numerical_vars:
    sns.boxplot(data=data[numerical_vars],x=col,ax=ax[i])
    i=i+1


# In[60]:


data['Humidity3pm'].fillna(data['Humidity3pm'].mean(),inplace=True)


# In[ ]:





# In[61]:



for num in numerical_vars:
    
    data[num].fillna(data[num].median(),inplace=True)


# In[62]:


data[numerical_vars].isnull().sum()


# In[63]:


data.isnull().sum()


# In[64]:


#Clean dataset
data.to_csv('Rain_australia.csv',index=False)


# ## Categorical data transformation
# 
# * Using LabelEncoder,LabelBinarizer
# * Save dataset
# * 
# 

# In[65]:


data=pd.read_csv('Rain_australia.csv')
data.head()


# In[66]:


data[categorical_variables].nunique()


# In[67]:


data.WindDir3pm.unique()


# In[68]:


data.Location.unique()


# __I have decided to use label encoder instead of onehotencoder, because first to avoid multicoleanility and so that our dataset does not grow too much, also the variables seem to have an order for example of city or north south west etc.__

# In[69]:


from sklearn.preprocessing import LabelBinarizer, LabelEncoder

lb=LabelBinarizer()
le=LabelEncoder()

for k in binary_var:
    data[k]= lb.fit_transform(data[k])
    


# In[70]:


for j in categorical_variables:
    data[j]= le.fit_transform(data[j])    


# ### we must scale data 

# In[71]:


from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()


# In[72]:


for column in [categorical_variables + numerical_vars]:
    data[column] = mm.fit_transform(data[column])


# In[73]:


data.describe().T.round(2)


# In[75]:


data.to_csv('clean_Rain_austrialia.csv',index=False)

