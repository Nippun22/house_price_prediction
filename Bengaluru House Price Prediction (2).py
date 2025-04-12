#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams["figure.figsize"]=(20,10) #defalut figure size


# In[2]:


df1= pd.read_csv(r"C:\Users\lajbh\Documents\Lovely Professional University\pythonDS\1 PROJECT\bengaluru.csv")
df1.head()


# In[3]:


df1.shape


# In[4]:


df1.groupby('area_type')['area_type'].agg('count')  #group by area type and count the area type using aggretation


# In[5]:


#drop availability, areatype, society, balcony
df2=df1.drop(['area_type','society','balcony','availability'],axis='columns')
df2.head()


# In[6]:


#Data cleaning
df2.isnull().sum() #the no. of rows where the value is NaN


# In[7]:


#instead of dropping the rows, replace it with median
#when the rows are small in no. then just drop the rows
df3=df2.dropna()
df3.isnull().sum()


# In[8]:


df3.shape


# In[9]:


#exploring size column now
df3['size'].unique()


# In[10]:


#new column only for bhk values. using split function to split two tokens from space and taking the first token as the bhk value
#applying lambda function for the same on the 'size' column
df3['bhk']=df3['size'].apply(lambda x: int(x.split(' ')[0]))


# In[11]:


df3.head()


# In[12]:


df3['bhk'].unique()


# In[13]:


df3[df3.bhk>20]


# In[14]:


df3.total_sqft.unique()


# In[15]:


#converting range into single number using average
#variation? check is_float
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
#using this function, I'm checking if the value can be converted 
#into a float. If not, it means the value is not integer, instead
#it is a range


# In[16]:


#outliers
df3[~df3['total_sqft'].apply(is_float)].head()
#this gives the rows where the is_float function returned false
#means the rows that contain ranges (or not int) as total_sqft


# In[17]:


#OUTLIER/ NON-UNIFORMITY HANDLING

#changing ranges into single numbers
#to convert these values, i'm using convert_sqft_to_num function

def convert_sqft_to_num(x):
    tokens=x.split('-')
    if len(tokens)==2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


# In[18]:


convert_sqft_to_num('2100 - 2850') #checking the function


# In[19]:


convert_sqft_to_num('34.46Sq. Meter')


# In[20]:


df4=df3.copy() #deep copy
df4['total_sqft']=df4['total_sqft'].apply(convert_sqft_to_num)
df4.head()


# In[21]:


df4.loc[30]


# In[22]:


#feature engg and dimensional reductionality
df5=df4.copy()


# In[23]:


#price per sqft
df5['price_per_sqft']=df5['price']*100000/df5['total_sqft']
df5.head()


# In[24]:


#exploring location column
#it's just a categorical feature
df5.location.unique()


# In[25]:


len(df5.location.unique())


# In[26]:


#text data-->dummy column
#dimentionality curse
#other category
#finding datapoints

df5.location=df5.location.apply(lambda x: x.strip())
location_stats=df5.groupby('location')['location'].agg('count')
location_stats


# In[27]:


#sorting by no. of datapoints
location_stats=df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats


# In[28]:


#<10 datapoints= other location

len(location_stats[location_stats<=10])


# In[29]:


location_stats_less_than_10=location_stats[location_stats<=10]
location_stats_less_than_10


# In[30]:


len(df5.location.unique())


# In[31]:


df5.location=df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df5.location.unique())


# In[32]:


df5.head(10)


# In[33]:


#outlier detection and removal
#1. std deviation
#2. domain knowledge

#let the threshold be 300
df5[df5.total_sqft/df5.bhk<300].head()
#these are data errors/ anomalies
#these need to be removed


# In[34]:


df5.shape


# In[35]:


df6=df5[~(df5.total_sqft/df5.bhk<300)]
df6.shape


# In[36]:


#price per sqft
df6.price_per_sqft.describe()


# In[37]:


#removing pps outliers
def remove_pps_outliers(df):
    df_out=pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>(m-st))&(subdf.price_per_sqft<=(m+st))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out

df7= remove_pps_outliers(df6)
df7.shape


# In[38]:


def plot_scatter_chart(df,location):
    bhk2=df[(df.location==location) & (df.bhk==2)]
    bhk3=df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize']=(15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK',s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+',color='green',label='3 BHK',s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price Per Square Feet")
    plt.title(location)
    plt.legend()
plot_scatter_chart(df7,'Hebbal')


# In[39]:


#so there are many points where price for 2 bhk is higher than
#price for 3 bhk. Let's remove these outliers now

def remove_bhk_outliers(df):
    exclude_indices=np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats={}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk]={
                'mean':np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats=bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices= np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8=remove_bhk_outliers(df7)
df8.shape


# In[40]:


plot_scatter_chart(df8,'Hebbal')


# In[41]:


#how many apartments/properties are there in per sqft area
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")


# In[42]:


#this looks like normal distribution, which is good.
#now exploring bathroom feature
df8.bath.unique()


# In[43]:


df8[df8.bath>10]


# In[44]:


#outliers= when no. of bathrooms > no. of bedrooms + 2

plt.hist(df8.bath, rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")


# In[45]:


df8[df8.bath>df8.bhk+2]


# In[46]:


df9=df8[df8.bath<df8.bhk+2]
df9.shape


# In[47]:


#size and price per sqft features are unnecessary now so 
#so I will remove these

df10=df9.drop(['size','price_per_sqft'],axis=1)
df10.head()


# In[48]:


df10.to_csv('Processed_Bengaluru_Dataset.csv')


# In[49]:


#Model building
#As Machine learning model cannot interpret text data
#so we need to convert it into a numeric data
#using 1 hot n coding or dummies

dummies=pd.get_dummies(df10.location)
dummies.head(3)


# In[50]:


df11=pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
df11.head()
#to avoid dummy variable trap, I'm using 1 less dummies column


# In[51]:


#now dropping location column
df12=df11.drop('location',axis='columns')
df12.head(2)


# In[52]:


df12.shape


# In[53]:


#X variable --> independent variable (removing price column)
X = df12.drop('price',axis='columns')
X.head()


# In[54]:


Y=df12.price #dependent variable
Y.head()


# In[55]:


#For training and testing of model
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.2,random_state=10)
#test size=0.2 means 20% samples will be used for testing and 
#remaining 80% will be used for model training
X_test


# In[56]:


#linear regression model
from sklearn.linear_model import LinearRegression
lr_clf=LinearRegression()
lr_clf.fit(X_train,Y_train)  #training
lr_clf.score(X_test, Y_test) #score


# In[57]:


# K-fold cross validation
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv=ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, Y, cv=cv)


# In[58]:


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,Y):
    algos={
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'normalize': [ True, False]
            }
        },
        'lasso':{
            'model':Lasso(),
            'params':{
                'alpha':[1,2],
                'selection':['random','cyclic']
            }
        },
        'decision_tree':{
            'model':DecisionTreeRegressor(),
            'params': {
                'criterion':[ 'mse', 'friedman_mse'],
                'splitter':['best','random']
            }
        }
    }
    scores=[]
    cv=ShuffleSplit(n_splits=5, test_size=0.2,random_state=0)
    for algo_name, config in algos.items():
        gs=GridSearchCV(config['model'],config['params'],cv=cv, return_train_score=False)
        gs.fit(X,Y)
        scores.append({
            'model': algo_name,
            'best_score':gs.best_score_,
            'best_params':gs.best_params_
        })
    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,Y)


# In[59]:


#this function concludes that linear regression is the best model
#having the best parameter 'normalise': False

X.columns


# In[60]:


def predict_price(location,sqft,bath,bhk):
    loc_index = np.where(X.columns==location)[0][0]
    
    x=np.zeros(len(X.columns))
    x[0]=sqft
    x[1]=bath
    x[2]=bhk
    if loc_index>=0:
        x[loc_index]=1
        
    return lr_clf.predict([x])[0]


# In[61]:


predict_price('1st Phase JP Nagar',1000,2,2)


# In[62]:


predict_price('1st Phase JP Nagar',1000,2,3)


# In[63]:


predict_price('Indira Nagar',1000,2,2)


# In[64]:


predict_price('Indira Nagar',1000,3,3)


# In[65]:


import pickle
with open('bengaluru.pickle','wb') as f:
    pickle.dump(lr_clf,f)


# In[66]:


import json
columns={
    'data_cloumns': [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))


# In[67]:


predict_price('Electronic City Phase II',1000,2,1)


# In[68]:


predict_price('Uttarahalli',1000,2,1)


# In[69]:


from flask import Flask, render_template_string, request


# In[77]:


app = Flask(__name__)

# Load model and columns
model = pickle.load(open("bengaluru.pickle", "rb"))
import json

import json

# Load model
model = pickle.load(open("bengaluru.pickle", "rb"))

# Load columns from JSON
with open("columns.json", "r") as f:
    columns = json.load(f)["data_cloumns"]


# In[78]:


# Load locations from dataset
df = pd.read_csv("Processed_Bengaluru_Dataset.csv")
locations = sorted(df["location"].unique())
location_options = "\n".join([f'<option value="{loc}">{loc}</option>' for loc in locations])


# In[80]:


with open("index.html", "r") as file:
    html_template = file.read()


# In[81]:


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        location = request.form["location"]
        sqft = float(request.form["total_sqft"])
        bhk = int(request.form["bhk"])
        bath = int(request.form["bath"])

        x = np.zeros(len(columns))
        try:
            loc_index = columns.index(location.lower())
        except ValueError:
            loc_index = -1

        x[0] = sqft
        x[1] = bath
        x[2] = bhk
        if loc_index >= 0:
            x[loc_index] = 1

        prediction = round(model.predict([x])[0], 2)

    return render_template_string(html_template, location_options=location_options, prediction=prediction)
if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




