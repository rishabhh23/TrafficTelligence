import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn import svm
import xgboost
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

data = pd.read_csv(r"C:\Users\rish2\OneDrive\Desktop\ml project\traffic volume.csv")

# see the contents of the data
# print(data.info())
# print(data.columns)

# handling the missing values in the numeric
# columns of data (i.e temp,rain and snow)

# checking for missing data
# data.isnull().sum()

#fill the missing cells with the mean of the whole column
data['temp']=data['temp'].fillna(data['temp'].mean())
data['rain']=data['rain'].fillna(data['rain'].mean())
data['snow']=data['snow'].fillna(data['snow'].mean())

#visualizing the data
# sns.pairplot(data)
# plt.show()
data[["day","month","year"]] = data["date"].str.split("-", expand=True)
data[["hours","minutes","seconds"]] = data["Time"].str.split(":", expand=True)
data.drop(columns=['date','Time'],axis=1,inplace=True)
print(data.head())
# print(data.columns)

#splitting into independant and dependant variables
y=data['traffic_volume']
x=data.drop(columns=['traffic_volume'],axis=1)

#feature scaling 
names=x.columns
x=scale(x)
x=pd.DataFrame(x,columns=names)
print(x.head())

#splitting the data into train data and test data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


