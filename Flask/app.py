import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn import svm
from sklearn import metrics
import pickle
import xgboost
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
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
data['weather'].fillna('Clouds',inplace=True)
data.drop(columns=['holiday'], inplace=True)  # Modifies the original DataFrame

#visualizing the data
# sns.pairplot(data)
# plt.show()
data[["day","month","year"]] = data["date"].str.split("-", expand=True)
data[["hours","minutes","seconds"]] = data["Time"].str.split(":", expand=True)
data.drop(columns=['date','Time'],axis=1,inplace=True)
print(data.head())
# print(data.columns)

# Create a LabelEncoder object
le = LabelEncoder()
# Fit the LabelEncoder to the weather data (learn the categories)
le.fit(data['weather'])
# Transform the 'weather' column to numerical labels
data['weather'] = le.transform(data['weather'])

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
lin_reg = linear_model. LinearRegression()
Dtree = tree. DecisionTreeRegressor()
Rand = ensemble. RandomForestRegressor()
svr = svm. SVR( )
#XGB = xgboost . XGBRegressor ()
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#builds the model, takes time to execute around 5 minutes
lin_reg.fit(x_train,y_train)
Dtree.fit(x_train,y_train)
Rand.fit(x_train,y_train)
svr.fit(x_train,y_train)
#XGB.fit(x_train,y_train)
p1 = lin_reg.predict(x_train)
p2 = Dtree.predict(x_train)
p3 = Rand.predict(x_train)
p4 = svr.predict(x_train)
#p5 = XGB.predict(x_train)

print(metrics.r2_score(p1,y_train))
print(metrics. r2_score(p2,y_train))
print(metrics.r2_score(p3,y_train))
print(metrics. r2_score(p4,y_train))
#print(metrics. r2_score(p5, y_train))

#takes 1 minute to execute
p1 = lin_reg.predict(x_test)
p2 = Dtree.predict(x_test)
p3 = Rand.predict(x_test)
p4 = svr.predict(x_test)
print(metrics. r2_score(p1,y_test))
print(metrics. r2_score(p2,y_test))
print(metrics. r2_score(p3,y_test))
print(metrics.r2_score(p4,y_test))
#print(metrics. r2_score(p5,y_test))

#makes new files in the directory
pickle. dump(Rand, open("model.pk1", 'wb'))
pickle.dump(le, open("encoder.pk1", 'wb'))


