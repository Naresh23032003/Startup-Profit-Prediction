import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

"""Data Collection"""

df = pd.read_csv("https://raw.githubusercontent.com/arib168/data/main/50_Startups.csv")

"""Data preparation"""

df.info()
#no null values found

df.State.replace(to_replace=["New York", "California", "Florida"], value=[-1,0,1], inplace=True) #Giving dummy values to categorical variable

"""Data visualisation"""

print(df.shape)

sns.distplot(df['Profit']) #distribution plot for profit

print(df['Profit'].max())

print(df['Profit'].min())

df.iloc[:,[0,1,2,4]].plot.area()

df.drop('State',axis=1).plot(kind='box') #box plot using pandas

sns.heatmap(df.corr(),annot=True)

scatter_matrix(df)

# from the above scatter matrix it can be seen that there is linear relation between variables profit and R&D spend , profit and Marketing spend, also there is no linear relationship between profit and Administration cost

"""Splitting the data"""

x = df.iloc[:, [0,1,3]] #admistration cost is not included in inputs
y = df.iloc[:, 4]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

"""Training the model"""

model = LinearRegression()
model.fit(x_train,y_train)

model.predict(x_train)

print(y_train)

"""Model evaluation"""

y_pred = model.predict(x_test)

# predicting the accuracy score
score=r2_score(y_test,y_pred)
print("r2 score = ",score)
print("mean_sqrd_error = ",mean_squared_error(y_test,y_pred))
print("root_mean_squared error of = ",np.sqrt(mean_squared_error(y_test,y_pred)))

#r2 score when admistration cost with other 3 parameters are included in input is 0.9348088470484867 (r2 score is reduced than the one in above cell is because there is no linear relationship between profit and Administration cost)
#r2 score when categorical variable/ State and admistration cost are not included in inputs is 0.9469407189577183 (including categorical variable with dummy values increases the r2 score which means increases the accuracy of the model)

"""Model prediction"""

print(len(y_pred))

print(y_pred, y_test)

# Best fit line
df1 = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
sns.regplot(x='Actual',y='Predicted',data=df1,color='red')