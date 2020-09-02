#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install pandas_datareader')
get_ipython().system('pip install sklearn')
get_ipython().system('pip install keras')
get_ipython().system('pip install tensorflow')


# In[2]:


#import libraries
import math
import pandas_datareader as web
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
plt.style.use('fivethirtyeight')

df = web.DataReader('AAPL',data_source='yahoo', start='2019-01-01', end='2020-12-06')
df


# In[3]:


df.shape


# In[4]:


#Visualising the Close price history
plt.figure(figsize=(14,7))
plt.title('CLOSE PRICE HISTORY')
plt.plot(df['Close'])
plt.xlabel('DATE',fontsize=19)
plt.ylabel('CLOSE PRICE HISTORY', fontsize=19)
plt.show()


# In[5]:


#Creating a dataframe with only Close values
data=df.filter(['Close'])
dataset=data.values
training_data_len=math.ceil( len(dataset)*.8)

#Scale all of the data to values 0 or 1(for convenience)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data= scaler.fit_transform(dataset) #fit_transform will subtract all values by the mean divided by st deviation

#Scaled training data set
train_data = scaled_data[0:training_data_len, : ]

#Splitting the data into x_train and y_train
x_train=[]
y_train=[]
for i in range (60, len(train_data)):
    x_train.append(train_data[i-60:i,0]) #the first column in the ‘x_train’ data set will contain values from the data set from index 0 to index 59 (60 values total) and the second column will contain values from the data set from index 1 to index 60 (60 values) and so on and so forth.
    y_train.append(train_data[i,0]) #‘y_train’ data set will contain the 61st value located at index 60 for it’s first column and the 62nd value located at index 61 of the data set for it’s second value and so on and so forth.
    
#Convert x_train and y_train in numpy arrays
x_train , y_train = np.array(x_train), np.array(y_train)

#Reshape the data to be 3-dimensional in the form [number of samples, number of time steps, and number of features]. The LSTM model is expecting a 3-dimensional data set.
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1], 1))

#Build the LSTM model to have two LSTM layers with 50 neurons and two Dense layers, one with 25 neurons and the other with 1 neuron.
#Build LSTM network
model = Sequential()
model.add(LSTM(units =50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the model using the training data sets. Note, fit is another name for train. Batch size is the total number of training examples present in a single batch, and epoch is the number of iterations when an entire data set is passed forward and backward through the neural network.
model.fit(x_train, y_train, batch_size=1, epochs=1)

#Test Data set
test_data=scaled_data[training_data_len - 60 : , :]

#create the x_data and y_data sets
x_test=[]
y_test= dataset[training_data_len: , : ]
for i in range (60, len(test_data)):
    x_test.append(test_data[i-60 : i,0])
    
# convert the independent test data set ‘x_test’ to a numpy array so it can be used for testing the LSTM model.
x_test = np.array(x_test) 

#Reshape the data to be 3-dimensional in the form [number of samples, number of time steps, and number of features]
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

#Getting the models predicted price values
predictions= model.predict(x_test)
predictions= scaler.inverse_transform(predictions) #Undo scaling

#Calculate/Get the value of RMSE
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
rmse


# In[6]:


#Plot/Create the data for the graph
train = data[:training_data_len]
valid = data[training_data_len:]

valid['Predictions'] = predictions

#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()


# In[7]:


#Show the valid and predicted prices
valid


# In[8]:


#Get the quote
apple_quote = web.DataReader('AAPL', data_source='yahoo', start='2019-01-01', end='2020-12-06')
#Create a new dataframe
new_df = apple_quote.filter(['Close'])
#Get teh last 60 day closing price 
last_60_days = new_df[-60:].values
#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
#Create an empty list
X_test = []
#Append teh past 60 days
X_test.append(last_60_days_scaled)
#Convert the X_test data set to a numpy array
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Get the predicted scaled price
pred_price = model.predict(X_test)
#undo the scaling 
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)


# In[ ]:




