#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[39]:


df = pd.read_csv("MSFT.csv")


# In[40]:


df.tail()


# In[41]:


df.shape


# In[42]:


df.info()


# In[43]:


df = df[['Date','Close']]


# In[44]:


df.head()


# In[45]:


df['Date']


# In[46]:


import datetime

def str_to_datetime(s):
    split = s.split('-')
    year,month,day = int(split[0]),int(split[1]),int(split[2])
    return datetime.datetime(year=year,month=month,day=day)
    
df_object = str_to_datetime("1986-03-13")

df_object


# In[47]:


df['Date']=df['Date'].apply(str_to_datetime)


# In[48]:


df['Date']


# In[49]:


df.index = df.pop('Date')


# In[50]:


df


# In[51]:


plt.plot(df.index,df['Close'])
plt.show()


# In[ ]:





# In[61]:


def df_windowed(dataframe,first_date_str,last_date_str,n=3):
    first_date = str_to_datetime(first_date_str)
    last_date = str_to_datetime(last_date_str)
    
    target_date = first_date
    
    dates = []
    X,Y = [],[]
    
    last_time = False
    while True:
        df_subset = dataframe.loc[:target_date].tail(n+1)
        
        if len(df_subset)!= n+1:
            print(f'error: window of size {n} is too large for date {target_date}')
            return 
        
        values = df_subset['Close'].to_numpy()
        x,y = values[:-1],values[-1]
        
        dates.append(target_date)
        X.append(x)
        Y.append(y)
        
        
        next_week = dataframe.loc[target_date:target_date+datetime.timedelta(days=7)]
        next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
        next_date_str = next_datetime_str.split('T')[0]
        year_month_day = next_date_str.split("-")
        year,month,day = year_month_day
        next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))
        
        if last_time:
            break
            
        target_date = next_date
        
        
        if target_date == last_date:
            last_time = True
            
            
    ret_df = pd.DataFrame({})
    ret_df["Target Date"] = dates
    
    X = np.array(X)
    for i in range(0,n):
        X[:, i]
        ret_df[f'Target-{n-i}'] = X[:, i]
        
    ret_df['Target'] = Y
    
    return ret_df

# start day second time around 22nd september 2022
windowed_df = df_windowed(df,'2022-09-08',"2023-09-08",n=3)

windowed_df
        
        
    
        
        


# In[62]:


def windowed_df_x_y(windowed_dataframe):
    df_as_np = windowed_dataframe.to_numpy()

    dates = df_as_np[:,0]
    
    middle_matrix = df_as_np[:,1:-1]
    X = middle_matrix.reshape((len(dates),middle_matrix.shape[1],1))
    
    Y = df_as_np[:,-1]
    
    return dates,X.astype(np.float32),Y.astype(np.float32)

dates,X,y = windowed_df_x_y(windowed_df)

dates.shape,X.shape,y.shape


# In[63]:


q_80 = int(len(dates)*.8)
q_90 = int(len(dates)*.9)
dates_train, X_train,y_train = dates[:q_80],X[:q_80],y[:q_80]

dates_val,X_val,y_val = dates[q_80:q_90],X[q_80:q_90],y[q_80:q_90]
dates_test,x_test,y_test = dates[q_90:],X[q_90:],y[q_90:]

plt.plot(dates_train,y_train)
plt.plot(dates_val,y_val)
plt.plot(dates_test,y_test)

plt.legend(['Train','Validation','Test'])


# In[64]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers


# In[65]:


model = Sequential([layers.Input((3,1)),
                    layers.LSTM(64),
                   layers.Dense(32,activation = 'relu'),
                   layers.Dense(32,activation = 'relu'),
                   layers.Dense(1)])
model.compile(loss = 'mse',
             optimizer = Adam(learning_rate = 0.01),
             metrics = ['mean_absolute_error'])

model.fit(X_train,y_train,validation_data = (X_val,y_val),epochs = 100)


# In[66]:


train_predictions = model.predict(X_train).flatten()

plt.plot(dates_train,train_predictions)
plt.plot(dates_train,y_train)
plt.legend(['Training Predictions','Training Observations'])


# In[67]:


val_predictions = model.predict(X_val).flatten()

plt.plot(dates_val,val_predictions)
plt.plot(dates_val,y_val)
plt.legend(["Validation predictions","Validation Observations"])


# In[68]:


test_predictions = model.predict(x_test).flatten()

plt.plot(dates_test,test_predictions)
plt.plot(dates_test,y_test)
plt.legend(["testing predictions", "testing observations"])


# In[69]:


plt.plot(dates_train,train_predictions)
plt.plot(dates_train,y_train)
plt.plot(dates_val,val_predictions)
plt.plot(dates_val,y_val)
plt.plot(dates_test,test_predictions)
plt.plot(dates_test,y_test)
plt.legend(["training predictions",
          "training observations",
          "validation predictions",
          "validation observations",
          "testing predictions",
          "testing observations"])


# In[72]:


from copy import deepcopy

recursive_predictions = []
recursive_dates = np.concatenate([dates_val,dates_test])

for target_dates in recursive_dates:
    last_window = deepcopy(X_train[-1])
    next_prediction = model.predict(np.array([last_window])).flatten()
    recursive_predictions.append(next_prediction)
    last_window[-1]= next_prediction


# In[73]:


plt.plot(dates_train,train_predictions)
plt.plot(dates_train,y_train)
plt.plot(dates_val,val_predictions)
plt.plot(dates_val,y_val)
plt.plot(dates_test,test_predictions)
plt.plot(dates_test,y_test)
plt.plot(recursive_dates,recursive_predictions)
plt.legend(["training predictions",
          "training observations",
          "validation predictions",
          "validation observations",
          "testing predictions",
          "testing observations",
           "recursive predictions"])


# In[ ]:




