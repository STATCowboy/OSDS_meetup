import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
# import statsmodels.tsa.stattools as tss
# import statsmodels.tsa.statespace.tools as ssm
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


loc = os.path.join('C:/Users/jj/OneDrive/Documents/GitHub/OSDS_meetup/07-18-2018_OSDS meetup_RNN-LSTM_Basics/data/') # this will put the directory data in variable loc

filenames = os.listdir(loc)

'''
create a for loop with pd.read_csv and add all read dataframes into the for loop
This will create a list of all dataframes

'''
weather_list=[pd.read_csv(loc+filenames[i], header=0,encoding="ISO-8859-2") for i in range(len(filenames))]

'''
We simply want to stack those datasets together and ignore original index

'''
weather_df=pd.concat(weather_list,ignore_index=True)

'''
I want to move Date column to be first column and change it to datetime and sort from oldest to newest

'''
weather_df=weather_df[['Date Time','CO2 (ppm)', 'H2OC (mmol/mol)', 'PAR (ľmol/m˛/s)',
       'SWDR (W/m˛)', 'T (degC)', 'Tdew (degC)', 'Tlog (degC)', 'Tpot (K)',
       'VPact (mbar)', 'VPdef (mbar)', 'VPmax (mbar)', 'max. PAR (ľmol/m˛/s)',
       'max. wv (m/s)', 'p (mbar)', 'rain (mm)', 'raining (s)', 'rh (%)',
       'rho (g/m**3)', 'sh (g/kg)', 'wd (deg)', 'wv (m/s)']]
weather_df['Date Time']=pd.to_datetime(weather_df['Date Time'])

weather_df=weather_df.sort_values('Date Time').reset_index(drop=True)


weather_df=weather_df.set_index(pd.DatetimeIndex(weather_df['Date Time']))

# weather_df=weather_df.drop('Unnamed: 0', axis=1)

'''
resample function will apply a groupby aggregation to the desired sampling period. in this case 60T will yield an hour
'''
weather_grpd=weather_df.resample('60T').median()

'''
Just in case we have any missing values, we will use forward fill to take care of that
'''
weather_grpd=weather_grpd.fillna(method='ffill')


weather_grpd


weather_grpd.info()

sns.lineplot(x=weather_grpd.iloc[100:230,:].index,y="T (degC)",data=weather_grpd.iloc[100:230,:])
plt.show()

fig=plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(211)
fig=sm.graphics.tsa.plot_acf(weather_grpd["T (degC)"].values.squeeze(),lags=94,ax=ax1)
plt.show()

print('Data Length:', len(weather_grpd))
print('number of rows to truncate:', len(weather_grpd)%24)
print("Total number of days:", len(weather_grpd)//24)

lengthindays=len(weather_grpd)//24
valdays=200
testdays=162
test_index=(lengthindays-testdays)*24 # taking the last 262 days to forecast on
val_index=(lengthindays-testdays-valdays)*24

scaler=StandardScaler()

'''We will fit on training data only, but then transform the entire data set'''
scaler=scaler.fit(weather_grpd.iloc[:val_index,:].values)

'''Transforming the entire dataset'''
weather_scaled=pd.DataFrame(scaler.transform(weather_grpd),columns=weather_grpd.columns,index=weather_grpd.index)

weather_scaled


def casual_observer(series,step):
    ypred=[series.iloc[i-step] for i in range(step,len(series))]
    actual=[series.iloc[i+step] for i in range(len(series)-step)]
    return (np.asarray(ypred).reshape(-1,1),np.asarray(actual).reshape(-1,1))

temp_naive,actual=casual_observer(weather_grpd['T (degC)'].iloc[val_index:test_index],24)
casual_error=mean_absolute_error(actual,temp_naive)
print('simple prediction Error:',casual_error)#/len(weather_grpd['T (degC)'].iloc[val_index:test_index]))


def seq_prep(df, y_var, hist, frcst, frcst_step, val, test):
       '''
       inputs:
       df: DataFrame that contains X, y variables
       y_var: y_var label, this will be a string
       hist: is an integer for the number of days lag
       frcst: is an integer for the number of days forward for forecast
       frcst_step: this is an integer for how many hours to forecast
       Val: is the index for Validation
       test: is the index for test

       outputs:
       train_seqx: List of Arrays containing the training X input. The shape per array should be (24*hist,number of columns in df)
       train_seqy: List of Arrays containing the training y (label), the shape per array should be (24*fore,)
       val_seqx: List of Arrays containing the valdiation X input. The shape per array should be (24*hist,number of columns in df)
       val_seqy: List of Arrays containing the validation y (label), the shape per array should be (24*fore,)
       test_seqx: List of Arrays containing the test X input. The shape per array should be (24*hist,number of columns in df)
       test_seqy: List of Arrays containing the test y (label), the shape per array should be (24*fore,)
       '''
       train_seqx = []
       train_seqy = []
       val_seqx = []
       val_seqy = []
       test_seqx = []
       test_seqy = []
       train_df = df.iloc[:val, :]
       val_df = df.iloc[val:test, :]
       test_df = df.iloc[test:, :]
       train_x_arr = train_df.values
       train_y_arr = train_df[y_var].values
       val_x_arr = val_df.values
       val_y_arr = val_df[y_var].values
       test_x_arr = test_df.values
       test_y_arr = test_df[y_var].values
       hist_seq = 24 * hist  # turning history into 24 hour x number of days vector
       fore_seq = 24 * frcst
       seq_len = (hist_seq + fore_seq)

       for i in range(len(train_y_arr)):
              if i + seq_len < len(train_y_arr):
                     train_seqx.append(train_x_arr[i:i + hist_seq, :])
                     train_seqy.append(train_y_arr[i + hist_seq:i + seq_len][:frcst_step])

              else:
                     val_index_add = i
                     break

       val_x_arr = np.insert(val_x_arr, 0, train_x_arr[val_index_add:, :], axis=0)
       val_y_arr = np.insert(val_y_arr, 0, train_y_arr[val_index_add:], axis=0)
       for i in range(len(val_y_arr)):
              if i + seq_len < len(val_y_arr):
                     val_seqx.append(val_x_arr[i:i + hist_seq, :])
                     val_seqy.append(val_y_arr[i + hist_seq:i + seq_len][:frcst_step])

              else:
                     test_index_add = i
                     break

       test_x_arr = np.insert(test_x_arr, 0, val_x_arr[test_index_add:, :], axis=0)
       test_y_arr = np.insert(test_y_arr, 0, val_y_arr[test_index_add:], axis=0)
       for i in range(len(test_y_arr)):
              if i + seq_len < len(test_y_arr):
                     test_seqx.append(test_x_arr[i:i + hist_seq, :])
                     test_seqy.append(test_y_arr[i + hist_seq:i + seq_len][:frcst_step])

       return train_seqx, train_seqy, val_seqx, val_seqy, test_seqx, test_seqy

train_X_list, train_y_list,val_X_list, val_y_list,test_X_list, test_y_list=seq_prep(weather_scaled,
                                                                                    "T (degC)",5,1,1,val_index,test_index)

train_X_list[0]


train_y_list[0]


print('Shape of input sequence:',train_X_list[0].shape)
print('Shape of output sequence:',train_y_list[0].shape)

from keras.preprocessing import sequence

'''Defining the sequence lengths for both input and output'''
X_seqlen=train_X_list[0].shape[0]
y_seqlen=train_y_list[0].shape[0]

'''
pad_sequences function is a very versatile function. it allows us to handle cases 
of uneven sequence length by padding or truncating to a specificed length.
Default pad value is 0, but user can apply any number or pass an impute into it

'''
train_X_seq = sequence.pad_sequences(train_X_list, dtype='float32', maxlen=X_seqlen, padding='post',truncating='post')
train_y_seq = sequence.pad_sequences(train_y_list,dtype='float32',maxlen=y_seqlen, padding='post',truncating='post')
val_X_seq = sequence.pad_sequences(val_X_list, dtype='float32', maxlen=X_seqlen, padding='post',truncating='post')
val_y_seq = sequence.pad_sequences(val_y_list,dtype='float32',maxlen=y_seqlen, padding='post',truncating='post')
test_X_seq = sequence.pad_sequences(test_X_list, dtype='float32', maxlen=X_seqlen, padding='post',truncating='post')
test_y_seq = sequence.pad_sequences(test_y_list,dtype='float32',maxlen=y_seqlen, padding='post',truncating='post')

'''WE got our Tensors'''
print('X shape:', train_X_seq.shape)
print('y shape:',train_y_seq.shape)

BATCH_SIZE=128
NUM_TIMESTEPS=train_X_seq.shape[1]
features=train_X_seq.shape[2]
output=train_y_seq.shape[1]

from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from keras.optimizers import RMSprop

model = Sequential()
'''The input to an LSTM layer has to be 3D tensor. The default LSTM layer output is 2D '''
model.add(LSTM(32, input_shape=(NUM_TIMESTEPS,features )))## The shape of each input sample is defined in 1st layer only
model.add(Dense(output))
model.compile(optimizer=RMSprop(), loss='mae')
model.summary()

history=model.fit(train_X_seq,train_y_seq,epochs=10,shuffle=False,batch_size=BATCH_SIZE,validation_data=(val_X_seq,val_y_seq))

import matplotlib.pyplot as plt
train_loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(train_loss))

plt.figure()

plt.plot(epochs, train_loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

mean_absolute_error(val_y_seq,model.predict(val_X_seq))

model2 = Sequential()
'''
Input to LSTM layer has to always be 3D, since the output is by default is 2D. We need to make 
return_sequences=True in order to stack a second LSTM layer. 
This will make the output of the 1st LSTM layer return a 3D output, which can now be input to the 2nd LSTM layer
dropout and Recurrent_dropout helps with regularizing the network and reduce overfitting. 
'''
model2.add(LSTM(32, input_shape=(NUM_TIMESTEPS,features ),dropout=0.3, recurrent_dropout=0.5,return_sequences=True))
model2.add(LSTM(64, input_shape=(NUM_TIMESTEPS,features ),dropout=0.3, recurrent_dropout=0.5))
model2.add(Dense(output))
model2.compile(optimizer=RMSprop(), loss='mae')
model2.summary()

history2=model2.fit(train_X_seq,train_y_seq,epochs=15,shuffle=False,batch_size=BATCH_SIZE,validation_data=(val_X_seq,val_y_seq))

train_loss = history2.history['loss']
val_loss = history2.history['val_loss']

epochs = range(len(train_loss))

plt.figure()

plt.plot(epochs, train_loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
