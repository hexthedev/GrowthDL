import numpy as np
import pandas as pd

train_ds = pd.read_csv('./_data/Google_Stock_Price_Train.csv')
train_set = train_ds.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
train_set_scaled = sc.fit_transform(train_set)

x_train = []
y_train = []
for i in range(60,1258):
    x_train.append( train_set_scaled[i-60:i, 0] )
    y_train.append( train_set_scaled[i, 0] )

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1) ) 


# Build the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

regressor.add( LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1) ) )
regressor.add( Dropout( rate=0.2) )

regressor.add( LSTM(units=50, return_sequences=True) )
regressor.add( Dropout( rate=0.2) )

regressor.add( LSTM(units=50, return_sequences=True) )
regressor.add( Dropout( rate=0.2) )

regressor.add( LSTM(units=50) )
regressor.add( Dropout( rate=0.2) )

regressor.add( Dense(units=1) )

regressor.compile( optimizer='adam', loss='mean_squared_error' )


#Fitting the RNN to the training set
regressor.fit(x_train, y_train, epochs=1, batch_size=100)



# predict and visualize
test_data = pd.read_csv("./_data/Google_Stock_Price_Test.csv")
real_stock_price = test_data.iloc[:, 1:2].values

# Get pred
dataset_total = pd.concat((train_ds['Open'], test_data['Open']), axis=0)
inputs = dataset_total[ len(dataset_total) - len(dataset_total) -60: ].values
inputs = inputs.reshape(-1,1)
input = sc.transform(inputs)

x_test = []
for i in range(60, 80):
    x_test.append(inputs[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform( predicted_stock_price )

print(predicted_stock_price)