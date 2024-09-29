#importing all necessary packages
# also using yahoo finance API
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import streamlit as st
from keras.models import load_model

end = datetime.datetime.now()
start = '2014-12-31'

st.title('Stock Price Prediction')

id = st.text_input('Enter Stock Ticker', 'AMZN')
Amazon_Data = yf.download (id, start, end)

#Describing Data
st.subheader('data from 2014-2024')
st.write(Amazon_Data.describe())



#visualizations
st.subheader('Stock Data')

def plot_graphs(fig, values, col_name, subplot_index):
    plt.subplot(*subplot_index)
    values.plot()
    plt.xlabel('Years')
    plt.ylabel(col_name)
    plt.title(f'{col_name} Data of {id}')

# Create a figure with the desired size
fig = plt.figure(figsize=(32, 16))
num_subplots = 6
subplot_rows = 3
subplot_cols = 2  # Fixed number of columns

for index, column in enumerate(Amazon_Data.columns, start=1):
    # Calling function to plot each graph
    plot_graphs(fig, Amazon_Data[column], column, (subplot_rows, subplot_cols, index))
    plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)


st.subheader('Moving Average')
#moving average concept
def mov_avg(fig , values, val_1 , val_2 , col_name , subplot_index):
    plt.subplot(*subplot_index)
    values.plot(label= f'{col_name}')
    val_1.plot(label= 'MA-50',linewidth=2.5)
    val_2.plot(label= 'MA-100',linewidth=2)
    plt.xlabel('Years')
    plt.ylabel('Moving Average')
    plt.title(' Moving Average for 'f'{col_name}')

fig = plt.figure(figsize=(20, 16))
num_subplots =6
subplot_rows = 3
subplot_cols = 2

for index, column in enumerate(Amazon_Data.columns, start=1):
  ma_50 = Amazon_Data[column].rolling(50).mean()
  ma_100 = Amazon_Data[column].rolling(100).mean()
  #calling fxn
  mov_avg(fig , Amazon_Data[column] , ma_50 , ma_100, column , (subplot_rows, subplot_cols, index))
  plt.legend()
  plt.tight_layout()
# Display the plot in Streamlit
st.pyplot(fig)

#splitting data into training and testing
training = pd.DataFrame(Amazon_Data['Adj Close'][0:int(len(Amazon_Data)*0.75)])
testing = pd.DataFrame(Amazon_Data['Adj Close'][int(len(Amazon_Data)*0.75):int(len(Amazon_Data))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data_arr = scaler.fit_transform(training)

x_val = list()
y_val = list()

for val in range(80, len(scaled_data_arr)):
  x_val.append(scaled_data_arr[val-80:val])
  y_val.append(scaled_data_arr[val])

x_val= np.array(x_val)
y_val= np.array(y_val)

#load model
#@st.cache(allow_output_mutation=True)

model = load_model('Stock_Prices.keras')

prev_80 = training.tail(80)
final = pd.concat([prev_80, testing], ignore_index=True)
input_values = scaler.fit_transform(final)

x_test= list()
y_test = list()
for item in range(80, len(input_values)):
  x_test.append(input_values[item-80:item])
  y_test.append(input_values[item])

x_test= np.array(x_test)
y_test= np.array(y_test)

pred = model.predict(x_test)
final_pred = scaler.inverse_transform(pred)
y_test_inv = scaler.inverse_transform(y_test)


training_len=len(training)
final_plot = pd.DataFrame(
    {
        'original_test' : y_test_inv.reshape(-1),
        'prediction' : final_pred.reshape(-1)
    },
    index = Amazon_Data.index[training_len: ]
)
final_plot.reset_index(drop=True, inplace=True)


# Example function to plot the data
def final_graph(a, b, col_name):
    plt.plot(a, label='Original Test')
    plt.plot(b, label='Prediction')
    plt.xlabel('Years')
    plt.ylabel(col_name)
    plt.title(f'{col_name}')
    plt.legend()

st.write(final_plot.head())

st.subheader('Final Prediction')
figures = plt.figure(figsize=(16, 8))
final_graph(final_plot['original_test'], final_plot['prediction'], 'Test Data')
plt.tight_layout()
plt.legend()
plt.show()
st.pyplot(figures)