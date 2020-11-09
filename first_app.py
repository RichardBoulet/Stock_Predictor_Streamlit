# Testing sreamlit getting started guide

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly
from fbprophet import Prophet
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# Set the page title and description
st.title("Stock Prediction Application")
st.write("Here is the first attempt at the first ever fintech ml model, ever!! This program wshould accept a stock ticker entry and return the predicted stock closing price for the next 7 days.")


TODAY = datetime.date.today()

# Streamlit option for text entry into model
ticker = st.text_input('Enter ticker symbol:')


# Train model

train_start_date = '2020-01-01'
train_end_date = TODAY.strftime('%Y-%m-%d')

data = yf.download(ticker, train_start_date, train_end_date)


# Plot of the downloaded stock price data 
fig = go.Figure(data = [go.Candlestick(x = data.index,
                                       open = data['Open'],
                                       high = data['High'],
                                       low = data['Low'],
                                       close = data['Adj Close'])])

fig.update_layout(title = f'{ticker} Open, Low, High and Adjusted Closing Prices from {train_start_date} to {train_end_date}',
                  title_x = 0.5,
                  yaxis_title = f'{ticker} Prices')

# moving figure call to end of script
# fig.show()



# Create proper traiing dataframe for fbprophet framework
data_copy = (data.copy()
             .reset_index()
             .rename(columns = {'Date':'ds', 'Adj Close':'y'})
             .drop(columns = ['Open', 'High', 'Low', 'Close', 'Volume'], axis = 1)
)

model = Prophet()

model.fit(data_copy)


# Predictions

# Set prediction ranges
predict_start_date = TODAY  # Possibly use dynamic dates for this? want to deploy live model for this to retrain if needed.
predict_range = 7
predict_end_date = predict_start_date + datetime.timedelta(days = predict_range)

dates = (pd.date_range(start = predict_start_date.strftime('%m/%d/%Y'), end = predict_end_date.strftime('%m/%d/%Y'))
         .to_frame(index = False, name = 'ds')
         .reset_index(drop = True)
)

forecast = model.predict(dates)



# Plot the predicted stock prices for the date range
fig2 = go.Figure()

fig2.add_trace(go.Scatter(x = forecast.ds,
                         y = forecast.trend,
                         mode = 'lines+markers'))

fig2.update_layout(title = f'{ticker} Potential Price Range from {predict_start_date} to {predict_end_date}',
                  title_x = 0.5,
                  yaxis_title = f'{ticker} Predicted Price Range')

# Moving figure.show to bottom of script to run at end once predicted results are in
#fig2.show()



prediction_list = forecast.tail(predict_range).to_dict('records')


#Convert
output = {}

for data in prediction_list:
    date = data['ds'].strftime('%m/%d/%Y')
    output[date] = data['trend']



# Title and chart for Figure 1 Hstorical Prices
st.write(f'{ticker} Open, Low, High and Adjusted Closing Prices Interactive Chart')
st.plotly_chart(fig)



# Add dataframe to streamlit app frontpage
st.write(f'Predicted Closing Prices for {ticker}')
st.write(pd.DataFrame.from_dict(output, orient = 'index', columns = ['Predicted Closing Price']))



# Title and chart for Figure 2 Predicted Prices
st.write(f'{ticker} Predicted Prices Interactive Chart')
st.plotly_chart(fig2)


