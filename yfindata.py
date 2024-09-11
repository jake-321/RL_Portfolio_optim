import pandas as pd
import yfinance as yf

tickers = (
    "COST",
    "CSCO", 
    "F",    
    "GS",   
    "AIG",   
    "AMGN", 
    "CAT" 
)

start_date = '2009-01-01'
end_date = '2024-01-01'


data = yf.download(tickers, start=start_date, end=end_date, interval='1d')

ohlc_data = data[['Open', 'High', 'Low', 'Adj Close']]


#print(ohlc_data.head())