import pandas as pd
import numpy as np
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import requests
import time
import base64
import pytz
import math
import random
import cvxpy as cp
from pylab import *
from collections import Counter
# from dateutil.relativedelta import relativedelta
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from PIL import Image


st.set_page_config(page_title='Stock Index Dashboard', page_icon=':money_with_wings:', 
                   layout="wide", initial_sidebar_state="expanded")

image = Image.open(r"stock_market.jpg")
st.image(image)#, width=800
st.markdown('<h1 style=text-align: center;>Stock Index Visualization & Price Prediction App</h1>', unsafe_allow_html=True)

st.markdown("""
Stock indices are measures of the performance of a group of stocks that represent a particular market or sector. A stock index is calculated based on the performance of a selected group of stocks, and it provides a snapshot of the overall performance of the market or sector that the index represents. This app allows users to:

* Visualize closing prices and volumes of major stock indices around the world, where the list of indices is based on [Yahoo Finance](https://finance.yahoo.com/world-indices/)
* Construct the [Efficient Fronter](https://www.investopedia.com/terms/e/efficientfrontier.asp) curve through random sampling and simulating performances of portfolios, each of which consists of different indices
* Generate price predictions for stock indices based on historical closing prices. The neural network with [Multi-layer Perceptron regressor (MLP Regressor)](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html) from the Sklearn library was used to construct the prediction model

This app mainly utlizes the following Python libraries: Streamlit, Pandas, NumPy, Sklearn, Matplotlib, Seaborn, Plotly, yFinance. Users are advised to use [dark theme on Streamlit](https://blog.streamlit.io/introducing-theming/#:~:text=To%20toggle%20between%20various%20themes,is%20Streamlit%27s%20new%20dark%20theme) for better contrast and appearance. Please feel free to reach out, give feedback or comments on this app to me via [Linkedin](https://www.linkedin.com/in/hai-vu/), [Email](mailto:namhaivu97@gmail.com), or [GitHub](https://github.com/namhaivu173), I'd love any opportunity to connect, learn and improve.

""")

# Get date information
date_now = int(datetime.datetime.now(pytz.utc).strftime('%d'))
month_now = int(datetime.datetime.now(pytz.utc).strftime('%m'))
year_now = int(datetime.datetime.now(pytz.utc).strftime('%Y'))
current_date = datetime.date(year=year_now, month=month_now, day=date_now)

# Specify time range
time_before = datetime.date(year=min(2010,year_now-10), month=1, day=1) # Take 10 years ago or 2010, whichever is smaller
time_now = datetime.date(year=year_now, month=month_now, day=1) # Take first day of month of today's date
time_max = time_now - datetime.timedelta(weeks=1) # Take 1 weeks before time_now

st.sidebar.header('Specify Historical Data Time Range')

# Create two input fields for the start and end dates
time_start = st.sidebar.date_input("Start date", value=time_before, max_value=time_max, key='start')
time_end = st.sidebar.date_input("End date", value=time_now, max_value=current_date, key='end')

# Convert input strings to datetime objects
if time_start and time_end:
    time_start = pd.to_datetime(time_start.strftime('%Y-%m-%d'))
    time_end = pd.to_datetime(time_end.strftime('%Y-%m-%d'))

    # Check that the start date is before the end date
    if time_start >= time_end:
        st.error("Error: Start date must be earlier than end date.")
else:
    time_start = None
    time_end = None

# Data cleaning and processing
#########################################################

# Import indices description
idx_info = pd.read_csv('Indices_Description.csv')

# Get names of major world indices from yahoo (https://finance.yahoo.com/world-indices)
url = "https://finance.yahoo.com/world-indices/"
r = requests.get(url)
df_list = pd.read_html(r.text)
world_idx = df_list[0]

# Get dict of names and tickers
ticker_name = dict(zip(world_idx['Symbol'],world_idx['Name']))
ticker_name['^NZ50'] = 'S&P/NZX 50 INDEX GROSS'

# Specify time range
# time_start = datetime.date(year=min(2010,year_now-10), month=1, day=1) # Take first day and month of 10 years ago / or take first day of 2010
# time_end = datetime.date(year=year_now, month=month_now, day=1) # Take first day of month of today's date

# Extract the risk free rate (10-yr treasury yield)
treasury_10y = yf.Ticker('^TNX')
treasury_10y = treasury_10y.history(period='max') # Get annual risk free rate
treasury_10y.index = pd.to_datetime(treasury_10y.index.strftime("%Y-%m-%d"))
treasury_10y = treasury_10y[treasury_10y.index <= str(time_end)]['Close'].tail(1).iloc[0]
treasury_10y = treasury_10y/100

# Get 1Y, 2Y, 10Y treasury rates
df_treasury = yf.download(['^IRX', '^TNX', '^TYX'], start=time_start, end=time_end)['Close']
df_treasury = df_treasury.resample('D').ffill()
df_treasury.columns = ['1-Year', '10-Year', '20-Year']

# Extract all major tickers symbol, closing price and volume
def get_tickers(tickers, start=time_start, end=time_end):
    ticker_list = []
    for idx in tickers:
        ticker_data = yf.Ticker(idx)
        df_ticker = ticker_data.history(period='1d', start=start, end=end)
        df_ticker['Ticker'] = idx
        df_ticker = df_ticker[['Ticker','Close','Volume']]
        ticker_list.append(df_ticker)

    # Store data in a list
    df_tickers = pd.concat(ticker_list, axis=0).reset_index()

    # Convert Date column to datetime
    df_tickers['Date'] = pd.to_datetime(pd.to_datetime(df_tickers['Date'], utc=True).dt.strftime('%Y-%m-%d'))
    return df_tickers

# Extract tickers' prices
df_tickers = get_tickers(ticker_name.keys())

# Define region for each index
region_idx = {
  'US & Canada' : ['^GSPC', '^DJI','^IXIC', '^RUT','^GSPTSE','^NYA','^XAX','^VIX','^CASE30','^JN0U.JO'],
  'South & Latin America' : ['^BVSP', '^MXX', '^IPSA', '^MERV'], #, '^MERV'
  'ASEAN': ['^STI', '^JKSE', '^KLSE'],
  'Oceania & Middle East': ['^AXJO', '^NZ50', '^AORD'],
  'Other Asia': ['^N225', '^HSI', '000001.SS', '399001.SZ', '^TWII', '^KS11', '^BSESN', '^TA125.TA'],
  'Europe' : ['^FTSE', '^GDAXI', '^FCHI', '^STOXX50E','^N100', '^BFX', '^BUK100P','IMOEX.ME']
}

# Map region with tickers
def getRegion(ticker):
    for k in region_idx.keys():
        if ticker in region_idx[k]:
            return k

# Store region info of tickers into a list
region_lst = []
for ticker in df_tickers['Ticker']:
    region_lst.append(getRegion(ticker))
    

# Create region column from list created
df_tickers['Region'] = region_lst
df_tickers2 = df_tickers[df_tickers['Region'].notna()]

# See earliest and latest dates
df_minDates = df_tickers2.groupby(['Ticker'])['Date'].agg([np.min, np.max]).reset_index()

# Count number of tickers by the earliest dates (when the price data is available)
df_countDates = df_minDates.groupby('amin')['Ticker'].size().reset_index(name='Ticker_Count')

# Cumulative sum of count
df_countDates['CumSum'] = df_countDates['Ticker_Count'].cumsum()

# Proportion changes in count
df_countDates['Proportion'] = df_countDates['CumSum']/sum(df_countDates['Ticker_Count'])

# See what start date can be used as the reference date
# The date should include the most number of tickers without removing too much data (take 75% of proportion)
df_countDates['Majority'] = np.where(df_countDates['Proportion'] >= 0.75, True, False)    

# Get value of reference date
minDate = df_countDates[df_countDates['Majority']==True].iloc[0,0]

# Get name of tickers whose prices are not available on reference date
remove_ticks = list(df_minDates[df_minDates['amin'] > minDate]['Ticker'])

# Remove these tickers and filter for the reference date
df_tickers2 = df_tickers2[(~df_tickers2['Ticker'].isin(remove_ticks)) & (df_tickers2['Date'] >= minDate)]

# Get earliest price of each ticker and store output in a dict
refDate = min(df_tickers2['Date'])
def reference_dict(df_tickers, value, refDate):
    ref_value = {}
    for ticker in df_tickers['Ticker'].unique():
        start_value = df_tickers[(df_tickers['Ticker'] == ticker) & (df_tickers['Date'] >= refDate) & (df_tickers[value] != 0)]
        start_value = start_value.sort_values(by=['Date'], ascending=True)[value]
        if len(start_value) == 0:
            start_value = 0
        else:
            start_value = start_value.head(1).values
        ref_value[ticker] = float(start_value)
    return ref_value

ref_price = reference_dict(df_tickers2, 'Close', refDate)

# Get earliest volume of each ticker and store output in a dict
ref_vol = reference_dict(df_tickers2, 'Volume', refDate)

# Create new columns for price
df_tickers2['Ref_Price'] = df_tickers2['Ticker'].apply(lambda x: ref_price[x])
df_tickers2['Ref_Return'] = (df_tickers2['Close']/df_tickers2['Ref_Price'] - 1)*100
df_tickers2['Daily_Return'] = df_tickers2.groupby('Ticker')['Close'].pct_change(1)

# Create new columns for volume
df_tickers2['Ref_Volume'] = df_tickers2['Ticker'].apply(lambda x: ref_vol[x])
df_tickers2['Ref_VolChg'] = (df_tickers2['Volume']/df_tickers2['Ref_Volume'] - 1)*100

# Rotate df so that dates are index, tickers are header, rows are values
def rotate_df(df_tickers, value):
    # Turn Ticker to column names, Date to index, value to table values
    #df_return = df_tickers.groupby(['Date', 'Ticker'])[value].first().unstack()
    df_return = df_tickers.pivot(index='Date', columns='Ticker', values=value)
    
    # Fill NA with closest values in the future, if NA then use closest values in the past
    df_return = df_return.fillna(method='ffill').fillna(method='bfill')

    return df_return

df_refReturn = rotate_df(df_tickers2, 'Ref_Return')
df_refVolChg = rotate_df(df_tickers2, 'Ref_VolChg')
df_dayReturn = rotate_df(df_tickers2, 'Daily_Return')
df_dayClose = rotate_df(df_tickers2, 'Close')

# Calculate the pearson correlation coefficient between indices
corr_idx = df_dayReturn.corr(method='pearson')

# Calculate annualized return
ann_returns = (1+df_dayReturn.mean(skipna=True))**252-1

# Calculate indices covariances
cov_idx = df_dayReturn.cov()*252

# Remove region from dict region_idx that are no longer in df
def remove_ticker(region_idx, df):
    
    region_list = [item for sublist in region_idx.values() for item in sublist]
    region_remove = set(region_list) - set(df['Ticker'])
    
    for i in region_idx.values():
        for j in region_remove:
            try:
                i.remove(j)
            except ValueError:
                pass
            
    #print('List of tickers removed:',region_remove)
    return region_idx

# Dict with redundant regions removed
region_idx2 = remove_ticker(region_idx, df_tickers2)

# Generate simulated portfolios based on indices' mean return & variance
def mean_variance(df_dayReturn, max_return=None, n_indices=6, n_portfolios=5000, random_seed=99):
    
    # Calculate annualized returns for all indices
    ann_returns = (1 + df_dayReturn.mean(skipna=True))**252 - 1
    
    # Calculate covariances between all indices
    cov_idx = df_dayReturn.cov()*252
    
    # Set random generator
    np.random.seed(random_seed)
    
    # Initialize empty df to store mean-variance of portfolio
    df_mean_var = pd.DataFrame(columns=['expReturn','expVariance','weights','tickers'])

    # Initialize counter for number of valid portfolios
    num_valid_portfolios = 0

    # Loop through and generate lots of random portfolios
    while num_valid_portfolios < n_portfolios:
        # Choose assets randomly without replacement
        assets = np.random.choice(list(df_dayReturn.columns), n_indices, replace=False)
        
        # Choose weights randomly
        weights = np.random.rand(n_indices)
        
        # Ensure weights sum to 1
        weights = weights/sum(weights)
        
        # Initialize values of Return & Variance
        portfolio_expReturn = 0
        portfolio_expVariance = 0
        
        for i in range(n_indices):
            # Port return = sumproduct(weights, asset return)
            portfolio_expReturn += weights[i] * ann_returns.loc[assets[i]]
            
            for j in range(n_indices):
                # Port var = sumproduct(weight1, weight2, Cov(asset1,asset2))
                portfolio_expVariance += weights[i] * weights[j] * cov_idx.loc[assets[i], assets[j]]
        
        # Check if portfolio_expReturn is less than or equal to max_return
        if max_return is None or portfolio_expReturn <= max_return:
            # Append values of returns, variances, weights and assets to df
            df_mean_var.loc[num_valid_portfolios] = [portfolio_expReturn] + [portfolio_expVariance] + [weights] + [assets]
            num_valid_portfolios += 1
            
        elif portfolio_expReturn > max_return:
            continue
#         # Append values of returns, variances, weights and assets to df
#         df_mean_var.loc[num_valid_portfolios] = [portfolio_expReturn] + [portfolio_expVariance] + [weights] + [assets]
#         num_valid_portfolios += 1
    
    # Sharpe Ratio = (portfolio return - risk-free return) / (std.dev of portfolio return)
    df_mean_var['Sharpe_Ratio'] = (df_mean_var['expReturn'] - treasury_10y)/(df_mean_var['expVariance']**0.5)
            
    return df_mean_var

# Generate optimized-return portfolios based on indices' mean return & maximum variance
def optimize_return(df_dayReturn, max_variance=1, n_indices=6, n_portfolios=5000, random_seed=99):

    # Calculate annualized returns for all indices
    ann_returns = (1 + df_dayReturn.mean(skipna=True))**252 - 1
    
    # Calculate covariances between all indices
    cov_idx = df_dayReturn.cov()*252
    
    # Set random generator
    np.random.seed(random_seed)
    
    # Initialize empty df to store mean-variance of portfolio
    df_mean_var = pd.DataFrame(columns=['expReturn','expVariance','weights','tickers'])

    # Initialize counter for number of valid portfolios
    num_valid_portfolios = 0

    # Loop through and generate lots of random portfolios
    while num_valid_portfolios < n_portfolios:
        # Choose assets randomly without replacement
        assets = np.random.choice(list(df_dayReturn.columns), n_indices, replace=False)
        
        # Choose weights randomly
        weights = cp.Variable(n_indices)
        
        # Define objective function to maximize expected return
        objective = cp.Maximize(weights.T @ ann_returns[assets])
        
        count=0
        while count < 1:
            # Randomize variance constraint
            max_var = np.random.uniform(0, max_variance)
            
            # Compute expected return of the portfolio
            #exp_return = weights.T @ ann_returns[assets]
            
            # Define constraints for sum of weights = 1, weights > 0, and variance <= max_variance
            constraints = [cp.sum(weights) == 1,
                           cp.quad_form(weights, cov_idx.loc[assets, assets]) <= max_var,
                           weights >= 0]
                           #weights >= 0.0001]

            # Define problem and solve using cvxpy
            problem = cp.Problem(objective, constraints)

            try:
                problem.solve()
                if problem.status == 'optimal':
                    count += 1  # increment counter variable
            except:
                continue
        
        # Extract weights and calculate expected return and variance
        weights = weights.value
        portfolio_expReturn = np.sum(ann_returns[assets] * weights)
        portfolio_expVariance = np.dot(weights.T, np.dot(cov_idx.loc[assets, assets], weights))
        
        # Append values of returns, variances, weights and assets to df
        df_mean_var.loc[num_valid_portfolios] = [portfolio_expReturn] + [portfolio_expVariance] + [weights] + [assets]
        num_valid_portfolios += 1
    
    # Sharpe Ratio = (portfolio return - risk-free return) / (std.dev of portfolio return)
    df_mean_var['Sharpe_Ratio'] = (df_mean_var['expReturn'] - treasury_10y)/(df_mean_var['expVariance']**0.5)
            
    return df_mean_var
#########################################################

st.sidebar.header('Specify Simulation Parameters')
idx_options = list(df_dayReturn.columns)
n_indices = st.sidebar.slider('Number of assets per portfolio',2,len(idx_options)-1,11)
n_portfolios = st.sidebar.slider('Number of portfolios simulated',1000,50000,5000)

small_n = n_portfolios//2
large_n = n_portfolios - small_n

df_simulation1 = mean_variance(df_dayReturn, n_indices=n_indices, n_portfolios=large_n, max_return=0.5)
max_var1 = df_simulation1['expVariance'].max()
df_simulation2 = optimize_return(df_dayReturn, n_indices=n_indices, n_portfolios=small_n, max_variance=max_var1)

# slider_minreturn1 = max(df_simulation1['expReturn'].min(),0)

# max_return = st.sidebar.slider('Maximum return constraint', 
#                                float(slider_minreturn1), 
#                                df_simulation1['expReturn'].max(), 
#                                float(df_simulation1['expReturn'].max()))

#########################################################

# Store results of simulations
#df_simulation = mean_variance(df_dayReturn, n_indices=n_indices, n_portfolios=n_portfolios) #, max_return=max_return
df_simulation = pd.concat([df_simulation1, df_simulation2], axis=0)

# Plot 1 settings (price chg)
ref_year = min(df_refReturn.index).strftime('%Y')
# max_colors = max(len(x) for x in region_idx2.values())
# time_plot1 = minDate - relativedelta(months=6)
# time_plot2 = time_end + relativedelta(months=6)

# Round up function with decimals
def my_ceil(a, precision=0):
    return np.round(a + 0.5 * 10**(-precision), precision)

# Round down function with decimals
def my_floor(a, precision=0):
    return np.round(a - 0.5 * 10**(-precision), precision)

# Lowest risk portfolio
df_minrisk = df_simulation.sort_values(by=['expVariance']).head(1)
df_minrisk_port = pd.DataFrame({'tickers': df_minrisk['tickers'].iloc[0], 'weights': df_minrisk['weights'].iloc[0]})

# Lowest return portfolio
df_minreturn = df_simulation.sort_values(by=['expReturn']).head(1)

# Highest risk portfolio
df_maxrisk = df_simulation.sort_values(by=['expVariance'], ascending=False).head(1)

# Highest return portfolio
df_maxreturn = df_simulation.sort_values(by=['expReturn'], ascending=False).head(1)
df_maxreturn_port = pd.DataFrame({'tickers': df_maxreturn['tickers'].iloc[0], 'weights': df_maxreturn['weights'].iloc[0]})

# Highest risk-adjusted return
df_maxadj = df_simulation.sort_values(by=['Sharpe_Ratio'], ascending=False).head(1)
df_maxadj_port = pd.DataFrame({'tickers': df_maxadj['tickers'].iloc[0], 'weights': df_maxadj['weights'].iloc[0]})

# Calculate the pearson correlation coefficient between indices
df_dayReturn_max = df_tickers2[df_tickers2['Ticker'].isin(list(df_maxadj_port['tickers']))]
df_dayReturn_max = rotate_df(df_dayReturn_max, 'Daily_Return')
corr_idx2 = df_dayReturn_max.corr(method='pearson')

# Delta x and y of security market line
del_x = np.array([0, df_maxadj['expVariance'].iloc[0]**0.5])
del_y = np.array([treasury_10y, df_maxadj['expReturn'].iloc[0]])

# Calculate the slope of the security market line
slope = np.polyfit(del_x, del_y, 1)[0]
maxreturn_y = slope*(df_maxreturn['expVariance'].iloc[0]**0.5) + treasury_10y
maxrisk_y = slope*(df_maxrisk['expVariance'].iloc[0]**0.5) + treasury_10y

# Calculate the angle between the line and the x-axis
my_angle = np.rad2deg(np.arctan(slope))

#########################################################

st.write('## Stock Indices Description')
with st.expander('CLICK HERE FOR MORE INFORMATION'):
    st.table(idx_info)

# Plot 1
#plt.style.use('classic')
#plt.style.use('_mpl-gallery')

st.write('## Stock Indices Historical Data')
#max_cols = len(region_idx2.keys()) // 3
midpoint = len(region_idx2) // 2

with st.expander('1 - INDEX HISTORICAL CLOSING PRICES', expanded=False):
    col1, col2 = st.columns(2)
    i = 0
    for key, value in region_idx2.items():
        if i < midpoint:
            with col1:
                st.write(f'<b>{key}</b>', unsafe_allow_html=True)
                st.line_chart(df_dayClose[region_idx2[key]])
        else:
            with col2:
                st.write(f'<b>{key}</b>', unsafe_allow_html=True)
                st.line_chart(df_dayClose[region_idx2[key]]) 
        i += 1

# Plot 2
with st.expander('2 - PRICE CHANGES WITH RESPECT TO START DATE', expanded=False):
    col1, col2 = st.columns(2)
    i = 0
    for key, value in region_idx2.items():
        if i < midpoint:
            with col1:
                st.write(f'<b>{key}</b>', unsafe_allow_html=True)
                st.line_chart(df_refReturn[region_idx2[key]])
        else:
            with col2:
                st.write(f'<b>{key}</b>', unsafe_allow_html=True)
                st.line_chart(df_refReturn[region_idx2[key]]) 
        i += 1

# Plot 3
with st.expander('3 - TRADING VOLUME CHANGES WITH RESPECT TO START DATE', expanded=False):
    col1, col2 = st.columns(2)
    i = 0
    for key, value in region_idx2.items():
        if i < midpoint:
            with col1:
                st.write(f'<b>{key}</b>', unsafe_allow_html=True)
                st.line_chart(df_refVolChg[region_idx2[key]])
        else:
            with col2:
                st.write(f'<b>{key}</b>', unsafe_allow_html=True)
                st.line_chart(df_refVolChg[region_idx2[key]]) 
        i += 1

# Plot 4
with st.expander('4 - CLOSING PRICE DISTRIBUTION BOXPLOTS', expanded=False):
    fig3, axes = plt.subplots(nrows=len(region_idx)//2, ncols=2, figsize=(15, 10)) # Adjust figure size as needed

    axes = axes.flatten()

    for i, region in enumerate(region_idx2.keys()):
        sns.boxplot(x='Ticker', y='Close', data=df_tickers2[df_tickers2['Region']==region], ax=axes[i])
        axes[i].set_title(region, fontweight='bold') # Set title for each subplot
        axes[i].set_ylabel('')
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', labelsize=10)
        axes[i].grid(linestyle='dashed')
    fig3.text(0.5, -0.01, 'Tickers', ha='center', fontweight ="bold", fontsize=12)
    fig3.text(0,0.5, "Closing Prices ($)\n", ha="center", va="center", rotation=90, fontweight ="bold", fontsize=12)
    fig3.suptitle("Boxplots showing price distributions of World Major Indices", fontweight ="bold", y=1, fontsize=16)
    fig3.patch.set_facecolor('#C7B78E')
    fig3.tight_layout() # Adjust subplot spacing
    st.pyplot(fig3, use_container_width=True)

# Plot 5
with st.expander('5 - TRADING VOLUME DISTRIBUTION BOXPLOTS', expanded=False):
    fig4, axes = plt.subplots(nrows=len(region_idx)//2, ncols=2, figsize=(15, 10)) # Adjust figure size as needed

    axes = axes.flatten()

    for i, region in enumerate(region_idx2.keys()):
        plot_region = df_tickers2[df_tickers2['Region']==region]
        sns.boxplot(x=plot_region['Ticker'], y=plot_region['Volume']/1000000, data=plot_region, ax=axes[i])
        axes[i].set_title(region, fontweight='bold') # Set title for each subplot
        axes[i].set_ylabel('')
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', labelsize=10)
        axes[i].grid(linestyle='dashed')
    fig4.text(0.5, -0.01, 'Tickers', ha='center', fontweight ="bold", fontsize=12)
    fig4.text(0,0.5, "Trading Volumes\n", ha="center", va="center", rotation=90, fontweight ="bold", fontsize=12)
    fig4.text(0.012,0.5, "(Unit: 1,000,000)\n", ha="center", va="center", rotation=90, fontstyle ="italic", fontsize=10)
    fig4.suptitle("Boxplots showing trading volume distributions of World Major Indices", fontweight ="bold", y=1, fontsize=16)
    fig4.patch.set_facecolor('#C7B78E')
    fig4.tight_layout() # Adjust subplot spacing
    st.pyplot(fig4, use_container_width=True)

# Plot 6
with st.expander("6 - CORRELATION MATRIX OF INDICES' DAILY RETURNS", expanded=False):

    # Create heatmap to visualize correlation matrix for indices
    fig5 = plt.figure(figsize=(12,8))
    fig5.suptitle("Correlation matrix of indices' daily returns\n", y=0.93, fontsize = 16, fontweight ="bold")
    sns.set(font_scale=0.8)
    sns.set_theme(style='white')
    g = sns.heatmap(corr_idx, annot=True, cmap="RdBu", annot_kws={"fontsize":8},
                    vmin=-1, vmax=1, fmt='.1f', mask=np.triu(corr_idx, k=1))

    # Set color bar title
    g.collections[0].colorbar.set_label("\nCorrelation Level", fontsize=14)

    # Set the label size
    g.collections[0].colorbar.ax.tick_params(labelsize=12)

    # Set axis titles
    g.set_xlabel("Tickers", fontsize=14)
    g.set_ylabel("Tickers", fontsize=14)
    g.set_facecolor('lightgray')

    #plt.text(16, -0.35, "(0.7 <= Absolute correlation < 1)", ha="center", va="center", fontsize=12, fontstyle ="italic")
    xticks = g.set_xticklabels(g.get_xticklabels(), fontsize=10, rotation=45, ha='right') 
    yticks = g.set_yticklabels(g.get_yticklabels(), fontsize=10)
    
    fig5.patch.set_facecolor('#C7B78E')
    fig5.tight_layout()
    st.pyplot(fig5, use_container_width=True)

#st.text("")
st.write('## Portfolio Efficient Frontier Simulation')

with st.expander('READ MORE ABOUT THE EFFICIENT FRONTIER', expanded=True):
    st.write("""
    The Efficient Frontier is a concept in [Modern Portfolio Theory (MPT)](https://www.investopedia.com/terms/m/modernportfoliotheory.asp) that describes the set of optimal portfolios that offer the highest expected return for a given level of risk or the lowest risk for a given level of expected return. The efficient frontier is derived by analyzing the risk and return of various portfolios composed of different combinations of assets. It, however, makes several assumptions about the market:
    - Investors are rational and risk-averse: Investors seek to maximize their returns while minimizing their risk. They are willing to accept some degree of risk in exchange for higher expected returns
    - Markets are efficient: Asset prices reflect all available information, and investors cannot consistently earn excess returns by analyzing publicly available information
    - Returns are normally distributed: The returns on individual assets are assumed to follow a normal distribution, and the correlation between assets is taken into account when constructing portfolios
    - Investors have access to the same information: Investors have access to the same information and make rational decisions based on that information
    - Investors can borrow and lend at the risk-free rate: Investors can borrow and lend at a risk-free rate, which is typically the interest rate on government bonds
    
    The main idea behind constructing the Efficient Frontier is that by combining assets with different risk and return characteristics, investors can construct a portfolio that offers the best balance of risk and return for their investment goals. Apart from the efficient frontier, the [Security Market Line (SML)](https://www.investopedia.com/terms/s/sml.asp) is also plotted, which can help determine whether an investment product would offer a favorable expected return compared to its level of risk.
        """)

st.text("")
fig6 = go.Figure()

fig6.add_trace(go.Scatter(x=df_simulation['expVariance']**0.5, # x-axis = annualized std.dev (volatility) 
                         y=df_simulation['expReturn'],  # y-axis = annualized returns
                         # Add color scale for sharpe ratio 
                         marker=dict(color=df_simulation['Sharpe_Ratio'], 
                                     showscale=True,
                                     size=7,
                                     line=dict(width=1),
                                     colorscale="Agsunset",
                                     colorbar=dict(title="Sharpe<br>Ratio")
                                    ), 
                         mode='markers',
                         showlegend=False))

# Plot Security Market Line (SML)
fig6.add_trace(go.Scatter(x=[0, df_maxrisk['expVariance'].iloc[0]**0.5], # x-axis = annualized std.dev (volatility) 
                         y=[treasury_10y, maxrisk_y], # y-axis = annualized returns
                         # Add color scale for sharpe ratio 
                         line=dict(
                             color='darkblue',
                             width=2,
                             dash='solid'), 
                         mode='lines',
                         name='Security Market Line',
                         showlegend=True))
# Plot risk-free rate
fig6.add_trace(go.Scatter(x=[0], 
                         y=[treasury_10y],
                         marker=dict(
                             color='darkorange',
                             size=12), 
                         mode='markers',
                         name='Risk-free Rate (' + str(round(treasury_10y*100,2)) + '%)',
                         showlegend=True))

# Plot min risk port
fig6.add_trace(go.Scatter(x=df_minrisk['expVariance']**0.5,
                         y=df_minrisk['expReturn'],  
                         # Add color scale for sharpe ratio 
                         marker=dict(
                             color='green',
                             size=12), 
                         mode='markers',
                         name='Lowest Volatility Portfolio',
                         showlegend=True))

# Plot max return port
fig6.add_trace(go.Scatter(x=df_maxreturn['expVariance']**0.5, 
                         y=df_maxreturn['expReturn'], 
                         # Add color scale for sharpe ratio 
                         marker=dict(
                             color='black',
                             size=12), 
                         mode='markers',
                         name='Highest Return Portfolio',
                         showlegend=True))

# Plot max shorpe ratio port
fig6.add_trace(go.Scatter(x=df_maxadj['expVariance']**0.5, 
                         y=df_maxadj['expReturn'], 
                         # Add color scale for sharpe ratio 
                         marker=dict(
                             color='darkred',
                             size=12), 
                         mode='markers',
                         name='Highest Sharpe Ratio Portfolio',
                         showlegend=True))
# Plot max return line
fig6.add_hline(y=df_maxadj['expReturn'].iloc[0], line_width=2, 
              line_dash="dot", line_color="black")

fig6.add_hline(y=df_maxadj['expReturn'].iloc[0]*1.01, line_width=1, 
              line_dash="dot", line_color="white",
              annotation_text="<b>Highest Sharpe Ratio</b>", 
              annotation_position="top right", # position of text
              annotation_font_color="black",
              annotation_font_size=13)

# Plot min var line
fig6.add_vline(x=df_minrisk['expVariance'].iloc[0]**0.5, line_width=2, 
              line_dash="dot", line_color="black")

fig6.add_vline(x=df_minrisk['expVariance'].iloc[0]**0.5*0.97, line_width=2, 
              line_dash="dot", line_color="white",
              annotation_text="<b>Lowest volatility<br></b>", 
              annotation_position="top left", # position of text
              annotation_font_color="black",
              annotation_font_size=13,
              annotation_textangle=-90)

# Plot risk-free line
fig6.add_vline(x=0, line_width=2, 
              line_dash="dot", line_color="black")

fig6.add_vline(x=-0.001, line_width=1, 
              line_dash="dot", line_color="white",
              annotation_text="<b>Risk-free line<br></b>", 
              annotation_position="top left", # position of text
              annotation_font_color="black",
              annotation_font_size=13,
              annotation_textangle=-90)

# Add text to SML (position of annotations above SML)
x_mean = round(df_maxadj['expVariance'].iloc[0]**0.5/2,2)
y_mean = my_ceil((df_maxadj['expReturn'].iloc[0] - treasury_10y)/2+treasury_10y,2)*1.1

# fig6.add_annotation(x=x_mean,  # x-coord of width
#                    y=y_mean,  # percentage of height
#                    text="<b>The Security Market Line</b>",
#                    font=dict(color='darkblue',size=13),
#                    textangle=-my_angle,  # set the angle of the text to the calculated angle
#                    xanchor='center', xref='x', yref='y',
#                    showarrow=False,
#                    arrowhead=0)

# Add title/labels
fig6.update_layout(template='simple_white',
                  xaxis=dict(title='Annualized Risk (Volatility)'),
                  yaxis=dict(title='Annualized Return'),# scaleanchor="x", scaleratio=1),
                  #yaxis_range=[min(-1,-my_ceil(df_maxreturn['expReturn'].iloc[0])),my_ceil(df_maxreturn['expReturn'].iloc[0],1)*2],
                  title='<b>Efficient Frontier - Simulations of ' + str(len(df_simulation)) + ' Random Portfolios</b><br><i>(each consists of up to ' 
                        + str(n_indices) + ' indices)</i>',
                  title_x=0.5,
                  coloraxis_colorbar=dict(title="Sharpe Ratio"),
                  plot_bgcolor='rgb(211, 211, 211)',
                  legend=dict(
                      x=1,
                      y=0,
                      entrywidth=0,
                      entrywidthmode='pixels',
                      bordercolor='black',
                      borderwidth=1,
                      xanchor='right',
                      yanchor='bottom',
                      bgcolor='rgb(211, 211, 211)',
                      font=dict(color='black')),
                  width=1200, height=700)

st.plotly_chart(fig6, use_container_width=True, theme=None)


st.text("")
with st.expander('PORTFOLIO ASSET DISTRIBUTION & PERFORMANCE', expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("#### Minimum Risk Portfolio")
        st.table(df_minrisk_port.sort_values(by='weights', ascending=False).reset_index(drop=True))
        st.write("#### Minimum Risk Performance")
        df_minrisk_score = df_minrisk[['expReturn','expVariance','Sharpe_Ratio']].reset_index(drop=True)
        df_minrisk_score['expVariance'] = df_minrisk_score['expVariance']**0.5
        df_minrisk_score = df_minrisk_score.T.rename(columns={0:'Score'},
                                                     index={'expReturn':'Return','expVariance':'Volatility'})
        st.table(df_minrisk_score)

    with c2:
        st.write("#### Maximum Return Portfolio")
        st.table(df_maxreturn_port.sort_values(by='weights', ascending=False).reset_index(drop=True))
        st.write("#### Maximum Return Performance")
        df_maxreturn_score = df_maxreturn[['expReturn','expVariance','Sharpe_Ratio']].reset_index(drop=True)
        df_maxreturn_score['expVariance'] = df_maxreturn_score['expVariance']**0.5
        df_maxreturn_score = df_maxreturn_score.T.rename(columns={0:'Score'},
                                                         index={'expReturn':'Return','expVariance':'Volatility'})
        st.table(df_maxreturn_score)

    with c3:
        st.write("#### Highest Sharpe Ratio Portfolio")
        st.table(df_maxadj_port.sort_values(by='weights', ascending=False).reset_index(drop=True))
        st.write("#### Highest Sharpe Ratio Performance")
        df_maxadj_score = df_maxadj[['expReturn','expVariance','Sharpe_Ratio']].reset_index(drop=True)
        df_maxadj_score['expVariance'] = df_maxadj_score['expVariance']**0.5
        df_maxadj_score = df_maxadj_score.T.rename(columns={0:'Score'},
                                                   index={'expReturn':'Return','expVariance':'Volatility'})
        st.table(df_maxadj_score)

#st.text("")
with st.expander('OTHER RELEVANT INFORMATION', expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        st.write("#### US Treasury Yield by maturity (%)", unsafe_allow_html=True)
        st.text("")
        st.line_chart(df_treasury)
    with c2:
        st.write("#### Correlation matrix of daily returns of indices in the highest Sharpe Ratio portfolio")
        fig6 = plt.figure(figsize=(12,6))
        fig6.suptitle("", y=0.93, fontsize = 16, fontweight ="bold")
        sns.set(font_scale=1)
        sns.set_theme(style='white')
        g = sns.heatmap(corr_idx2, annot=True, cmap="RdBu", annot_kws={"fontsize":12},
                        vmin=-1, vmax=1, fmt='.2f', mask=np.triu(corr_idx2, k=1))

        # Set color bar title
        g.collections[0].colorbar.set_label("\nCorrelation Level", fontsize=14)

        # Set the label size
        g.collections[0].colorbar.ax.tick_params(labelsize=12)

        # Set axis titles
        g.set_xlabel("Tickers", fontsize=14)
        g.set_ylabel("", fontsize=14)
        g.set_facecolor('lightgray')
        xticks = g.set_xticklabels(g.get_xticklabels(), fontsize=10, rotation=45, ha='right') 
        yticks = g.set_yticklabels(g.get_yticklabels(), fontsize=10, rotation=0)

        fig6.patch.set_facecolor('#C7B78E')
        fig6.tight_layout()
        st.pyplot(fig6, use_container_width=True)

    
# Price Prediction
#########################################################

st.write('## Stock Indices Price Prediction')

ticker_lists = list(sort(list(ticker_name.keys())))

st.write('### Specify Prediction Parameters')
c1, c2 = st.columns(2)
with c1:
    pick_ticker = st.selectbox('Select ticker to make price prediction:', 
                               ticker_lists,ticker_lists.index('^GSPC'))
    st.markdown(f'You have selected: **{ticker_name[pick_ticker]}**')
with c2:
    pred_rows = st.slider('Select length of lookback period (in days) for training:',5,252,30)
    st.write('The lookback period in this case refers to the number of days in the past whose price will be used as training data to predict closing price of the next day. In this case, you have chosen ', pred_rows, '-day lookback period, meaning that the price of day ', pred_rows+1, 'will be predicted based on prices from the previous ', pred_rows, ' days')
    
#########################################################
st.write('### Prediction Model Outputs')

# Pick one ticker to predict
#pick_ticker = '^GSPC'
df_ticks = yf.Ticker(pick_ticker)

# Get name of chosen ticker
ticker_chosen = ticker_name[pick_ticker]

# Get closing price of ticker
#df_ticks = df_ticks.history(period='1d', start=time_start, end=time_end)['Close'].reset_index()
df_ticks = df_ticks.history(period='max')['Close'].reset_index()

# Convert Date to datetime and use as table index
df_ticks['Date'] = pd.to_datetime(pd.to_datetime(df_ticks['Date'], utc=True).dt.strftime('%Y-%m-%d'))
df_ticks = df_ticks.set_index(df_ticks['Date'])
df_ticks = df_ticks.drop('Date',axis=1)

# Choose scaler
std_scaler = StandardScaler()

# Apply scaler
scaled_data = std_scaler.fit_transform(df_ticks.values)

# Split 80/20
train_rows = math.ceil(len(df_ticks)*.8)
df_train = scaled_data[0:train_rows]
df_test = scaled_data[train_rows:len(df_ticks)]

# Function that split a dataframe based on look_back value
def lookback_split(dataset, look_back=30):
    data_X, data_Y = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back)]
        data_X.append(a)
        data_Y.append(dataset[i + look_back])
    return np.array(data_X), np.array(data_Y)

# Assuming df_train and df_test are 1D arrays of daily closing prices
#pred_rows = 30
x_train, y_train = lookback_split(df_train, look_back=pred_rows)
x_test, y_test = lookback_split(df_test, look_back=pred_rows)

# Reshape arrays
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1])
y_train = y_train.reshape(y_train.shape[0])
y_test = y_test.reshape(y_test.shape[0])

# Construct neural network using Multi-layer Perceptron regressor
model = MLPRegressor(hidden_layer_sizes=(20, 20), activation='relu', solver='adam', max_iter=1000, random_state=99)
model.fit(x_train, y_train)

# Create predictions
y_pred = model.predict(x_test)

# Inverse the scaling process and reshape arrays into columns
y_test = std_scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred = std_scaler.inverse_transform(y_pred.reshape(-1, 1))

# Show prediction scores
def mape_score(y_true, y_pred):
    """
    Calculate the mean absolute percentage error (MAPE) between two arrays
    """
    return np.mean(np.abs((y_true - y_pred) / y_true))

rmse_test = mean_squared_error(y_test, y_pred, squared=False)
mape_test = mape_score(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)

# Save performance score to df
df_score = pd.DataFrame([{'RMSE': rmse_test, 'MAPE': mape_test, 'R_Squared': r2_test}]).T
df_score.columns = ['Model Score']

# Save train, test, pred data into a df for plotting
train_price = df_ticks[:(train_rows+pred_rows)]
test_price = df_ticks[(train_rows+pred_rows):]
pred_price = pd.DataFrame(y_pred, columns=['Predictions'])
pred_price = pred_price.set_index(test_price.index)
pred_price['Date'] = pred_price.index
pred_price['Close'] = y_test

# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="Price_Predictions.csv">Download Price Prediction Outputs</a>'
    return href


#########################################################

c1, c2 = st.columns(2)
with c1:
    # Comparing predictions to testing prices
    st.write("#### Training, Testing & Predicted price chart")
    fig7, ax = plt.subplots(figsize=(12,8))
    ax.plot(train_price['Close'],linewidth=2)
    ax.plot(pred_price[['Close','Predictions']],linewidth=2)

    ax.set_title(ticker_chosen + ' price predictions (' + str(pred_rows) + '-day lookback period)\n', fontsize=17, fontweight="bold")
    ax.set_xlabel('\nTime', fontsize=15)
    ax.set_ylabel('Closing Price\n', fontsize=15)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(['Train','Test','Prediction'], loc='upper left', fontsize=15)
    ax.set_facecolor('lightgray')
    fig7.patch.set_facecolor('#C7B78E')
    fig7.tight_layout()
    st.pyplot(fig7, use_container_width=True)
with c2:
    st.write("#### Actual vs. Predicted Prices (last 5 trading days)")
    st.table(pred_price.reindex(columns=['Close','Predictions']).tail(5))
    st.markdown(filedownload(pred_price), unsafe_allow_html=True)
    st.write("#### Prediction Model Performance")
    st.table(df_score)
    
st.write("#### Actual vs. Predicted price chart")
st.line_chart(pred_price[['Close','Predictions']])

st.text('')
st.write("## THANKS FOR VISITING!")
st.write('Created by: [Hai Vu](https://www.linkedin.com/in/hai-vu/)')
