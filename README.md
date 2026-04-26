<h1 align="center">
IndexPulse: Stock Index Visualization & Prediction App
</h1>

<p align="center">
<img src="https://www.investopedia.com/thmb/rw1cnmtQgCcEDVxuAybDFo1U3_E=/750x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/Term-Definitions_Index-665572b2712d4a6ca49b3f49179e3733.jpg" 
alt="" title="" width="60%" height="60%">
</p>
<p align="center">
<sup><i>(Image Source: https://www.investopedia.com/terms/i/index.asp)</i></sup>
</p>

<p align="center">
<b>🚀 Click the Streamlit icon to try the live app → <a href="https://indexpulse.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit" height="30"></a>
<br>or visit https://indexpulse.streamlit.app/
</b>
</p>

<p align="center">
<b>📓 The accompanied Jupyter notebook is best rendered on NBViewer → <a href="https://nbviewer.org/github/namhaivu173/Stock_Index_App/blob/main/indices-test.ipynb">
    <img src="https://user-images.githubusercontent.com/2791223/29387450-e5654c72-8294-11e7-95e4-090419520edb.png" alt="Open in NBViewer" height="30"></a></b>
</p>

## I. Main goals of project:
This project focuses on Stock indices, which are measures of the performance of a group of stocks that represent a particular market or sector. A stock index is calculated based on the performance of a selected group of stocks, and it provides a snapshot of the overall performance of the market or sector that the index represents. In this project, my main goal is to:
- Visualize closing prices and volumes of major stock indices around the world, where the list of indices is scraped from [Yahoo Finance](https://finance.yahoo.com/world-indices/)
- Construct the [Efficient Frontier curve](https://www.investopedia.com/terms/e/efficientfrontier.asp) through random sampling and simulating performances of portfolios, each of which consists of different indices. Calculate Value at Risk (VaR) and show information of high-performance portfolios
- Generate price predictions for stock indices based on historical closing prices. The neural network with [Multi-layer Perceptron regressor (MLP Regressor)](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html) from the Sklearn library was used to construct the prediction model
- Create a Streamlit app that contains both the visualization, the efficient frontier simulation and the price prediction model

In reality, this app could be helpful for risk-averse investment funds (typically pension funds or mutual funds) who often have a long-term investment approach and want to make risk-adjusted returns.

## II. How the Portfolio Simulation Works

### Efficient Frontier
The simulation constructs the [Efficient Frontier](https://www.investopedia.com/terms/e/efficientfrontier.asp) by generating thousands of random portfolios, each composed of a different subset of indices with randomly assigned weights. For each portfolio, the simulation computes the **annualised expected return** and **annualised volatility (risk)**, then plots them together to reveal the risk-return trade-off across the full opportunity set.

Two sampling strategies are combined to improve coverage:
- **Mean-variance sampling** — weights are drawn from a uniform Dirichlet distribution, producing a broad scatter of portfolios across the risk-return space.
- **Return-optimised sampling** — weights are found via constrained optimisation (SLSQP) to maximise expected return at each draw, filling in the upper edge of the frontier.

The **Capital Market Line (CML)** is then drawn from the risk-free rate (10-year US Treasury yield at the selected end date) through the **tangency portfolio** — the portfolio with the highest Sharpe Ratio — representing the best possible risk-adjusted combination of the risk-free asset and the risky portfolio.

Three special portfolios are highlighted: **Minimum Risk**, **Maximum Return**, and **Maximum Sharpe Ratio**.

### Value at Risk (VaR)
VaR is calculated using the **parametric (normal distribution) method**. For a chosen holding period *t* and confidence level *α*:

1. Scale each portfolio's annualised return and variance to the *t*-day horizon.
2. Compute the expected portfolio value and its dollar standard deviation.
3. Use the normal inverse CDF (`norm.ppf`) at the `1 − α` percentile to find the worst-case portfolio value at that confidence level.
4. **VaR = Initial Investment − Worst-case Value**. A result of $0 means the portfolio is expected to be profitable even in the bottom percentile of outcomes.

### Key Assumptions
- Returns follow a **normal distribution**; tail risks (fat tails, skewness) are not explicitly modelled.
- **Correlations between indices are assumed stable** over the selected historical window.
- Investors can borrow and lend at the **risk-free rate** (10-year US Treasury yield).
- **Fractional weights** are allowed — portfolios are not constrained to round lots.
- All index returns are **converted to USD** so they are comparable on a common currency basis. Forex indices (USD Index, AUD/GBP/EUR/JPY indices) are excluded from the simulation since they measure currency strength rather than equity market performance.
- Indices with **extreme annualised returns** (above Q3 + 3 × IQR of the peer group) are automatically excluded. These outliers typically arise when a local-currency index is converted to USD during a period of severe currency devaluation (e.g. Argentine Peso, Russian Ruble), making nominal returns appear in the hundreds of thousands of percent and distorting the optimizer. Such indices remain visible in the Stock Index Dashboard.

## III. Screenshots of app UI:
![screen1](https://github.com/namhaivu173/Stock_Index_App/blob/main/app_screenshots/Screenshot_1.png)
![screen2](https://github.com/namhaivu173/Stock_Index_App/blob/main/app_screenshots/Screenshot_2.png)
![screen3](https://github.com/namhaivu173/Stock_Index_App/blob/main/app_screenshots/Screenshot_3.png)
![screen4](https://github.com/namhaivu173/Stock_Index_App/blob/main/app_screenshots/Screenshot_4.png)
![screen5](https://github.com/namhaivu173/Stock_Index_App/blob/main/app_screenshots/Screenshot_5.png)
![screen6](https://github.com/namhaivu173/Stock_Index_App/blob/main/app_screenshots/Screenshot_6.png)

<!-- (https://nbviewer.org/github/namhaivu173/Stock_Index_App/blob/main/indices-test.ipynb) -->
