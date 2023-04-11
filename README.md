<h1 align="center">
Building Stock Index App with Streamlit
</h1>

<p align="center">
<img src="https://www.investopedia.com/thmb/rw1cnmtQgCcEDVxuAybDFo1U3_E=/750x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/Term-Definitions_Index-665572b2712d4a6ca49b3f49179e3733.jpg" 
alt="" title="" width="60%" height="60%">
</p>
<p align="center">
<sup><i>(Image Source: https://www.investopedia.com/terms/i/index.asp)</i></sup>
</p>

<b>You can open the app by clicking on the Streamlit icon: [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://namhaivu173-stock-index-app-indices-app-6amttx.streamlit.app/)

The accompanied Jupyter notebook can be viewed [using NBViewer](https://nbviewer.org/github/namhaivu173/Stock_Index_App/blob/1cbc0791eed046b18ae12a9ef30cc038f3a61f28/Indices_Dashboard.ipynb)
</b>

## Main goals of project:
This project focuses on Stock indices, which are are measures of the performance of a group of stocks that represent a particular market or sector. A stock index is calculated based on the performance of a selected group of stocks, and it provides a snapshot of the overall performance of the market or sector that the index represents. In this project, my main goal is to:
- Visualize closing prices and volumes of major stock indices around the world, where the list of indices is based on [Yahoo Finance](https://finance.yahoo.com/world-indices/)
- Construct the [Efficient Fronter curve](https://www.investopedia.com/terms/e/efficientfrontier.asp) through random sampling and simulating performances of portfolios, each of which consists of different indices
- Generate price predictions for stock indices based on historical closing prices. The neural network with [Multi-layer Perceptron regressor (MLP Regressor)](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html) from the Sklearn library was used to construct the prediction model
- Create a Streamlit app that contains both the visualization, the efficient frontier simulation and the price prediction model
