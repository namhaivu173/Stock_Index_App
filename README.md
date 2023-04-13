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

<b>You can open the app by clicking on the Streamlit icon [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://haivu173-stock-indices-tool.streamlit.app/)

The accompanied Jupyter notebook can be best rendered using NBViewer [![Streamlit App](https://camo.githubusercontent.com/c45f3816fe3efb095a64468c409bfbd40e971a85fdcc85fc101ee6aaa8943b10/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f72656e6465722d6e627669657765722d6f72616e67652e7376673f636f6c6f72423d66333736323626636f6c6f72413d346434643464)](https://nbviewer.org/github/namhaivu173/Stock_Index_App/blob/acc81b70eb9c8014c50fd92a4f8c1e0850520d16/Indices_Dashboard.ipynb)
</b>

## Main goals of project:
This project focuses on Stock indices, which are measures of the performance of a group of stocks that represent a particular market or sector. A stock index is calculated based on the performance of a selected group of stocks, and it provides a snapshot of the overall performance of the market or sector that the index represents. In this project, my main goal is to:
- Visualize closing prices and volumes of major stock indices around the world, where the list of indices is scraped from [Yahoo Finance](https://finance.yahoo.com/world-indices/)
- Construct the [Efficient Frontier curve](https://www.investopedia.com/terms/e/efficientfrontier.asp) through random sampling and simulating performances of portfolios, each of which consists of different indices
- Generate price predictions for stock indices based on historical closing prices. The neural network with [Multi-layer Perceptron regressor (MLP Regressor)](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html) from the Sklearn library was used to construct the prediction model
- Create a Streamlit app that contains both the visualization, the efficient frontier simulation and the price prediction model

In reality, this app could be helpful for risk-averse investment funds (typically pension funds or mutual funds) who often have a long-term investment approach and want to make risk-adjusted returns.

## Screenshots of app UI:
![screen1](https://github.com/namhaivu173/Stock_Index_App/blob/e535d4dfa634bc9d6276e744a0070b84f12db852/App%20Screenshots/s1.PNG)
![screen2](https://github.com/namhaivu173/Stock_Index_App/blob/c6509c880bf0fa66bbdddf9c0d999439ac997889/App%20Screenshots/s2.png)
![screen3](https://github.com/namhaivu173/Stock_Index_App/blob/c6509c880bf0fa66bbdddf9c0d999439ac997889/App%20Screenshots/s3.png)
![screen4](https://github.com/namhaivu173/Stock_Index_App/blob/c6509c880bf0fa66bbdddf9c0d999439ac997889/App%20Screenshots/s4.png)
![screen5](https://github.com/namhaivu173/Stock_Index_App/blob/e535d4dfa634bc9d6276e744a0070b84f12db852/App%20Screenshots/s5.png)
![screen6](https://github.com/namhaivu173/Stock_Index_App/blob/e535d4dfa634bc9d6276e744a0070b84f12db852/App%20Screenshots/s6.png)

<!-- (https://nbviewer.org/github/namhaivu173/Stock_Index_App/blob/main/Indices_Dashboard.ipynb) -->
