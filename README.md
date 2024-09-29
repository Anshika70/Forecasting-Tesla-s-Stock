# Tesla Stock Forecasting using GRU Algorithm

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction
This project focuses on forecasting Tesla's stock prices using a deep learning model based on the GRU (Gated Recurrent Unit) architecture. GRU is a type of Recurrent Neural Network (RNN) that is effective for sequential data and is used here for time-series forecasting. The project demonstrates how to build and train a GRU model to predict future stock prices based on historical data.

## Dataset
The dataset used for this project includes Tesla's historical stock prices. It contains features like:
- Open price
- Close price
- High price
- Low price
- Volume

The dataset can be obtained from popular sources like Yahoo Finance, Alpha Vantage, or other stock market data providers. In this project, the dataset is preprocessed before feeding it to the model.

## Model Architecture
The GRU model is designed to handle sequential data, which is crucial for time-series forecasting like stock price prediction. The key layers in the model include:
- GRU Layer: Captures temporal dependencies in the stock prices.
- Dense Layer: Final output layer to predict the stock price for the next time step.

The architecture includes:
- Input Layer: Takes the historical stock prices as input.
- GRU Layer(s): Extracts features from the time-series data.
- Dense Layer: Outputs the predicted stock price.

## Usage
Once the environment is set up and dependencies are installed, you can start training the model or run predictions:

1. **Train the GRU model**:
    ```bash
    python train_model.py
    ```

2. **Make predictions** using the trained model:
    ```bash
    python predict.py
    ```

### Key Scripts
- `train_model.py`: Script to train the GRU model using the Tesla stock price dataset.
- `predict.py`: Script to generate predictions from the trained GRU model.

## Results
After training the GRU model, the stock price predictions for Tesla are evaluated based on common metrics such as Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).

### Visualization
The project includes visualizations comparing the predicted stock prices and actual stock prices over time:

![Stock Price Prediction vs Actual](./results/prediction_vs_actual.png)


