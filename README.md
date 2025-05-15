# RNN-based Stock Price Prediction for American Airlines (AAL)

## Introduction

This project implements a Recurrent Neural Network (RNN) to forecast the next-day closing price of American Airlines (AAL) stock. Leveraging time series data, the model captures temporal dependencies to provide accurate price predictions.

## Dataset

- Source: Yahoo Finance

- Ticker: AAL

- Period: December 1, 2022 to November 30, 2023

- Features: Open, High, Low, Close, Adj Close, Volume

## Features

- Utilizes raw OHLC data along with adjusted closing price and trading volume

- Constructs sequences of 5-day historical windows to predict the following day's closing price (input shape: (5, 6))
## Preprocessing

1. Normalization:

    - Scales all features to the [0,1] range using `MinMaxScaler`

2. Sequence Generation:

    - Custom `rnn_data_setup` function creates training samples with lookback windows and corresponding targets

3. Data Splitting:

    - Training set: 144 samples

    - Validation set: 48 samples

    - Test set: 48 samples

## Model Architecture

- Type: Sequential RNN

- Layers:

    - `SimpleRNN` with 16 units (linear activation)

    - `Dense` output layer (1 unit)

- Hyperparameters:

    - Learning rate: 0.0012

    - Batch size: 512

    - Loss: Mean Squared Error (MSE)

    - Metric: Mean Absolute Error (MAE)

    - Optimizer: Adam

## Training

- Epochs: 100

- Validation: Monitored on a hold-out validation set during training (verbose=0)

## Results

- Prediction Shape: (48, 1)

- Evaluation Metrics on Test Set:

    - Mean Squared Error (MSE): 0.08615

    - Root Mean Squared Error (RMSE): 0.29352

## Usage

1. Clone the repository:
   
   `git clone https://github.com/CSwebD/RNN-AAL-Prediction.git`
   
   `cd RNN-AAL-Prediction`

3. Install dependencies:

    `pip install -r requirements.txt`

3. Launch Jupyter Notebook:

   `RNN.pred.AAL_jupyter.ipynb`

4. Execute all cells to reproduce preprocessing, training, and evaluation.

## Dependencies

Python 3.x

numpy, pandas, matplotlib, seaborn, yfinance, tensorflow (keras), scikit-learn

## Contributing

Contributions are welcome! Please fork the repo, create a branch for your feature or bugfix, and submit a pull request.

## License

This project is licensed under the [MIT License](/LICENSE). Feel free to use and modify for educational and research purposes.
