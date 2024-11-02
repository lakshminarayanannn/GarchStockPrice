
import streamlit as st
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from pykalman import KalmanFilter
import talib
import warnings
import os
import pickle
from arch import arch_model

warnings.filterwarnings('ignore')

class DataLoader:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        
    def fetch_data(self):
        """Fetch and preprocess stock data."""
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        if data.empty:
            raise ValueError("Data for the specified date range is not available.")
        
        data['Returns'] = data['Adj Close'].pct_change()
        data = self.calculate_technical_indicators(data)
        data.dropna(inplace=True)
        return data
    
    @staticmethod
    def calculate_technical_indicators(df):
        """Calculate various technical indicators."""
        df['RSI'] = talib.RSI(df['Adj Close'], timeperiod=14)
        macd, macdsignal, _ = talib.MACD(df['Adj Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = macdsignal
        
        upperband, middleband, lowerband = talib.BBANDS(df['Adj Close'])
        df['BB_upper'] = upperband
        df['BB_middle'] = middleband
        df['BB_lower'] = lowerband
        
        df['OBV'] = talib.OBV(df['Adj Close'], df['Volume'])
        return df


class LSTMPredictor:
    def __init__(self, time_steps=20, model_id=None):
        self.time_steps = time_steps
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.model_id = model_id
        self.model_dir = 'saved_models'
        os.makedirs(self.model_dir, exist_ok=True)
        
    def get_model_paths(self):
        return {
            'model': os.path.join(self.model_dir, f'{self.model_id}_model.h5'),
            'feature_scaler': os.path.join(self.model_dir, f'{self.model_id}_feature_scaler.pkl'),
            'target_scaler': os.path.join(self.model_dir, f'{self.model_id}_target_scaler.pkl')
        }
    
    def save_model(self):
        if not self.model_id:
            return
        
        paths = self.get_model_paths()
        self.model.save(paths['model'])
        with open(paths['feature_scaler'], 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        with open(paths['target_scaler'], 'wb') as f:
            pickle.dump(self.target_scaler, f)
        print(f"Model saved with ID: {self.model_id}")
    
    def load_model_func(self):
        if not self.model_id:
            return False
        
        paths = self.get_model_paths()
        if not all(os.path.exists(path) for path in paths.values()):
            return False
        
        self.model = load_model(paths['model'])
        with open(paths['feature_scaler'], 'rb') as f:
            self.feature_scaler = pickle.load(f)
        with open(paths['target_scaler'], 'rb') as f:
            self.target_scaler = pickle.load(f)
        print(f"Model loaded from ID: {self.model_id}")
        return True
        
    def prepare_data(self, df, feature_columns, target_column, is_training=True):
        """Prepare data for LSTM modeling."""
        features = df[feature_columns].values
        target = df[target_column].values.reshape(-1, 1)
        
        if is_training:
            features_scaled = self.feature_scaler.fit_transform(features)
            target_scaled = self.target_scaler.fit_transform(target)
        else:
            features_scaled = self.feature_scaler.transform(features)
            target_scaled = self.target_scaler.transform(target)
        
        return self.create_sequences(features_scaled, target_scaled)
    
    def create_sequences(self, features, target):
        """Create sequences for LSTM input."""
        X, y = [], []
        for i in range(self.time_steps, len(features)):
            X.append(features[i-self.time_steps:i])
            y.append(target[i])
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Create and compile LSTM model with modified architecture."""
        from keras.optimizers import Adam
        
        self.model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape,
                 kernel_initializer='he_normal'),
            Dropout(0.2),
            LSTM(50, return_sequences=False,
                 kernel_initializer='he_normal'),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
 
        optimizer = Adam(learning_rate=0.001)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')
        
    def train(self, X_train, y_train, epochs=200, batch_size=32):
        """Train the LSTM model with learning rate scheduling."""
        from keras.callbacks import ReduceLROnPlateau, EarlyStopping
        

        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  
            patience=5,
            min_lr=0.00001,
            verbose=1
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
        
 
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[lr_scheduler, early_stopping],
            verbose=1
        )
        
        if self.model_id:
            self.save_model()
        
        return history
    
    def predict(self, X_test):
        """Make predictions and inverse transform the results."""
        predictions_scaled = self.model.predict(X_test, batch_size=1)
        return self.target_scaler.inverse_transform(predictions_scaled).flatten()


class GARCHPredictor:
    def __init__(self, p=1, q=1):
        self.p = p
        self.q = q
        self.model = None
        self.res = None

    def fit(self, returns):
        self.model = arch_model(returns * 100, vol='Garch', p=self.p, q=self.q, dist='normal')
        self.res = self.model.fit(disp='off')

    def forecast(self, horizon):
        forecast = self.res.forecast(horizon=horizon)
        forecast_variance = forecast.variance.values[-1, :]
        forecast_volatility = np.sqrt(forecast_variance) / 100  # Scale back to original
        return forecast_volatility


class SimpleKalmanPredictor:
    def __init__(self):
        self.kf = None

    def initialize_filter(self, initial_price):
        """Initialize the Kalman Filter with simple 1D state."""
        self.kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=initial_price,
            initial_state_covariance=1,
            observation_covariance=1,
            transition_covariance=0.01
        )

    def predict(self, measurements):
        """Make predictions using the Kalman Filter."""
        state_means, state_covs = self.kf.filter(measurements)
        return state_means.flatten()


class StockPredictor:
    def __init__(self, ticker, start_date, end_date, prediction_days, model_id=None, garch_p=1, garch_q=1):
        self.prediction_days = prediction_days
        self.data_loader = DataLoader(ticker, start_date, end_date)
        self.lstm_predictor = LSTMPredictor(time_steps=prediction_days, model_id=model_id)
        self.kalman_predictor = SimpleKalmanPredictor()
        self.garch_predictor = GARCHPredictor(p=garch_p, q=garch_q)
        
    def run_prediction(self):
        """Run LSTM, Kalman, and GARCH predictions."""
        df = self.data_loader.fetch_data()
        
        if len(df) < self.prediction_days * 2:
            raise ValueError("Not enough data for the specified prediction window")
        
        train_data = df[:-self.prediction_days]
        test_data = df[-self.prediction_days*2:]
        
        feature_columns = ['Adj Close', 'RSI', 'MACD', 'MACD_Signal', 
                           'BB_upper', 'BB_middle', 'BB_lower', 'OBV']
        
        X_train, y_train = self.lstm_predictor.prepare_data(
            train_data, feature_columns, 'Adj Close', is_training=True)
        X_test, y_test = self.lstm_predictor.prepare_data(
            test_data, feature_columns, 'Adj Close', is_training=False)
        
        if not self.lstm_predictor.load_model_func():
            print("Training new LSTM model...")
            self.lstm_predictor.build_model((X_train.shape[1], X_train.shape[2]))
            self.lstm_predictor.train(X_train, y_train)
        
        lstm_predictions = self.lstm_predictor.predict(X_test)[-self.prediction_days:]
        
        prices = train_data['Adj Close'].values
        test_prices = test_data['Adj Close'].values[-self.prediction_days:]
        self.kalman_predictor.initialize_filter(prices[0])
        kalman_predictions = self.kalman_predictor.predict(test_prices)
        kalman_predictions = kalman_predictions[-self.prediction_days:]
        
        returns = train_data['Returns'].dropna()
        self.garch_predictor.fit(returns)
        garch_forecast_volatility = self.garch_predictor.forecast(self.prediction_days)
        
        test_returns = test_data['Returns'].dropna()[-self.prediction_days:]
        realized_volatility = np.abs(test_returns.values)
        
        dates = test_data.index[-self.prediction_days:]
        actual_prices = test_data['Adj Close'].values[-self.prediction_days:]
        
        return (dates,
                actual_prices,
                lstm_predictions,
                kalman_predictions,
                realized_volatility,
                garch_forecast_volatility)
    
    def plot_results(self, dates, actual, lstm_pred, kalman_pred, realized_volatility, garch_forecast_volatility):
        """Plot the predictions against actual values and volatility."""
        plt.figure(figsize=(15, 7))
        plt.plot(dates, actual, label='Actual Price', color='blue', linewidth=2)
        plt.plot(dates, lstm_pred, label='LSTM Prediction', color='green', linestyle='--')
        plt.plot(dates, kalman_pred, label='Kalman Filter Prediction', color='red', linestyle=':')
        plt.title(f'Stock Price Prediction ({self.prediction_days} days)')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
        plt.clf()
        
        print_metrics = True  

        if print_metrics:
            st.subheader("Prediction Metrics")
            st.markdown("**LSTM Predictions:**")
            st.write(f"MSE: {mean_squared_error(actual, lstm_pred):.2f}")
            st.write(f"RMSE: {np.sqrt(mean_squared_error(actual, lstm_pred)):.2f}")
            st.write(f"MAE: {mean_absolute_error(actual, lstm_pred):.2f}")
            
            st.markdown("**Kalman Predictions:**")
            st.write(f"MSE: {mean_squared_error(actual, kalman_pred):.2f}")
            st.write(f"RMSE: {np.sqrt(mean_squared_error(actual, kalman_pred)):.2f}")
            st.write(f"MAE: {mean_absolute_error(actual, kalman_pred):.2f}")
        
        plt.figure(figsize=(15, 7))
        plt.plot(dates, garch_forecast_volatility, marker='o', label='GARCH Forecasted Volatility')
        plt.plot(dates, realized_volatility, marker='x', label='Realized Volatility (Abs Returns)')
        plt.title('GARCH Forecasted Volatility vs Realized Volatility')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
        plt.clf()
        
        if print_metrics:
            st.markdown("**Volatility Metrics:**")
            st.write(f"MSE: {mean_squared_error(realized_volatility, garch_forecast_volatility):.6f}")
            st.write(f"RMSE: {np.sqrt(mean_squared_error(realized_volatility, garch_forecast_volatility)):.6f}")
            st.write(f"MAE: {mean_absolute_error(realized_volatility, garch_forecast_volatility):.6f}")


class SimpleKalmanPredictorFuture:
    def __init__(self):
        self.kf = None
        self.states_mean = None
        self.states_covariance = None

    def initialize_filter(self, initial_price):
        """Initialize the Kalman Filter with simple 1D state."""
        self.kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=initial_price,
            initial_state_covariance=1,
            observation_covariance=1,
            transition_covariance=0.01
        )

    def filter(self, measurements):
        """Apply Kalman filter to measurements and store states."""
        self.states_mean, self.states_covariance = self.kf.filter(measurements)
        return self.states_mean.flatten()

class StockPredictorFuture:
    def __init__(self, ticker, start_date, end_date, prediction_days, model_id=None, garch_p=1, garch_q=1):
        self.prediction_days = prediction_days
        self.data_loader = DataLoader(ticker, start_date, end_date)
        self.lstm_predictor = LSTMPredictor(time_steps=prediction_days, model_id=model_id)
        self.kalman_predictor = SimpleKalmanPredictorFuture()
        self.garch_predictor = GARCHPredictor(p=garch_p, q=garch_q)
        self.df = None
        self.feature_columns = ['Adj Close', 'RSI', 'MACD', 'MACD_Signal', 
                              'BB_upper', 'BB_middle', 'BB_lower', 'OBV']
    
    def prepare_models(self):
        """Prepare and train all models using historical data."""
        self.df = self.data_loader.fetch_data()
        
        if len(self.df) < self.prediction_days * 2:
            raise ValueError("Not enough data for the specified prediction window")
        
        train_data = self.df
        

        X_train, y_train = self.lstm_predictor.prepare_data(
            train_data, self.feature_columns, 'Adj Close', is_training=True)
        
        if not self.lstm_predictor.load_model_func():
            print("Training new LSTM model...")
            self.lstm_predictor.build_model((X_train.shape[1], X_train.shape[2]))
            self.lstm_predictor.train(X_train, y_train)
        
        prices = train_data['Adj Close'].values
        self.kalman_predictor.initialize_filter(prices[0])
        self.kalman_predictor.filter(prices)

        returns = train_data['Returns'].dropna()
        self.garch_predictor.fit(returns)
    
    def predict_future(self, n_days):
        """Make predictions for the next n days into the future."""
        if self.df is None:
            self.prepare_models()
        
        last_date = self.df.index[-1]
        future_dates = [last_date + timedelta(days=x) for x in range(1, n_days + 1)]
        

        lstm_future = self._predict_lstm_future(n_days)
        

        kalman_future = self._predict_kalman_future(n_days)
        
        garch_future = self.garch_predictor.forecast(n_days)
        
        return future_dates, lstm_future, kalman_future, garch_future
    
    def _predict_lstm_future(self, n_days):
        """Helper method for LSTM future prediction."""
        last_sequence = self.df[self.feature_columns].values[-self.prediction_days:]
        predictions = []
        
        last_sequence_scaled = self.lstm_predictor.feature_scaler.transform(last_sequence)
        current_sequence = last_sequence_scaled.copy()
        
        for _ in range(n_days):
            current_sequence_reshaped = current_sequence.reshape(1, self.prediction_days, len(self.feature_columns))
            
            next_pred_scaled = self.lstm_predictor.model.predict(current_sequence_reshaped, verbose=0)
            
            next_pred = self.lstm_predictor.target_scaler.inverse_transform(next_pred_scaled)[0][0]
            predictions.append(next_pred)
                        
            new_row = current_sequence[-1:].copy()
            new_row[0, 0] = next_pred_scaled[0][0]  # Update only the price
            current_sequence = np.vstack((current_sequence[1:], new_row))
        
        return np.array(predictions)
    
    def _predict_kalman_future(self, n_days):
        """Helper method for Kalman future prediction."""
        last_state_mean = self.kalman_predictor.states_mean[-1]
        last_state_cov = self.kalman_predictor.states_covariance[-1]
        
        predictions = []
        current_state = last_state_mean
        current_covariance = last_state_cov
        
        for _ in range(n_days):
            next_state_mean = current_state
            next_state_cov = current_covariance + self.kalman_predictor.kf.transition_covariance
            
            predictions.append(float(next_state_mean))
            
            current_state = next_state_mean
            current_covariance = next_state_cov
        
        return np.array(predictions)
    
    def plot_future_predictions(self, n_days):
        """Plot future predictions for all models."""
        future_dates, lstm_future, kalman_future, garch_future = self.predict_future(n_days)

        plt.figure(figsize=(15, 7))
        
        historical_dates = self.df.index[-30:]
        historical_prices = self.df['Adj Close'].values[-30:]
        plt.plot(historical_dates, historical_prices, label='Historical Price', color='blue')
        
        plt.plot(future_dates, lstm_future, label='LSTM Future Prediction', color='green', linestyle='--')
        plt.plot(future_dates, kalman_future, label='Kalman Future Prediction', color='red', linestyle=':')
        
        plt.title(f'Future Stock Price Predictions (Next {n_days} days)')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
        plt.clf()
        
        plt.figure(figsize=(15, 7))
        
        historical_volatility = np.abs(self.df['Returns'].values[-30:])
        plt.plot(historical_dates, historical_volatility, label='Historical Volatility', color='blue')
        
        plt.plot(future_dates, garch_future, label='GARCH Volatility Forecast', color='orange', linestyle='--')
        
        plt.title(f'Future Volatility Prediction (Next {n_days} days)')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
        plt.clf()


def main():
    st.title(" Stock Prediction Dashboard")
    st.sidebar.title("Configuration")
    
    stock_options = {
        'Tesla': 'TSLA',
        'Apple': 'AAPL',
        'Palo Alto Networks': 'PANW',
        'Meta': 'META',
        'NVIDIA': 'NVDA',
    }
    selected_stock_name = st.sidebar.selectbox("Select Stock", list(stock_options.keys()))
    selected_stock = stock_options[selected_stock_name]
    
    today = datetime.today()
    default_start = today - timedelta(days=5*365)  # 5 years ago
    start_date = st.sidebar.date_input("Start Date", default_start)
    end_date = st.sidebar.date_input("End Date", today)
    
    if start_date > end_date:
        st.sidebar.error("Error: End date must fall after start date.")
    
    prediction_days = st.sidebar.slider("Prediction Window (days)", min_value=10, max_value=60, value=15)
    
    st.sidebar.header("Actions")
    backtest_button = st.sidebar.button("🔍 Backtest")
    future_predict_button = st.sidebar.button("Future Prediction")
    
    model_id_map = {
        'TSLA': 'TSLA_daily_v7',
        'AAPL': 'AAPL_daily_v7',
        'PANW': 'PANW_daily_v7',
        'META': 'META_daily_v7',
        'NVDA': 'NVDA_daily_v7',
    }
    model_id = model_id_map.get(selected_stock, 'default_model')
    
    garch_p = 1
    garch_q = 1
    
    if backtest_button:
        st.header(f" Backtest Results for {selected_stock_name} ({selected_stock})")
        with st.spinner('Running backtest... This may take a few minutes.'):
            try:
                predictor = StockPredictor(
                    ticker=selected_stock,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    prediction_days=prediction_days,
                    model_id=model_id,
                    garch_p=garch_p,
                    garch_q=garch_q
                )
                dates, actual, lstm_pred, kalman_pred, realized_volatility, garch_forecast_volatility = predictor.run_prediction()
                predictor.plot_results(dates, actual, lstm_pred, kalman_pred, realized_volatility, garch_forecast_volatility)
            except Exception as e:
                st.error(f"Error during backtest: {e}")
    
    if future_predict_button:
        st.header(f"Future Predictions for {selected_stock_name} ({selected_stock})")
        with st.spinner('Running future predictions... This may take a few minutes.'):
            try:
                predictor_future = StockPredictorFuture(
                    ticker=selected_stock,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    prediction_days=prediction_days,
                    model_id=model_id,
                    garch_p=garch_p,
                    garch_q=garch_q
                )
                future_days = st.sidebar.slider("Future Prediction Days", min_value=1, max_value=30, value=5)
                predictor_future.plot_future_predictions(future_days)
            except Exception as e:
                st.error(f"Error during future prediction: {e}")

if __name__ == "__main__":
    main()
