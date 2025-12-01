import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import datetime as dt
from textblob import TextBlob
import random

# Mock sentiment analysis
def fetch_news_headlines(query="stock market"):
    # Mock headlines for testing
    mock_headlines = [
        f"{query} stock hits all-time high",
        f"Investors optimistic about {query}",
        f"{query} faces challenges in the market",
    ]
    return mock_headlines

def analyze_sentiment(headlines):
    # Mock sentiment analysis
    return random.uniform(-1, 1)  # Random sentiment score between -1 and 1

# Set page configuration
st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# Start and end dates for stock data
start = '2014-01-01'
end = dt.datetime.now().strftime('%Y-%m-%d')  # Dynamic end date for live updates

# Streamlit app title
st.title('Stock Price Prediction and Detailed Analysis 📈')

# Sidebar inputs for user customization
st.sidebar.header("User Inputs")
user_input = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, TSLA)", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime(start))
end_date = st.sidebar.date_input("End Date", pd.to_datetime(end))

# Validate date range
if start_date >= end_date:
    st.error("Error: End Date must be after Start Date.")
else:
    # Load data using yfinance
    with st.spinner('Loading data...'):
        df = yf.download(user_input, start=start_date, end=end_date)
    if df.empty:
        st.error("No data found for the given ticker. Please try another ticker.")
    else:
        st.success('Data loaded successfully!')

        # Display basic data
        st.write(f"Stock data for {user_input} from {start_date} to {end_date}:")
        st.write(df.head())

        # Data description
        st.subheader('Data Overview and Statistics')
        st.write("This section provides an overview of the stock price data and basic statistics.")
        st.write(df.describe())

        # Feature Engineering: Adding SMA, MACD, Bollinger Bands, and RSI
        st.subheader("Feature Engineering: Technical Indicators")
        st.write("""
        To better understand the stock's behavior, additional metrics such as 
        Simple Moving Averages (SMA), MACD, Bollinger Bands, and RSI 
        (Relative Strength Index) are calculated.
        """)

        # Simple Moving Averages (SMA)
        df['sma_50'] = df['Close'].rolling(window=50).mean()  # 50-day SMA
        df['sma_100'] = df['Close'].rolling(window=100).mean()  # 100-day SMA

        # MACD Calculation
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        df['20_SMA'] = df['Close'].rolling(window=20).mean()
        df['20_STD'] = df['Close'].rolling(window=20).std()
        df['Upper_Band'] = df['20_SMA'] + (df['20_STD'] * 2)
        df['Lower_Band'] = df['20_SMA'] - (df['20_STD'] * 2)

        # RSI Calculation
        rsi_period = 14
        delta = df['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=rsi_period, min_periods=1).mean()
        avg_loss = loss.rolling(window=rsi_period, min_periods=1).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Calculate Daily Returns
        df['Daily_Return'] = df['Close'].pct_change()

        # Calculate Volatility (20-day rolling standard deviation of daily returns)
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(20)

        # Add Sentiment Analysis
        df['Sentiment'] = 0  # Initialize sentiment column
        headlines = fetch_news_headlines(query=user_input)  # Fetch mock headlines
        if headlines:  # Only calculate sentiment if headlines are found
            sentiment = analyze_sentiment(headlines)  # Calculate mock sentiment score
            df['Sentiment'] = sentiment  # Add sentiment to the DataFrame
        else:
            st.warning("No headlines found. Using default sentiment score (0).")

        # Drop NaN values caused by rolling calculations
        df.dropna(inplace=True)

        st.write("Sample of the dataset with new features (SMA, MACD, Bollinger Bands, RSI, Volatility, Sentiment):")
        st.write(df[['Close', 'sma_50', 'sma_100', 'MACD', 'Signal_Line', 'Upper_Band', 'Lower_Band', 'RSI', 'Volatility', 'Sentiment']].head())

        # Visualizations of metrics
        st.subheader('Visualization of Metrics')

        # SMA Visualizations
        st.subheader('Closing Price with SMA (50 and 100 Days)')
        fig_sma = plt.figure(figsize=(12, 6))
        plt.plot(df['Close'], label='Closing Price', color='blue')
        plt.plot(df['sma_50'], label='50-Day SMA', color='orange')
        plt.plot(df['sma_100'], label='100-Day SMA', color='green')
        plt.title(f'Closing Price and Moving Averages for {user_input}')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig_sma)

        # MACD Visualization
        st.subheader('MACD and Signal Line')
        fig_macd = plt.figure(figsize=(12, 6))
        plt.plot(df['MACD'], label='MACD', color='blue')
        plt.plot(df['Signal_Line'], label='Signal Line', color='red')
        plt.title(f'MACD for {user_input}')
        plt.xlabel('Time')
        plt.ylabel('MACD')
        plt.legend()
        st.pyplot(fig_macd)

        # Bollinger Bands Visualization
        st.subheader('Bollinger Bands')
        fig_bollinger = plt.figure(figsize=(12, 6))
        plt.plot(df['Close'], label='Closing Price', color='blue')
        plt.plot(df['Upper_Band'], label='Upper Band', color='red')
        plt.plot(df['Lower_Band'], label='Lower Band', color='green')
        plt.fill_between(df.index, df['Upper_Band'], df['Lower_Band'], color='lightgray', alpha=0.3)
        plt.title(f'Bollinger Bands for {user_input}')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig_bollinger)

        # RSI Visualization
        st.subheader('Relative Strength Index (RSI)')
        fig_rsi = plt.figure(figsize=(12, 6))
        plt.plot(df['RSI'], label='RSI', color='green')
        plt.axhline(30, linestyle='--', color='red', label='Oversold (30)')
        plt.axhline(70, linestyle='--', color='blue', label='Overbought (70)')
        plt.title(f'Relative Strength Index (RSI) for {user_input}')
        plt.xlabel('Time')
        plt.ylabel('RSI')
        plt.legend()
        st.pyplot(fig_rsi)

        # Split data into training and testing sets
        data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)

        # Load the trained LSTM model
        model = load_model('keras_model.h5')

        # Prepare testing data
        past_100_days = data_training.tail(100)
        final_df = past_100_days.append(data_testing, ignore_index=True)
        input_data = scaler.fit_transform(final_df)

        x_test = []
        y_test = []

        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i - 100:i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)

        # Predict using the LSTM model
        y_predicted = model.predict(x_test)
        scale_factor = 1 / scaler.scale_[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        # Evaluate model performance
        st.subheader('Model Performance Metrics')
        mse = mean_squared_error(y_test, y_predicted)
        mae = mean_absolute_error(y_test, y_predicted)
        r2 = r2_score(y_test, y_predicted)

        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"R² Score: {r2:.2f}")

        # Predictions vs Original
        st.subheader('Predictions vs Original Prices')
        fig2 = plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='Original Price', color='b')
        plt.plot(y_predicted, label='Predicted Price', color='r')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.title("Original vs Predicted Prices")
        st.pyplot(fig2)

        # Investment Recommendation Section
        st.subheader('📈 Future Stock Movement & Investment Recommendation')

        # Check SMA trend
        if not df.empty and 'sma_50' in df.columns and 'sma_100' in df.columns and 'Close' in df.columns:
            # Extract scalar values explicitly
            latest_sma_50 = df['sma_50'].iloc[-1].item()  # Ensure scalar
            latest_sma_100 = df['sma_100'].iloc[-1].item()  # Ensure scalar
            latest_price = df['Close'].iloc[-1].item()  # Ensure scalar

            # Debug: Print values and types
            st.write(f"Debug: latest_sma_50 = {latest_sma_50}, type = {type(latest_sma_50)}")
            st.write(f"Debug: latest_sma_100 = {latest_sma_100}, type = {type(latest_sma_100)}")
            st.write(f"Debug: latest_price = {latest_price}, type = {type(latest_price)}")

            # Ensure all values are scalars
            if isinstance(latest_sma_50, (int, float)) and isinstance(latest_sma_100, (int, float)) and isinstance(latest_price, (int, float)):
                if latest_sma_50 > latest_sma_100 and latest_price > latest_sma_50:
                    sma_trend = "Bullish 📊 (Uptrend - Positive Signal)"
                elif latest_sma_50 < latest_sma_100 and latest_price < latest_sma_50:
                    sma_trend = "Bearish 📉 (Downtrend - Negative Signal)"
                else:
                    sma_trend = "Neutral ⚖️ (Unclear Trend)"
            else:
                sma_trend = "Neutral ⚖️ (Invalid Data)"
        else:
            sma_trend = "Neutral ⚖️ (Data Unavailable)"
            latest_price = None

        # RSI Calculation
        latest_rsi = df['RSI'].iloc[-1].item()  # Ensure scalar
        if latest_rsi > 70:
            rsi_trend = "Overbought (Stock may fall soon) 🚨"
        elif latest_rsi < 30:
            rsi_trend = "Oversold (Stock may rise soon) ✅"
        else:
            rsi_trend = "Stable (No extreme movement expected) ⚖️"

        # Volatility Calculation
        latest_volatility = df['Volatility'].iloc[-1].item()  # Ensure scalar
        average_volatility = df['Volatility'].mean()
        if latest_volatility > average_volatility:
            volatility_trend = "High (Risky Investment) ⚠️"
        else:
            volatility_trend = "Low (Stable Investment) ✅"

        # Recent Price Movement
        recent_price_change = df['Close'].iloc[-1].item() - df['Close'].iloc[-5].item()  # Ensure scalar
        if recent_price_change > 0:
            price_trend = "Upward ↗️"
        else:
            price_trend = "Downward ↘️"

        # Display Investment Metrics
        st.write(f"**📌 Stock Ticker:** {user_input}")
        st.write(f"**📌 Latest Price:** ${latest_price:.2f}")
        st.write(f"**📌 SMA Trend:** {sma_trend}")
        st.write(f"**📌 RSI Indicator:** {rsi_trend}")
        st.write(f"**📌 Volatility:** {volatility_trend}")
        st.write(f"**📌 Recent Price Trend (Last 5 Days):** {price_trend}")

        # Final Recommendation Logic
        if (
            sma_trend == "Bullish 📊 (Uptrend - Positive Signal)"
            and rsi_trend != "Overbought (Stock may fall soon) 🚨"
            and volatility_trend == "Low (Stable Investment) ✅"
            and price_trend == "Upward ↗️"
        ):
            st.success(f"✅ **Recommended: BUY {user_input} 📈** (Strong Uptrend, No Overbought Condition, Low Volatility, Upward Price Trend)")
        elif (
            sma_trend == "Bearish 📉 (Downtrend - Negative Signal)"
            or rsi_trend == "Overbought (Stock may fall soon) 🚨"
            or price_trend == "Downward ↘️"
        ):
            st.error(f"❌ **Recommended: AVOID {user_input} 📉** (Downtrend, Overbought Condition, or Downward Price Trend)")
        else:
            st.warning(f"⚠️ **Recommendation: HOLD {user_input} ⚖️** (Market Unclear, Wait for Confirmation)")

        # Export predictions as a downloadable file
        st.subheader("Download Predictions")
        final_output = pd.DataFrame({'Original': y_test.flatten(), 'Predicted': y_predicted.flatten()})
        st.download_button("Download Predictions as CSV", final_output.to_csv(index=False), file_name="predictions.csv")

        # Project summary
        st.subheader("Project Summary")
        st.write("""
        This project provides a comprehensive stock price prediction system, integrating historical data analysis 
        with technical indicators like SMA, Daily Returns, Volatility, and RSI. Using an LSTM neural network, 
        the app predicts future stock prices and provides detailed insights into market trends.
        """)