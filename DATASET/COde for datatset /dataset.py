import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def engineer_features(df):
    """
    Comprehensive feature engineering for stock prediction with XAI
    """
    print("🔧 Starting Feature Engineering...")
    
    # Make a copy
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    # Check if 'Adj Close' exists, if not use 'Close'
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    print(f"   Using '{price_col}' for calculations")
    
    # Create Adj Close if it doesn't exist
    if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']
    
    # Group by ticker for proper time series calculations
    processed_dfs = []
    
    for ticker in df['Ticker'].unique():
        print(f"   Processing {ticker}...", end=" ")
        ticker_df = df[df['Ticker'] == ticker].copy()
        
        # ============================================
        # 1. PRICE-BASED FEATURES
        # ============================================
        
        # Daily returns
        ticker_df['Daily_Return'] = ticker_df['Adj Close'].pct_change()
        ticker_df['Log_Return'] = np.log(ticker_df['Adj Close'] / ticker_df['Adj Close'].shift(1))
        
        # Price changes
        ticker_df['Price_Change'] = ticker_df['Adj Close'] - ticker_df['Adj Close'].shift(1)
        ticker_df['Price_Range'] = ticker_df['High'] - ticker_df['Low']
        ticker_df['Price_Range_Pct'] = (ticker_df['High'] - ticker_df['Low']) / ticker_df['Open'] * 100
        
        # Gap (difference between open and previous close)
        ticker_df['Gap'] = ticker_df['Open'] - ticker_df['Adj Close'].shift(1)
        ticker_df['Gap_Pct'] = (ticker_df['Open'] - ticker_df['Adj Close'].shift(1)) / ticker_df['Adj Close'].shift(1) * 100
        
        # ============================================
        # 2. MOVING AVERAGES (Multiple Timeframes)
        # ============================================
        
        for window in [5, 10, 20, 50, 100, 200]:
            # Simple Moving Average
            ticker_df[f'SMA_{window}'] = ticker_df['Adj Close'].rolling(window=window).mean()
            
            # Exponential Moving Average
            ticker_df[f'EMA_{window}'] = ticker_df['Adj Close'].ewm(span=window, adjust=False).mean()
            
            # Price relative to MA
            ticker_df[f'Price_to_SMA_{window}'] = (ticker_df['Adj Close'] / ticker_df[f'SMA_{window}'] - 1) * 100
        
        # Golden Cross / Death Cross indicators
        ticker_df['MA_Cross_50_200'] = ticker_df['SMA_50'] - ticker_df['SMA_200']
        ticker_df['MA_Cross_20_50'] = ticker_df['SMA_20'] - ticker_df['SMA_50']
        
        # ============================================
        # 3. VOLATILITY INDICATORS
        # ============================================
        
        # Historical volatility (rolling std of returns)
        for window in [5, 10, 20, 30]:
            ticker_df[f'Volatility_{window}d'] = ticker_df['Daily_Return'].rolling(window=window).std()
        
        # ATR (Average True Range)
        high_low = ticker_df['High'] - ticker_df['Low']
        high_close = np.abs(ticker_df['High'] - ticker_df['Adj Close'].shift())
        low_close = np.abs(ticker_df['Low'] - ticker_df['Adj Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        ticker_df['ATR_14'] = true_range.rolling(14).mean()
        
        # Bollinger Bands
        ticker_df['BB_Middle'] = ticker_df['Adj Close'].rolling(20).mean()
        bb_std = ticker_df['Adj Close'].rolling(20).std()
        ticker_df['BB_Upper'] = ticker_df['BB_Middle'] + (bb_std * 2)
        ticker_df['BB_Lower'] = ticker_df['BB_Middle'] - (bb_std * 2)
        ticker_df['BB_Width'] = (ticker_df['BB_Upper'] - ticker_df['BB_Lower']) / ticker_df['BB_Middle']
        ticker_df['BB_Position'] = (ticker_df['Adj Close'] - ticker_df['BB_Lower']) / (ticker_df['BB_Upper'] - ticker_df['BB_Lower'])
        
        # ============================================
        # 4. MOMENTUM INDICATORS
        # ============================================
        
        # RSI (Relative Strength Index)
        delta = ticker_df['Adj Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        ticker_df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        exp1 = ticker_df['Adj Close'].ewm(span=12, adjust=False).mean()
        exp2 = ticker_df['Adj Close'].ewm(span=26, adjust=False).mean()
        ticker_df['MACD'] = exp1 - exp2
        ticker_df['MACD_Signal'] = ticker_df['MACD'].ewm(span=9, adjust=False).mean()
        ticker_df['MACD_Hist'] = ticker_df['MACD'] - ticker_df['MACD_Signal']
        
        # Stochastic Oscillator
        low_14 = ticker_df['Low'].rolling(14).min()
        high_14 = ticker_df['High'].rolling(14).max()
        ticker_df['Stochastic_K'] = 100 * (ticker_df['Adj Close'] - low_14) / (high_14 - low_14)
        ticker_df['Stochastic_D'] = ticker_df['Stochastic_K'].rolling(3).mean()
        
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            ticker_df[f'ROC_{period}'] = ((ticker_df['Adj Close'] - ticker_df['Adj Close'].shift(period)) / 
                                          ticker_df['Adj Close'].shift(period)) * 100
        
        # ============================================
        # 5. VOLUME-BASED FEATURES
        # ============================================
        
        # Volume changes
        ticker_df['Volume_Change'] = ticker_df['Volume'].pct_change()
        ticker_df['Volume_MA_20'] = ticker_df['Volume'].rolling(20).mean()
        ticker_df['Volume_Ratio'] = ticker_df['Volume'] / ticker_df['Volume_MA_20']
        
        # On-Balance Volume (OBV)
        ticker_df['OBV'] = (np.sign(ticker_df['Adj Close'].diff()) * ticker_df['Volume']).fillna(0).cumsum()
        
        # Money Flow Index (MFI)
        typical_price = (ticker_df['High'] + ticker_df['Low'] + ticker_df['Adj Close']) / 3
        money_flow = typical_price * ticker_df['Volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
        mfi_ratio = positive_flow / negative_flow
        ticker_df['MFI_14'] = 100 - (100 / (1 + mfi_ratio))
        
        # ============================================
        # 6. TEMPORAL FEATURES
        # ============================================
        
        ticker_df['Day_of_Week'] = ticker_df['Date'].dt.dayofweek
        ticker_df['Month'] = ticker_df['Date'].dt.month
        ticker_df['Quarter'] = ticker_df['Date'].dt.quarter
        ticker_df['Day_of_Month'] = ticker_df['Date'].dt.day
        ticker_df['Week_of_Year'] = ticker_df['Date'].dt.isocalendar().week
        
        # Cyclical encoding for temporal features
        ticker_df['Day_Sin'] = np.sin(2 * np.pi * ticker_df['Day_of_Week'] / 7)
        ticker_df['Day_Cos'] = np.cos(2 * np.pi * ticker_df['Day_of_Week'] / 7)
        ticker_df['Month_Sin'] = np.sin(2 * np.pi * ticker_df['Month'] / 12)
        ticker_df['Month_Cos'] = np.cos(2 * np.pi * ticker_df['Month'] / 12)
        
        # ============================================
        # 7. LAG FEATURES (for sequence models)
        # ============================================
        
        for lag in [1, 2, 3, 5, 10]:
            ticker_df[f'Close_Lag_{lag}'] = ticker_df['Adj Close'].shift(lag)
            ticker_df[f'Volume_Lag_{lag}'] = ticker_df['Volume'].shift(lag)
            ticker_df[f'Return_Lag_{lag}'] = ticker_df['Daily_Return'].shift(lag)
        
        # ============================================
        # 8. TARGET VARIABLES (for prediction)
        # ============================================
        
        # Next day prediction targets
        ticker_df['Target_Next_Close'] = ticker_df['Adj Close'].shift(-1)
        ticker_df['Target_Next_Return'] = ticker_df['Daily_Return'].shift(-1)
        
        # Multi-day prediction targets
        for days in [3, 5, 10]:
            ticker_df[f'Target_{days}d_Return'] = (ticker_df['Adj Close'].shift(-days) / ticker_df['Adj Close'] - 1) * 100
        
        # Binary classification targets
        ticker_df['Target_Direction'] = (ticker_df['Target_Next_Return'] > 0).astype(int)
        ticker_df['Target_Up_5pct'] = (ticker_df['Target_Next_Return'] > 0.05).astype(int)
        
        processed_dfs.append(ticker_df)
        print(f"✅ {len(ticker_df.columns)} features")
    
    # Combine all tickers
    final_df = pd.concat(processed_dfs, ignore_index=True)
    
    return final_df


def prepare_ml_dataset(df, target_col='Target_Next_Return', drop_first_n=200):
    """
    Prepare dataset for ML/DL training with proper train/test split
    """
    print(f"\n📊 Preparing ML Dataset with target: {target_col}")
    
    # Remove rows with NaN in target or features (due to rolling windows)
    df_clean = df.dropna(subset=[target_col])
    
    # Remove initial rows where rolling features are still being calculated
    df_clean = df_clean.groupby('Ticker').apply(lambda x: x.iloc[drop_first_n:]).reset_index(drop=True)
    
    # Separate features and target
    exclude_cols = ['Date', 'Ticker'] + [col for col in df_clean.columns if col.startswith('Target_')]
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    # Time-based split (80/20)
    split_idx = int(len(df_clean) * 0.8)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Metadata
    dates_train = df_clean['Date'].iloc[:split_idx]
    dates_test = df_clean['Date'].iloc[split_idx:]
    tickers_train = df_clean['Ticker'].iloc[:split_idx]
    tickers_test = df_clean['Ticker'].iloc[split_idx:]
    
    print(f"✅ Features: {len(feature_cols)}")
    print(f"✅ Training samples: {len(X_train)}")
    print(f"✅ Test samples: {len(X_test)}")
    print(f"📅 Train period: {dates_train.min()} to {dates_train.max()}")
    print(f"📅 Test period: {dates_test.min()} to {dates_test.max()}")
    
    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'dates_train': dates_train, 'dates_test': dates_test,
        'tickers_train': tickers_train, 'tickers_test': tickers_test,
        'feature_names': feature_cols
    }


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Load your dataset
    input_file = "Selected_FAANG_NVDA_TSLA_2022-01-01_to_2025-10-01.csv"
    
    print("="*80)
    print("🚀 STOCK PREDICTION FEATURE ENGINEERING PIPELINE")
    print("="*80)
    
    # Read data
    print(f"\n📁 Loading: {input_file}")
    df = pd.read_csv(input_file)
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Convert price and volume columns to numeric
    print("🔧 Converting data types...")
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with invalid data
    initial_rows = len(df)
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    if len(df) < initial_rows:
        print(f"⚠️  Removed {initial_rows - len(df)} rows with invalid data")
    
    print(f"✅ Data cleaned: {len(df)} valid rows")
    
    # Engineer features
    df_features = engineer_features(df)
    
    # Save enhanced dataset
    output_file = input_file.replace('.csv', '_ENHANCED.csv')
    df_features.to_csv(output_file, index=False)
    
    print("\n" + "="*80)
    print(f"✅ Enhanced dataset saved: {output_file}")
    print(f"📊 Total features: {len(df_features.columns)}")
    print(f"📈 Ready for: ML, DL, Time Series, XAI")
    print("="*80)
    
    # Prepare ML-ready dataset
    ml_data = prepare_ml_dataset(df_features, target_col='Target_Next_Return')
    
    print("\n💡 RECOMMENDED MODELS:")
    print("   📈 ML: XGBoost, LightGBM, Random Forest, CatBoost")
    print("   🧠 DL: LSTM, GRU, Transformer, CNN-LSTM")
    print("   🔍 XAI: SHAP, LIME, Feature Importance, Attention Weights")
    print("="*80)