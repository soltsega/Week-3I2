import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Load and prepare the data
def load_and_prepare_data(filepath):
    # Load data
    df = pd.read_csv(filepath)
    
    # Basic data cleaning
    df = df.drop_duplicates()
    
    # Convert date columns to datetime if needed
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
    
    # Filter for policies with claims for severity modeling
    df_claims = df[df['TotalClaims'] > 0].copy()
    
    return df, df_claims


    # Setup the feature engineering function
    