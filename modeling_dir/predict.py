import os
import joblib
import xgboost as xgb
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PremiumOptimizer:
    """A class to optimize insurance premiums using trained ML models."""
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the premium optimizer with trained models.
        
        Args:
            model_dir: Directory containing the model files. If None, uses 'models' in the same directory as this script.
        """
        # Set up model directory
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).parent / 'models'
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Model paths
        self.xgb_path = self.model_dir / 'xgboost_model.json'
        self.rf_path = self.model_dir / 'claim_probability_model.pkl'
        
        # Initialize models
        self.severity_model = None
        self.prob_model = None
        self.severity_features = None
        self.prob_features = None
        self.load_models()
    
    def load_models(self) -> None:
        """Load the trained models from disk and their feature names."""
        try:
            # Check if model files exist
            if not self.xgb_path.exists():
                raise FileNotFoundError(f"XGBoost model not found at {self.xgb_path}")
            if not self.rf_path.exists():
                raise FileNotFoundError(f"Random Forest model not found at {self.rf_path}")
            
            logger.info(f"Loading models from {self.model_dir}")
            
            # Load XGBoost model
            self.severity_model = xgb.Booster()
            self.severity_model.load_model(str(self.xgb_path))
            
            # Load Random Forest model
            self.prob_model = joblib.load(self.rf_path)
            
            # Load or define feature names
            self.load_feature_names()
            
            logger.info("Successfully loaded both models and their feature names")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def load_feature_names(self):
        """Load or infer feature names for both models."""
        try:
            # For scikit-learn model
            if hasattr(self.prob_model, 'feature_names_in_'):
                self.prob_features = list(self.prob_model.feature_names_in_)
            else:
                # Fallback if feature names not available
                self.prob_features = [
                    'CalculatedPremiumPerTerm',
                    'CoverCategory_freq',
                    'CoverGroup_freq',
                    'CoverType_freq',
                    'ExcessSelected_freq',
                    'Section_freq',
                    'SumInsured',
                    'TotalPremium'
                ]
            
            # For XGBoost model
            try:
                self.severity_features = self.severity_model.feature_names
            except:
                # Fallback to prob_features if XGBoost feature names not available
                self.severity_features = self.prob_features.copy()
                
            logger.info(f"Probability model features: {self.prob_features}")
            logger.info(f"Severity model features: {self.severity_features}")
            
        except Exception as e:
            logger.error(f"Error loading feature names: {str(e)}")
            raise
    
    def preprocess_input(self, input_data: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess input data for prediction with correct feature order and transformations."""
        try:
        # Convert input to DataFrame
        df = pd.DataFrame([input_data])
        
        # Add log transformations if needed
        if 'TotalPremium' in df.columns and 'log_TotalPremium' not in df.columns:
            df['log_TotalPremium'] = np.log1p(df['TotalPremium'])
        if 'TotalClaims' in df.columns and 'log_TotalClaims' not in df.columns:
            df['log_TotalClaims'] = np.log1p(df['TotalClaims'])
        
        # Ensure all required columns exist
        missing_severity = set(self.severity_features) - set(df.columns)
        missing_prob = set(self.prob_features) - set(df.columns)
        
        if missing_severity:
            logger.warning(f"Missing features for severity model: {missing_severity}")
        if missing_prob:
            logger.warning(f"Missing features for probability model: {missing_prob}")
        
        # Ensure all required columns exist
        all_required = set(self.severity_features + self.prob_features)
        missing_cols = all_required - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required features: {missing_cols}")
        
        # Reorder columns to match training data
        severity_df = df[self.severity_features]
        prob_df = df[self.prob_features]
        
    return severity_df, prob_df
        
        except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, float]:
        """Make predictions for a single policy."""
        try:
            # Preprocess input
            severity_features, prob_features = self.preprocess_input(input_data)
            
            # Make predictions
            claim_prob = self.prob_model.predict_proba(prob_features)[0][1]
            claim_severity = self.severity_model.predict(xgb.DMatrix(severity_features))[0]
            
            # Calculate premium
            expense_loading = 0.20
            profit_margin = 0.10
            base_premium = claim_prob * claim_severity
            optimal_premium = base_premium * (1 + expense_loading + profit_margin)
            
            return {
                'claim_probability': float(claim_prob),
                'expected_claim_severity': float(claim_severity),
                'base_risk_premium': float(base_premium),
                'recommended_premium': float(optimal_premium),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

def main():
    """Main function to demonstrate the PremiumOptimizer usage."""
    try:
        # Initialize predictor
        predictor = PremiumOptimizer()
        
        # Example input - update with your actual feature values
        example_input = {
            'CalculatedPremiumPerTerm': 1000.0,
            'CoverCategory_freq': 0.5,
            'CoverGroup_freq': 0.3,
            'CoverType_freq': 0.4,
            'ExcessSelected_freq': 0.2,
            'Section_freq': 0.1,
            'SumInsured': 50000.0,
            'TotalPremium': 1200.0
        }
        
        print("\nExpected feature order for probability model:", predictor.prob_features)
        print("Expected feature order for severity model:", predictor.severity_features)
        
        # Make prediction
        result = predictor.predict(example_input)
        
        # Print results
        if result.get('status') == 'success':
            print("\nPremium Optimization Results:")
            for key, value in result.items():
                if key != 'status':
                    print(f"{key.replace('_', ' ').title()}: {value:,.2f}")
        else:
            print(f"\nError: {result.get('message', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()