# API Reference

## `calculate_optimal_premium`
```python
def calculate_optimal_premium(
    X_risk: pd.DataFrame, 
    X_prob: pd.DataFrame,
    severity_model: xgb.Booster,
    prob_model: RandomForestClassifier,
    expense_loading: float = 0.20,
    profit_margin: float = 0.10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate optimal insurance premiums.
    
    Args:
        X_risk: Features for claim severity prediction
        X_prob: Features for claim probability prediction
        severity_model: Trained XGBoost model
        prob_model: Trained Random Forest model
        expense_loading: Expense loading factor (default: 0.20)
        profit_margin: Desired profit margin (default: 0.10)
        
    Returns:
        Tuple of (optimal_premiums, base_risk_premiums)
    """