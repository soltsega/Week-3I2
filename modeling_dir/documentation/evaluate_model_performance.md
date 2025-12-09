def evaluate_model_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_type: str = "regression"
) -> Dict[str, float]:
    """
    Evaluate model performance metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        model_type: Type of model ("regression" or "classification")
        
    Returns:
        Dictionary of performance metrics
    """