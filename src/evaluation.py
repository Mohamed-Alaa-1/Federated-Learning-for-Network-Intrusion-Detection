import logging
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from typing import Dict, List
class EvaluationMetrics:
    """Comprehensive evaluation metrics calculation."""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        logger = logging.getLogger(__name__)
        
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'true_positive': tp,
                'true_negative': tn,
                'false_positive': fp,
                'false_negative': fn
            }
            
            if y_pred_proba is not None and len(np.unique(y_true)) > 1:
                try:
                    metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
                except ValueError as e:
                    logger.warning(f"Could not calculate AUC-ROC: {str(e)}")
                    metrics['auc_roc'] = np.nan
            else:
                metrics['auc_roc'] = np.nan
                
            logger.debug(f"Calculated metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise
    
    @staticmethod
    def aggregate_cv_results(cv_results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Aggregate cross-validation results with statistics."""
        logger = logging.getLogger(__name__)
        
        metrics = {}
        for metric in cv_results[0].keys():
            values = [result[metric] for result in cv_results if not np.isnan(result[metric])]
            if values:
                metrics[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            else:
                metrics[metric] = {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
        
        logger.debug(f"Aggregated CV results for {len(cv_results)} folds")
        return metrics