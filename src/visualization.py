import logging
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
class Visualizer:
    """Enhanced visualization class for model performance and learning curves."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figures directory
        self.fig_dir = Path("figures")
        self.fig_dir.mkdir(exist_ok=True)
    
    def plot_model_comparison(self, all_results: Dict[str, Dict[str, Any]], save_path: str = None):
        """Create comprehensive model comparison plots."""
        self.logger.info("Creating model comparison visualizations")
        
        # Prepare data for plotting
        models = []
        datasets = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        auc_scores = []
        accuracy_stds = []
        f1_stds = []
        
        for model_name, result in all_results.items():
            metrics = result['aggregated_metrics']
            models.append(model_name.replace('_', '\n'))
            datasets.append(result.get('dataset', 'Combined'))
            accuracies.append(metrics['accuracy']['mean'])
            precisions.append(metrics['precision']['mean'])
            recalls.append(metrics['recall']['mean'])
            f1_scores.append(metrics['f1_score']['mean'])
            auc_scores.append(metrics['auc_roc']['mean'] if not np.isnan(metrics['auc_roc']['mean']) else 0)
            accuracy_stds.append(metrics['accuracy']['std'])
            f1_stds.append(metrics['f1_score']['std'])
        
        # Create comprehensive comparison plot
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Accuracy comparison with error bars
        ax1 = plt.subplot(2, 3, 1)
        bars1 = plt.bar(models, accuracies, yerr=accuracy_stds, capsize=5, 
                        color=sns.color_palette("husl", len(models)), alpha=0.8)
        plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc, std in zip(bars1, accuracies, accuracy_stds):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                     f'{acc:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 2. F1-Score comparison with error bars
        ax2 = plt.subplot(2, 3, 2)
        bars2 = plt.bar(models, f1_scores, yerr=f1_stds, capsize=5,
                        color=sns.color_palette("husl", len(models)), alpha=0.8)
        plt.title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('F1-Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, f1, std in zip(bars2, f1_scores, f1_stds):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                     f'{f1:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 3. Precision vs Recall scatter plot
        ax3 = plt.subplot(2, 3, 3)
        scatter = plt.scatter(recalls, precisions, c=accuracies, s=150, 
                              cmap='viridis', alpha=0.7, edgecolors='black')
        plt.colorbar(scatter, label='Accuracy')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision vs Recall', fontsize=14, fontweight='bold')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        # Add model labels to points
        for i, model in enumerate(models):
            plt.annotate(model.replace('\n', '_'), (recalls[i], precisions[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # 4. Multi-metric radar chart (normalized)
        ax4 = plt.subplot(2, 3, 4, projection='polar')
        
        # Prepare data for radar chart - take top 5 models by accuracy
        top_indices = np.argsort(accuracies)[-5:]
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
        
        colors = sns.color_palette("husl", len(top_indices))
        
        for i, idx in enumerate(top_indices):
            values = [accuracies[idx], precisions[idx], recalls[idx], 
                      f1_scores[idx], max(0, auc_scores[idx])]  # Ensure non-negative AUC
            values = np.concatenate((values, [values[0]]))  # Complete the circle
            
            ax4.plot(angles, values, 'o-', linewidth=2, label=models[idx].replace('\n', '_'), color=colors[i])
            ax4.fill(angles, values, alpha=0.1, color=colors[i])
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(metrics_names)
        ax4.set_ylim(0, 1)
        ax4.set_title('Top 5 Models - Multi-Metric Comparison', fontsize=12, fontweight='bold', pad=20)
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 5. Dataset-wise performance
        ax5 = plt.subplot(2, 3, 5)
        dataset_performance = {}
        for model_name, result in all_results.items():
            dataset = result.get('dataset', 'Combined')
            if dataset not in dataset_performance:
                dataset_performance[dataset] = []
            dataset_performance[dataset].append(result['aggregated_metrics']['accuracy']['mean'])
        
        dataset_names = list(dataset_performance.keys())
        dataset_means = [np.mean(perfs) for perfs in dataset_performance.values()]
        dataset_stds = [np.std(perfs) for perfs in dataset_performance.values()]
        
        bars5 = plt.bar(dataset_names, dataset_means, yerr=dataset_stds, capsize=5,
                        color=sns.color_palette("Set2", len(dataset_names)), alpha=0.8)
        plt.title('Average Performance by Dataset', fontsize=14, fontweight='bold')
        plt.ylabel('Average Accuracy', fontsize=12)
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, mean, std in zip(bars5, dataset_means, dataset_stds):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                     f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 6. Model type comparison (group by algorithm type)
        ax6 = plt.subplot(2, 3, 6)
        
        # Group models by type
        model_types = {}
        for model_name, result in all_results.items():
            if 'DNN' in model_name or 'LSTM' in model_name or 'Federated' in model_name:
                model_type = 'Deep Learning'
            elif 'RandomForest' in model_name:
                model_type = 'Random Forest'
            elif 'DecisionTree' in model_name:
                model_type = 'Decision Tree'
            else:
                model_type = 'Other'
            
            if model_type not in model_types:
                model_types[model_type] = []
            model_types[model_type].append(result['aggregated_metrics']['accuracy']['mean'])
        
        type_names = list(model_types.keys())
        type_means = [np.mean(accs) for accs in model_types.values()]
        type_stds = [np.std(accs) for accs in model_types.values()]
        
        bars6 = plt.bar(type_names, type_means, yerr=type_stds, capsize=5,
                        color=sns.color_palette("Set3", len(type_names)), alpha=0.8)
        plt.title('Performance by Algorithm Type', fontsize=14, fontweight='bold')
        plt.ylabel('Average Accuracy', fontsize=12)
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, mean, std in zip(bars6, type_means, type_stds):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                     f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.fig_dir / f"model_comparison_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        self.logger.info(f"Model comparison plot saved to {save_path}")
        plt.show()
        
        return save_path
    
    def plot_learning_curves(self, centralized_trainer, federated_trainer=None, save_path: str = None):
        """Plot learning curves for deep learning models."""
        self.logger.info("Creating learning curves visualization")
        
        # Collect all learning curves
        all_curves = {}
        if hasattr(centralized_trainer, 'learning_curves'):
            all_curves.update(centralized_trainer.learning_curves)
        
        if federated_trainer and hasattr(federated_trainer, 'learning_curves'):
            all_curves.update(federated_trainer.learning_curves)
        
        if not all_curves:
            self.logger.warning("No learning curves available for plotting")
            return None
        
        # Create learning curves plot
        n_models = len(all_curves)
        if n_models == 0:
            return None
            
        fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 10))
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        
        colors = sns.color_palette("husl", 3)  # For train, val, and average
        
        for idx, (model_name, curves_list) in enumerate(all_curves.items()):
            # Average across folds
            if isinstance(curves_list, list) and len(curves_list) > 0:
                # Handle different curve structures
                if 'train_losses' in curves_list[0]:
                    # Individual fold curves
                    max_epochs = max(len(fold['train_losses']) for fold in curves_list)
                    
                    avg_train_losses = []
                    avg_val_losses = []
                    avg_train_accs = []
                    avg_val_accs = []
                    
                    for epoch in range(max_epochs):
                        epoch_train_losses = [fold['train_losses'][epoch] for fold in curves_list 
                                              if epoch < len(fold['train_losses'])]
                        epoch_val_losses = [fold['val_losses'][epoch] for fold in curves_list 
                                            if epoch < len(fold['val_losses'])]
                        epoch_train_accs = [fold['train_accuracies'][epoch] for fold in curves_list 
                                            if epoch < len(fold['train_accuracies'])]
                        epoch_val_accs = [fold['val_accuracies'][epoch] for fold in curves_list 
                                          if epoch < len(fold['val_accuracies'])]
                        
                        avg_train_losses.append(np.mean(epoch_train_losses) if epoch_train_losses else np.nan)
                        avg_val_losses.append(np.mean(epoch_val_losses) if epoch_val_losses else np.nan)
                        avg_train_accs.append(np.mean(epoch_train_accs) if epoch_train_accs else np.nan)
                        avg_val_accs.append(np.mean(epoch_val_accs) if epoch_val_accs else np.nan)
                else:
                    # Already averaged curves
                    avg_train_losses = curves_list[0].get('train_losses', [])
                    avg_val_losses = curves_list[0].get('val_losses', [])
                    avg_train_accs = curves_list[0].get('train_accuracies', [])
                    avg_val_accs = curves_list[0].get('val_accuracies', [])
            else:
                continue
            
            # Plot Loss curves
            ax_loss = axes[0, idx]
            epochs = range(1, len(avg_train_losses) + 1)
            
            ax_loss.plot(epochs, avg_train_losses, label='Training Loss', 
                         color=colors[0], linewidth=2, marker='o', markersize=4)
            ax_loss.plot(epochs, avg_val_losses, label='Validation Loss', 
                         color=colors[1], linewidth=2, marker='s', markersize=4)
            
            ax_loss.set_title(f'{model_name} - Loss Curves', fontsize=12, fontweight='bold')
            ax_loss.set_xlabel('Epoch', fontsize=10)
            ax_loss.set_ylabel('Loss', fontsize=10)
            ax_loss.legend()
            ax_loss.grid(True, alpha=0.3)
            
            # Plot Accuracy curves
            ax_acc = axes[1, idx]
            ax_acc.plot(epochs, avg_train_accs, label='Training Accuracy', 
                        color=colors[0], linewidth=2, marker='o', markersize=4)
            ax_acc.plot(epochs, avg_val_accs, label='Validation Accuracy', 
                        color=colors[1], linewidth=2, marker='s', markersize=4)
            
            ax_acc.set_title(f'{model_name} - Accuracy Curves', fontsize=12, fontweight='bold')
            ax_acc.set_xlabel('Epoch', fontsize=10)
            ax_acc.set_ylabel('Accuracy', fontsize=10)
            ax_acc.legend()
            ax_acc.grid(True, alpha=0.3)
            ax_acc.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.fig_dir / f"learning_curves_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        self.logger.info(f"Learning curves plot saved to {save_path}")
        plt.show()
        
        return save_path
    
    def plot_confusion_matrices(self, all_results: Dict[str, Dict[str, Any]], save_path: str = None):
        """Plot confusion matrices for all models."""
        self.logger.info("Creating confusion matrices visualization")
        
        # Calculate average confusion matrices
        n_models = len(all_results)
        if n_models == 0:
            return None
            
        cols = min(4, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (model_name, result) in enumerate(all_results.items()):
            if idx >= len(axes):
                break
                
            # Calculate average confusion matrix across folds
            cv_results = result['cv_results']
            avg_cm = np.zeros((2, 2))
            
            for fold_result in cv_results:
                tp = fold_result['true_positive']
                tn = fold_result['true_negative']
                fp = fold_result['false_positive']
                fn = fold_result['false_negative']
                cm = np.array([[tn, fp], [fn, tp]])
                avg_cm += cm
            
            avg_cm /= len(cv_results)
            
            # Plot confusion matrix
            ax = axes[idx]
            sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Blues', ax=ax,
                        xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
            ax.set_title(f'{model_name.replace("_", " ")}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=9)
            ax.set_ylabel('Actual', fontsize=9)
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.fig_dir / f"confusion_matrices_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        self.logger.info(f"Confusion matrices plot saved to {save_path}")
        plt.show()
        
        return save_path

class ResultsAnalyzer:
    """Enhanced results analysis and visualization."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.visualizer = Visualizer()
    
    def print_results_table(self, all_results: dict[str, dict[str, any]]):
        """Print formatted results table."""
        self.logger.info("Generating comprehensive results table")
        
        print("\n" + "="*100)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("="*100)
        
        # Create results DataFrame
        results_data = []
        for model_name, result in all_results.items():
            metrics = result['aggregated_metrics']
            row = {
                'Model': model_name,
                'Dataset': result.get('dataset', 'Combined'),
                'Accuracy': f"{metrics['accuracy']['mean']:.4f} ± {metrics['accuracy']['std']:.4f}",
                'Precision': f"{metrics['precision']['mean']:.4f} ± {metrics['precision']['std']:.4f}",
                'Recall': f"{metrics['recall']['mean']:.4f} ± {metrics['recall']['std']:.4f}",
                'F1-Score': f"{metrics['f1_score']['mean']:.4f} ± {metrics['f1_score']['std']:.4f}",
                'AUC-ROC': f"{metrics['auc_roc']['mean']:.4f} ± {metrics['auc_roc']['std']:.4f}" if not np.isnan(metrics['auc_roc']['mean']) else 'N/A'
            }
            results_data.append(row)
            
            # Log detailed results for each model
            self.logger.info(f"Model: {model_name}")
            self.logger.info(f"  Dataset: {result.get('dataset', 'Combined')}")
            self.logger.info(f"  Accuracy: {metrics['accuracy']['mean']:.4f} ± {metrics['accuracy']['std']:.4f}")
            self.logger.info(f"  Precision: {metrics['precision']['mean']:.4f} ± {metrics['precision']['std']:.4f}")
            self.logger.info(f"  Recall: {metrics['recall']['mean']:.4f} ± {metrics['recall']['std']:.4f}")
            self.logger.info(f"  F1-Score: {metrics['f1_score']['mean']:.4f} ± {metrics['f1_score']['std']:.4f}")
            if not np.isnan(metrics['auc_roc']['mean']):
                self.logger.info(f"  AUC-ROC: {metrics['auc_roc']['mean']:.4f} ± {metrics['auc_roc']['std']:.4f}")
            if 'training_time' in metrics:
                self.logger.info(f"  Training Time: {metrics['training_time']['total']:.2f}s (avg: {metrics['training_time']['mean']:.2f}s per fold)")
        
        results_df = pd.DataFrame(results_data)
        print(results_df.to_string(index=False))
        
        self.logger.info("Results table generated and displayed")
    
    def save_results_to_csv(self, all_results: dict[str, dict[str, any]], filename: str = None):
        """Save detailed results to CSV with timestamp."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'nids_results_pytorch_{timestamp}.csv'
        
        self.logger.info(f"Saving detailed results to {filename}")
        
        detailed_results = []
        
        for model_name, result in all_results.items():
            for fold_idx, fold_result in enumerate(result['cv_results']):
                row = {
                    'Model': model_name,
                    'Dataset': result.get('dataset', 'Combined'),
                    'Fold': fold_idx + 1,
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    **fold_result
                }
                detailed_results.append(row)
        
        results_df = pd.DataFrame(detailed_results)
        results_df.to_csv(filename, index=False)
        
        self.logger.info(f"Detailed results saved to {filename}")
        self.logger.info(f"Saved {len(detailed_results)} result records")
        
        return filename
    
    def save_summary_to_csv(self, all_results: dict[str, dict[str, any]], filename: str = None):
        """Save summary results to CSV."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'nids_summary_{timestamp}.csv'
        
        self.logger.info(f"Saving summary results to {filename}")
        
        summary_results = []
        
        for model_name, result in all_results.items():
            metrics = result['aggregated_metrics']
            row = {
                'Model': model_name,
                'Dataset': result.get('dataset', 'Combined'),
                'Accuracy_Mean': metrics['accuracy']['mean'],
                'Accuracy_Std': metrics['accuracy']['std'],
                'Precision_Mean': metrics['precision']['mean'],
                'Precision_Std': metrics['precision']['std'],
                'Recall_Mean': metrics['recall']['mean'],
                'Recall_Std': metrics['recall']['std'],
                'F1_Score_Mean': metrics['f1_score']['mean'],
                'F1_Score_Std': metrics['f1_score']['std'],
                'AUC_ROC_Mean': metrics['auc_roc']['mean'] if not np.isnan(metrics['auc_roc']['mean']) else None,
                'AUC_ROC_Std': metrics['auc_roc']['std'] if not np.isnan(metrics['auc_roc']['std']) else None,
                'Training_Time_Total': metrics.get('training_time', {}).get('total', None),
                'Training_Time_Mean': metrics.get('training_time', {}).get('mean', None),
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            summary_results.append(row)
        
        summary_df = pd.DataFrame(summary_results)
        summary_df.to_csv(filename, index=False)
        
        self.logger.info(f"Summary results saved to {filename}")
        return filename