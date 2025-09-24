import time
from datetime import datetime
import logging

import config  # Import all settings
from src.utils import setup_logging
from src.data_processing import DataLoader
from src.training.centralized import CentralizedTrainer
from src.training.federated import FederatedTrainer
from src.visualization import ResultsAnalyzer

def run_experiment():
    """Main execution function to run the entire NIDS experiment."""
    logger, log_filename = setup_logging()
    
    logger.info("="*80)
    logger.info("NETWORK INTRUSION DETECTION SYSTEM EVALUATION")
    logger.info(f"Using device: {config.DEVICE}")
    logger.info("="*80)

    # --- Initialize components ---
    data_loader = DataLoader(data_dir=config.DATA_DIR)
    centralized_trainer = CentralizedTrainer()
    federated_trainer = FederatedTrainer()
    results_analyzer = ResultsAnalyzer()
    
    # --- Load Data ---
    datasets = data_loader.load_all_datasets()
    if not datasets:
        logger.error("No datasets found. Exiting.")
        return

    all_results = {}
    
    # --- PHASE 1: Centralized Training ---
    # ... (Your centralized training loop, now much cleaner)
    # This loop will call centralized_trainer.train_model_cv(...)

    # --- PHASE 2: Federated Learning ---
    # ... (Your federated training logic)
    # This will call federated_trainer.federated_averaging_cv(...)

    # --- PHASE 3: Results Analysis & Visualization ---
    results_analyzer.print_results_table(all_results)
    results_analyzer.save_results_to_csv(all_results)
    results_analyzer.visualizer.plot_model_comparison(all_results)
    # ... etc.

    logger.info(f"Evaluation complete. Full log at {log_filename}")

if __name__ == "__main__":
    run_experiment()