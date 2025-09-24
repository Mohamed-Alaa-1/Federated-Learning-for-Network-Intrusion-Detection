class DataLoader:
    """Enhanced data loading and preprocessing utilities."""
    
    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path.cwd()
        self.drop_cols = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_SRC_PORT', 'L4_DST_PORT', 'Attack']
        self.logger = logging.getLogger(__name__)
        
    def load_dataset(self, filename: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load and preprocess a single dataset."""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            self.logger.error(f"Dataset not found: {filepath}")
            raise FileNotFoundError(f"Dataset not found: {filepath}")
            
        self.logger.info(f"Loading dataset: {filename}")
        
        try:
            df = pd.read_csv(filepath)
            original_shape = df.shape
            df = df.fillna(0)  # Handle missing values
            
            # Extract labels and features
            y = df['Label'].values
            X = df.drop(self.drop_cols + ['Label'], axis=1, errors='ignore')
            
            unique_labels, label_counts = np.unique(y, return_counts=True)
            self.logger.info(f"Dataset {filename} loaded successfully:")
            self.logger.info(f"  Original shape: {original_shape}")
            self.logger.info(f"  Final feature shape: {X.shape}")
            self.logger.info(f"  Class distribution: {dict(zip(unique_labels, label_counts))}")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error loading {filename}: {str(e)}")
            raise
    
    def load_all_datasets(self) -> Dict[str, Tuple[pd.DataFrame, np.ndarray]]:
        """Load all datasets with progress tracking."""
        datasets = {}
        filenames = {
            'UNSW1': 'unsw1.csv',
            'UNSW2': 'unsw2.csv', 
            'IoT': 'IOT.csv'
        }
        
        self.logger.info("Starting dataset loading process...")
        print("Loading datasets...")
        
        for name, filename in tqdm(filenames.items(), desc="Loading datasets", unit="dataset"):
            try:
                datasets[name] = self.load_dataset(filename)
                self.logger.info(f"Successfully loaded dataset: {name}")
            except FileNotFoundError:
                self.logger.warning(f"Dataset {name} ({filename}) not found, skipping...")
                continue
            except Exception as e:
                self.logger.error(f"Failed to load dataset {name}: {str(e)}")
                continue
        
        self.logger.info(f"Dataset loading completed. Loaded {len(datasets)} datasets: {list(datasets.keys())}")
        return datasets
    
    @staticmethod
    def preprocess_data(X: pd.DataFrame, y: np.ndarray, scaler_type: str = 'minmax') -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess features and labels."""
        logger = logging.getLogger(__name__)
        
        # Convert to numpy and handle inf/nan values
        X_processed = np.nan_to_num(X.values.astype(np.float32), nan=0.0, posinf=1e10, neginf=-1e10)
        y_processed = y.astype(np.float32)
        
        logger.debug(f"Data preprocessing completed: X_shape={X_processed.shape}, y_shape={y_processed.shape}")
        return X_processed, y_processed

    @staticmethod
    def custom_under_sample_5_to_1(X: np.ndarray, y: np.ndarray, random_state: int = RANDOM_SEED) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform custom under-sampling to achieve 5:1 ratio (majority:minority).
        """
        logger = logging.getLogger(__name__)
        
        # Get class distribution
        unique_classes, counts = np.unique(y, return_counts=True)
        class_counts = dict(zip(unique_classes, counts))
        
        logger.info(f"Before custom 5:1 under-sampling: {class_counts}")
        
        if len(unique_classes) != 2:
            logger.warning("Custom 5:1 sampling designed for binary classification. Proceeding with available classes.")
        
        # Identify minority and majority classes
        minority_class = unique_classes[np.argmin(counts)]
        majority_class = unique_classes[np.argmax(counts)]
        
        # Get indices for each class
        minority_indices = np.where(y == minority_class)[0]
        majority_indices = np.where(y == majority_class)[0]
        
        # Calculate target sample sizes (5:1 ratio)
        minority_count = len(minority_indices)
        target_majority_count = minority_count * 5
        
        # If majority class has fewer samples than target, use all available
        if len(majority_indices) < target_majority_count:
            target_majority_count = len(majority_indices)
            logger.warning(f"Majority class has only {len(majority_indices)} samples, using all available.")
        
        # Random sampling
        np.random.seed(random_state)
        selected_majority_indices = np.random.choice(
            majority_indices, 
            size=target_majority_count, 
            replace=False
        )
        
        # Combine indices
        selected_indices = np.concatenate([minority_indices, selected_majority_indices])
        np.random.shuffle(selected_indices)  # Shuffle to avoid class ordering
        
        X_resampled = X[selected_indices]
        y_resampled = y[selected_indices]
        
        # Log results
        final_unique, final_counts = np.unique(y_resampled, return_counts=True)
        final_class_counts = dict(zip(final_unique, final_counts))
        ratio = final_counts[1] / final_counts[0] if len(final_counts) > 1 else 0
        
        logger.info(f"After custom 5:1 under-sampling: {final_class_counts}")
        logger.info(f"Achieved ratio (majority:minority): {ratio:.2f}:1")
        logger.info(f"Final dataset shape: X={X_resampled.shape}, y={y_resampled.shape}")
        
        return X_resampled, y_resampled