# General Settings
RANDOM_SEED = 42
DEVICE = 'cuda'  # or 'cpu'

# Data Settings
DATA_DIR = "data/"
DATASETS = {
    'UNSW1': 'unsw1.csv',
    'UNSW2': 'unsw2.csv',
    'IoT': 'IOT.csv'
}
DROP_COLS = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_SRC_PORT', 'L4_DST_PORT', 'Attack']

# Preprocessing Settings
SCALER_TYPE = 'minmax'

# Model Hyperparameters
DNN_DROPOUT_RATE = 0.2
LSTM_DROPOUT_RATE = 0.2
RF_N_ESTIMATORS = 50
RF_MAX_DEPTH = 10
DT_MAX_DEPTH = 10

# Training Settings
N_FOLDS = 3
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-5
PATIENCE = 15

# Federated Learning Settings
FL_ROUNDS = 8
FL_LOCAL_EPOCHS = 3