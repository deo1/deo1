import pandas as pd

# =============================================================================
# Constants
# =============================================================================
DATA_PATH = "./data/"
LOGS_PATH = DATA_PATH + "user_logs.csv"
LOGS_CHUNK_PATH = DATA_PATH + "/user_logs/"
CHUNK_SIZE = 10**6

# =============================================================================
# Load and process data
# =============================================================================

test = next(pd.read_csv(LOGS_PATH, chunksize=CHUNK_SIZE))
