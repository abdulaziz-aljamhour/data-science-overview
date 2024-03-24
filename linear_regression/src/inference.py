from joblib         import load
from preprocessor   import get_data

import numpy as np

import os
import logging

logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s - %(levelname)s - %(message)s')

np.random.seed(2447)
MODELS_PATH = os.path.join('./models/')

def select_random_record(X: np.ndarray, n: int=1) -> np.ndarray:
    '''Sample an `n` amount of random rows from X'''
    num_rows = X.shape[0]
    idx = np.random.randint(num_rows, size=n)
    rows = X[idx, :]
    return rows

def main():
    model = load(os.path.join(MODELS_PATH, 'regr.joblib'))
    scaler = load(os.path.join(MODELS_PATH, 'scaler.joblib')) # data is scaled from preprocessing but there should be a seperate preprocesssing for inference
    sample_data = get_data()[1] # data is prepped
    X = select_random_record(sample_data, 5)
    logging.info(f'inferencing on the following {X.shape[0]} records')
    logging.info(X)
    y_hat = model.predict(X)
    logging.info(f'prediction => {y_hat}')
    logging.info(f'prediction size => {y_hat.shape}')
    

if __name__ == '__main__':
    main()