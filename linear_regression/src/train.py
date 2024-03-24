import logging
import os

from sklearn.linear_model   import LinearRegression
from sklearn.metrics        import mean_squared_error
from preprocessor           import get_data
from joblib                 import dump

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

MODELS_PATH = os.path.join('./models/')


X_train, X_test, y_train, y_test = get_data()
regr = LinearRegression()
logging.info(f'Training a Linear Regression model')
regr.fit(X_train, y_train)
logging.info('Training complete')
training_pred = regr.predict(X_train)
training_mse = mean_squared_error(y_train, training_pred)
logging.info(f'Model MSE training score: {training_mse}')
logging.info(f'Testing regression model')
y_hat = regr.predict(X_test)
testing_mse = mean_squared_error(y_test, y_hat)
logging.info(f'Model testing MSE score: {testing_mse}')
logging.info(f'Testing complete')
regr_path = os.path.join(MODELS_PATH, 'regr.joblib')
try:
    logging.info(f'Saving model on file, path={regr_path}')
    dump(regr, regr_path)
    logging.info(f'model saved...')
except Exception as e:
    logging.critical(f'Error has occured while trying to save model: {e}')