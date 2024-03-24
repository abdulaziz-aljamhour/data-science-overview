import pandas   as pd
import numpy    as np

from typing                     import Annotated, Union, Tuple
from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import StandardScaler
from joblib                     import dump

import logging
import os

logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s - %(levelname)s - %(message)s')


MODELS_PATH = os.path.join('./models/')

def load_data(path: str='./data/Concrete_Data_Yeh.csv') -> \
    Annotated[pd.DataFrame, 'df']:
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f'error while trying to load in data: {e}')
     
def rename_columns(df: pd.DataFrame, mapper=None) -> \
    Annotated[pd.DataFrame, 'df_renamed']:
    if mapper != None:
        df_renamed = df.rename(mapper=mapper, axis=1)
    else:
        df_renamed = df.copy()
        df_renamed.columns = df_renamed.columns.str.lower()
    return df_renamed

def scale(X: np.ndarray) -> Annotated[np.ndarray, 'X_scaled']:
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    dump(scaler, os.path.join(MODELS_PATH, 'scaler.joblib'))
    return X_scaled

def split_data(X: np.ndarray, y: np.ndarray, test_size: float=.3) -> Tuple[
    Annotated[np.ndarray, 'X_train'],
    Annotated[np.ndarray, 'X_test'],
    Annotated[np.ndarray, 'y_train'],
    Annotated[np.ndarray, 'y_test']]:
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                         test_size=test_size)
    return X_train, X_test, y_train, y_test

def get_X_y(df: pd.DataFrame, y_name='csmpa') -> Tuple[
    Annotated[np.ndarray, 'X'],
    Annotated[np.ndarray, 'y']]:
    X = df.copy()
    y = X.pop(y_name)
    return X.values, y.values

def preprocess_data(df: pd.DataFrame) -> Tuple[
    Annotated[np.ndarray, 'X_train'],
    Annotated[np.ndarray, 'X_test'],
    Annotated[np.ndarray, 'y_train'],
    Annotated[np.ndarray, 'y_test'],]:
    df_renamed = rename_columns(df)
    X, y = get_X_y(df_renamed)
    X = scale(X)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return X_train, X_test, y_train, y_test

def get_data(path: str=None) -> Tuple[
    Annotated[np.ndarray, 'X_train'],
    Annotated[np.ndarray, 'X_test'],
    Annotated[np.ndarray, 'y_train'],
    Annotated[np.ndarray, 'y_test'],
]:
    # load in data
    if path == None:
        df = load_data()
    else:
        df = load_data(path)
    # preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(df)
    return X_train, X_test, y_train, y_test 


    

    