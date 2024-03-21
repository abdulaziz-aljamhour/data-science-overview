import pandas   as pd
import numpy    as np

from typing                     import Annotated, Union
from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import StandardScaler

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

def scale(df: pd.DataFrame) -> Union[Annotated[pd.DataFrame, 'df_scaled'],
                                     Annotated[StandardScaler, 'scaler']]:
    scaler = StandardScaler()
    scaler.fit(df)
    scaled_data = scaler.transform(df)
    df_scaled = pd.DataFrame(scaled_data, columns=df.columns)
    return df_scaled, scaler

def split_data(X: np.ndarray, y: np.ndarray, test_size: float=.3) -> Union[
    Annotated[np.ndarray, 'X_train'],
    Annotated[np.ndarray, 'X_test'],
    Annotated[np.ndarray, 'y_train'],
    Annotated[np.ndarray, 'y_test']]:
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                         test_size=test_size)
    return X_train, X_test, y_train, y_test

def get_X_y(df: pd.DataFrame, y_name='csmpa') -> Union[
    Annotated[np.ndarray, 'X'],
    Annotated[np.ndarray, 'y']]:
    X = df.copy()
    y = X.pop(y_name)
    return X.values, y.values

def preprocess_data(df: pd.DataFrame) -> Union[
    Annotated[np.ndarray, 'X_train'],
    Annotated[np.ndarray, 'X_test'],
    Annotated[np.ndarray, 'y_train'],
    Annotated[np.ndarray, 'y_test'],
    Annotated[StandardScaler, 'scaler']]:
    df_renamed = rename_columns(df)
    df_scaled, scaler = scale(df_renamed)
    X, y = get_X_y(df_scaled)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return X_train, X_test, y_train, y_test, scaler