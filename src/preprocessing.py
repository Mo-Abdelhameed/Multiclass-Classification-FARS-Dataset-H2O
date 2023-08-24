import pandas as pd
import numpy as np
import warnings
from imblearn.over_sampling import SMOTE
from feature_engine.encoding import OneHotEncoder
from utils import read_csv_in_directory
from config import paths
from joblib import load, dump

warnings.filterwarnings('ignore')


def handle_class_imbalance(transformed_data: pd.DataFrame, transformed_labels: pd.Series):
    """
    Handle class imbalance using SMOTE.

    Args:
        transformed_data (pd.DataFrame): The transformed data.
        transformed_labels (pd.Series): The transformed labels.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing the balanced data and
            balanced labels.
    """
    # Adjust k_neighbors parameter for SMOTE
    # set k_neighbors to be the smaller of two values:
    #       1 and,
    #       the number of instances in the minority class minus one
    k_neighbors = min(
        1, sum(transformed_labels == min(transformed_labels.value_counts().index)) - 1
    )
    smote = SMOTE(k_neighbors=k_neighbors, random_state=0)
    balanced_data, balanced_labels = smote.fit_resample(
        transformed_data, transformed_labels
    )
    return balanced_data, balanced_labels


def impute_with_probability(data: pd.DataFrame, column: str):
    value_counts = data[column].value_counts()
    counts = value_counts.values
    index = value_counts.index

    probabilities = counts / counts.sum()

    imputation = np.random.choice(index.tolist(), size=data[column].isna().sum(), p=probabilities)
    missing = data[(data[column].isna())]
    missing[column] = imputation
    data.update(missing)


def impute_with_mode(data: pd.DataFrame, column: str):
    mode = data[column].mode()[0]
    data[column] = data[column].fillna(mode)
    return data, mode


def impute(df: pd.DataFrame, training: bool):
    columns_with_missing = df.columns[df.isna().any()]
    if training:
        imputation_values = {}
        for c in columns_with_missing:
            if c == 'numoccs':
                value = df['numoccs'].mean()
                df['numoccs'].fillna(value, inplace=True)
            else:
                df, value = impute_with_mode(df, c)
            imputation_values[c] = value
        dump(imputation_values, paths.IMPUTATION_VALUES_FILE)
    else:
        imputation_values = load(paths.IMPUTATION_VALUES_FILE)
        for c in columns_with_missing:
            df[c].fillna(imputation_values[c], inplace=True)
    return df


def preprocess(data_dir: str = paths.TRAIN_DIR, training: bool = True) -> pd.DataFrame:
    """
    Performs data preprocessing.
    Args:
        data_dir (str): Path to the directory containing the data.
        training (bool): Indicates whether preprocessing is for training or testing.

    Returns: (pd.DataFrame) The preprocessed data.
    """
    df = read_csv_in_directory(data_dir)
    if training:
        target = df['driver_factor']
        df = df.drop(columns='driver_factor')

    df = impute(df, training)

    if training:
        df['driver_factor'] = target
        target = df['driver_factor']
        df = df.drop(columns='driver_factor')
        encoder = OneHotEncoder(top_categories=6, drop_last=True, drop_last_binary=True)
        encoder.fit(df)
        dump(encoder, paths.ENCODER_FILE)
    else:
        encoder = load(paths.ENCODER_FILE)

    df = encoder.transform(df)

    if training:
        df['driver_factor'] = target
        x = df.drop(columns='driver_factor')
        y = df['driver_factor']
        x, y = handle_class_imbalance(x, y)
        x['driver_factor'] = y
        df = x
    return df
