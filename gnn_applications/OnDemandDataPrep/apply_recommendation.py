import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer


def encode(X):
    categorical_cols = X.select_dtypes(include=['object']).columns
    print('categorical_cols', categorical_cols)
    le = LabelEncoder()
    for col in categorical_cols:
        X[col] = X[col].fillna('xxx')
        non_placeholder_mask = (X[col] != 'xxx')
        X.loc[non_placeholder_mask, col] = le.fit_transform(X.loc[non_placeholder_mask, col])
        X[col] = X[col].replace('xxx', float('nan'))

    return X

def get_columns_to_be_cleaned(df: pd.DataFrame):
    for na_type in {'none', 'n/a', 'na', 'NaN', 'nan', 'missing', '?', '', ' '}:
        if na_type in {'?', '', ' '}:
            df.replace(na_type, np.nan, inplace=True)
        else:
            df.replace(r'(^)' + na_type + r'($)', np.nan, inplace=True, regex=True)
    columns = pd.DataFrame(df.isnull().sum())
    columns.columns = ['Missing values']
    columns['Feature'] = columns.index
    columns = columns[columns['Missing values'] > 0]
    columns.sort_values(by='Missing values', ascending=False, inplace=True)
    columns.reset_index(drop=True, inplace=True)
    return columns


def check_for_uncleaned_features(df: pd.DataFrame):  # clean by recommendations
    uncleaned_features = list(get_columns_to_be_cleaned(df=df)['Feature'])
    # print('check:', df.isna().sum())
    if len(uncleaned_features) == 0:
        print('\nall features look clean')
    else:
        print(f'\n{uncleaned_features} are still uncleaned')
        return 1


def apply_cleaning(df, cleaning_op):
    columns_to_be_cleaned = get_columns_to_be_cleaned(df)
    object_cols = df.select_dtypes(include=['object']).columns
    X_clean = pd.DataFrame()

    if cleaning_op == 'Fill':
        def fillna_mean(column):
            mean = column.mean()
            return column.fillna(mean)

        if len(object_cols) > 0:
            X_clean[object_cols] = df[object_cols].fillna(value='x')
            X_clean = X_clean.fillna(value=0)

        else:
            X_clean = df.apply(fillna_mean)
        X_clean.reset_index(drop=True, inplace=True)


    # Interpolate
    elif cleaning_op == 'Interpolate':
        # If there are string columns, apply pad
        if len(object_cols) > 0:
            print('non num', df.isna().sum())
            df = df.interpolate(method='pad', axis=1)
            check = check_for_uncleaned_features(df=df)
            count = 0

            while check == 1 and count < 10:
                count = count + 1
                if count % 2 == 0:
                    df.interpolate(method='pad', limit=None, inplace=True)
                    check = check_for_uncleaned_features(df=df)
                else:
                    df.interpolate(method='pad', limit=None, limit_direction='backward', inplace=True)
                    check = check_for_uncleaned_features(df=df)
        else:
            df.interpolate(axis=1, inplace=True)
            check = check_for_uncleaned_features(df=df)
            count = 0

            while check == 1 and count < 10:
                print(df.isnull())
                count = count + 1
                if count % 2 == 0:
                    df.interpolate(method='linear', limit=None, inplace=True)
                    check = check_for_uncleaned_features(df=df)
                else:
                    df.interpolate(method='linear', limit=None, limit_direction='backward', inplace=True)
                    check = check_for_uncleaned_features(df=df)
        X_clean = df

    elif cleaning_op == 'SimpleImputer':
        X_clean = pd.DataFrame(SimpleImputer(strategy='most_frequent').fit_transform(df), columns=df.columns)

    elif cleaning_op == 'KNNImputer':
        X_clean = encode(df)
        X_clean = pd.DataFrame(KNNImputer().fit_transform(X_clean), columns=X_clean.columns)


    elif cleaning_op == 'IterativeImputer':
        X_clean = encode(df)
        X_clean = pd.DataFrame(IterativeImputer().fit_transform(X_clean), columns=X_clean.columns)


    return X_clean
