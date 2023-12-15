import sys
from memory_profiler import memory_usage
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
import pandas as pd
import time
sys.path.append(r'C:\Users\niki_\OneDrive\Documents\GitHub\kglids')
from api.api import KGLiDS
RANDOM_STATE = 30
RANDOM_STATE_SAMPLING = 30
np.random.seed(RANDOM_STATE)


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


def ml(X_train, X_test, y_train, y_test, ml_type):
    if ml_type == 'binary':
        score_f1_r2, score_acc_mse = ml_binary_classification(X_train, X_test, y_train, y_test)
    elif ml_type == 'multi-class':
        score_f1_r2, score_acc_mse = ml_multiclass_classification(X_train, X_test, y_train, y_test)
    return score_f1_r2, score_acc_mse


def ml_multiclass_classification(X_train, X_test, y_train, y_test):
    rfc = RandomForestClassifier(random_state=RANDOM_STATE)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    return f1, accuracy


def ml_binary_classification(X_train, X_test, y_train, y_test):
    rfc = RandomForestClassifier(random_state=RANDOM_STATE)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    return f1, accuracy

def ml_regression(X, y):
    rfc = RandomForestRegressor(random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    score_f1_r2 = cross_val_score(rfc, X, y, cv=cv, scoring='r2').mean()
    scoring = make_scorer(mean_squared_error)
    scores = cross_val_score(rfc, X, y, cv=cv, scoring=scoring)
    score_acc_mse = np.mean(scores)
    return score_f1_r2, score_acc_mse

def get_columns_to_be_cleaned(df: pd.DataFrame):
    for na_type in {'none', 'n/a', 'na', 'NaN', 'nan', 'missing', '?', '', ' '}:
        if na_type in {'?', '', ' '}:
            df.replace(na_type, np.nan, inplace=True)
        else:
            df.replace(r'(^)' + na_type + r'($)', np.nan, inplace=True, regex=True)
    columns = pd.DataFrame(df.isnull().sum())
    print('columns', columns)
    columns.columns = ['Missing values']
    columns['Feature'] = columns.index
    columns = columns[columns['Missing values'] > 0]
    columns.sort_values(by='Missing values', ascending=False, inplace=True)
    columns.reset_index(drop=True, inplace=True)
    return columns


def check_for_uncleaned_features(df: pd.DataFrame):  # clean by recommendations
    uncleaned_features = list(get_columns_to_be_cleaned(df=df)['Feature'])
    print('check:', df.isna().sum())
    if len(uncleaned_features) == 0:
        print('\nall features look clean')
    else:
        print(f'\n{uncleaned_features} are still uncleaned')
        return 1

def test_dataset(data_fn, dependent_variable, ml_type):
    kglids = KGLiDS(endpoint='localhost', port=5821)
    le = LabelEncoder()
    start = time.time()
    accuracy_per_fold = []
    f1_per_fold = []
    mem_usage_max = []
    df = pd.read_csv('experiments/OnDemandDataPrep_experiments/Cleaning_datasets/'+data_fn + '.csv')

    fold = 1
    print(df)
    recommended_cleaning = kglids.recommend_cleaning_operations(df, data_fn)

    df = kglids.apply_cleaning_operations(recommended_cleaning, df)
    for train_index, test_index in StratifiedKFold(n_splits=10, shuffle=True,
                                                   random_state=RANDOM_STATE).split(df, df[dependent_variable]):  # 10-fold cross-validation
        print(data_fn,": Fold ",fold)
        train_set = df.iloc[train_index]
        test_set = df.iloc[test_index]
        X_train = train_set.loc[:, train_set.columns != dependent_variable]
        y_train = train_set[dependent_variable]
        X_test = test_set.loc[:, test_set.columns != dependent_variable]
        y_test = test_set[dependent_variable]
        f1, acc = ml(X_train, X_test, y_train, y_test, ml_type)
        accuracy_per_fold.append(acc)
        f1_per_fold.append(f1)
        fold = fold + 1

    time_taken = f'{(time.time() - start):.2f}'
    f1_per_dataset = f'{np.mean(f1_per_fold):.4f}'
    accuracy_per_dataset = f'{np.mean(accuracy_per_fold):.4f}'
    return f1_per_dataset, accuracy_per_dataset,time_taken

if __name__ == '__main__':
    file_list = ["adult.csv", "breastcancerwisconsin.csv", "cleveland_heart_disease.csv", "credit.csv", "credit-g.csv",
                 "hepatitis.csv", "horsecolic.csv", "housevotes84.csv", "jm1.csv", "titanic.csv", "higgs.csv",
                 "APSFailure.csv", "albert.csv"]

    ml_type_dict = {}
    dependent_variable_dict = {}
    df = pd.read_csv('experiments/OnDemandDataPrep_experiments/datasets2.csv')
    for filename in file_list:
        row = df[df['Data cleaning dataset'] == filename]

        # Check if the row is not empty
        if not row.empty:
            ml_type_dict[filename] = row['Task'].values[0]  # Access the first element of the Series
            dependent_variable_dict[filename] = row['dependent_variable'].values[0]

    for filename in file_list:
        name = filename[:-4]
        print(type(dependent_variable_dict[filename]))
        print(ml_type_dict[filename])
        results = memory_usage((test_dataset, (name,dependent_variable_dict[filename],ml_type_dict[filename],)), retval=True)
        max_memory = max(results[0])
        f1 = results[1][0]
        accuracy = results[1][1]
        total_time = results[1][2]
        print(f"{name},{f1},{accuracy},{total_time},{max_memory}\n")
        with open("experiments/OnDemandDataPrep_experiments/Results/final_result_cleaning_test.txt", "a") as f:
            f.write(f"{name},{f1},{accuracy},{total_time},{max_memory}\n")
