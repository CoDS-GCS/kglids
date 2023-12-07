import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from memory_profiler import memory_usage
import sys
sys.path.append(r'C:\Users\niki_\OneDrive\Documents\GitHub\kglids')
from api.api import KGLiDS


RANDOM_STATE = 30
RANDOM_STATE_SAMPLING = 30
np.random.seed(RANDOM_STATE)


def test_dataset(filename):
    start = time.time()
    kglids = KGLiDS(endpoint='localhost', port=5821)
    name = filename[:-4]
    df = pd.read_csv("experiments/OnDemandDataPrep_experiments/Transformation_datasets/"+filename)
    fold = 1
    f1_per_fold = []
    accuracy_per_fold = []
    label_dict = {'Ionosphere':'column_ai','Sonar':'Class','Pimaindianssubset':'Outcome','abalone':'Rings',
                  'banknote_authentication':'class','dermatology':'class','ecoli':'class','fertility_Diagnosis':'diagnosis',
                  'haberman':'survival','letter_recognition':'letter','libras':'class','poker':'CLASS','shuttle':'class',
                  'waveform':'class','wine':'class','featurefourier':'class','featurepixel':'class','opticaldigits':'class'}

    for train_index, test_index in StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(df, df[
        [label_dict[name]]]): # 5-fold cross-validation
        train_set = df.iloc[train_index]
        test_set = df.iloc[test_index]
        recommended_transformations = kglids.recommend_transformation_operations(train_set, name)

        train_set = kglids.apply_transformation_operations(train_set, recommended_transformations,label_dict[name])

        test_set = kglids.apply_transformation_operations(test_set, recommended_transformations,label_dict[name])

        X_train = train_set.loc[:, train_set.columns != label_dict[name]]
        y_train = train_set[label_dict[name]]
        X_test = test_set.loc[:, test_set.columns != label_dict[name]]
        y_test = test_set[label_dict[name]]

        model = RandomForestClassifier(random_state=RANDOM_STATE)
        model.fit(X=X_train, y=y_train)
        y_out = model.predict(X=X_test)
        f1 = f1_score(y_true=y_test, y_pred=y_out, average='macro')
        f1_per_fold.append(f1)
        accuracy = accuracy_score(y_test, y_out)
        accuracy_per_fold.append(accuracy)
        fold = fold + 1

    time_taken = f'{(time.time() - start):.2f}'
    f1_per_dataset = f'{np.mean(f1_per_fold):.4f}'
    accuracy_per_dataset = f'{np.mean(accuracy_per_fold):.4f}'
    return f1_per_dataset, accuracy_per_dataset, time_taken



if __name__ == '__main__':
    file_list = [ 'ecoli.csv', 'featurefourier.csv','featurepixel.csv', 'fertility_Diagnosis.csv',
                'haberman.csv', "Ionosphere.csv", 'letter_recognition.csv', 'libras.csv',"Pimaindianssubset.csv",
                'opticaldigits.csv', 'poker.csv', 'shuttle.csv', "Sonar.csv", 'waveform.csv', 'wine.csv']
    for filename in file_list:
        results = memory_usage((test_dataset, (filename,)), retval=True)
        max_memory = max(results[0])
        f1 = results[1][0]
        accuracy = results[1][1]
        total_time = results[1][2]
        with open("experiments/OnDemandDataPrep_experiments/Results/final_result_transformation_test.txt", "a") as f:
            f.write(f"{filename},{f1},{accuracy},{total_time},{max_memory}\n")