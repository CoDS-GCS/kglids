from collections import defaultdict
import io
import pickle

import pandas as pd
import stardog
from tqdm import tqdm

def query_stardog(dataset_name, estimator) -> pd.DataFrame: 
    estimator_uris = {
        'RandomForestClassifier': '<http://kglids.org/resource/library/sklearn/ensemble/RandomForestClassifier>',
        'RandomForestRegressor': '<http://kglids.org/resource/library/sklearn/ensemble/RandomForestRegressor>',
        'GradientBoostingClassifier': '<http://kglids.org/resource/library/sklearn/ensemble/GradientBoostingClassifier>',
        'GradientBoostingRegressor': '<http://kglids.org/resource/library/sklearn/ensemble/GradientBoostingRegressor>'
    }
    
    QUERY = """
    prefix pipeline: <http://kglids.org/ontology/pipeline/>
    prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    prefix kglids: <http://kglids.org/ontology/>
    
    SELECT ?pipeline ?votes ?hp ?hpValue
    WHERE
    {
    ?pipeline a kglids:Pipeline.
    ?pipeline kglids:isPartOf <http://kglids.org/resource/kaggle/DATASET_NAME_PLACEHOLDER>.
    ?pipeline pipeline:hasVotes ?votes.
    graph ?pipeline {
        ?import_statement a kglids:Statement.
        ?import_statement pipeline:inControlFlow <http://kglids.org/resource/import>.
        ?import_statement pipeline:callsClass ESTIMATOR_URI_PLACEHOLDER.
        ?model_statement a kglids:Statement.
        ?model_statement pipeline:callsClass ESTIMATOR_URI_PLACEHOLDER.
        <<?model_statement pipeline:hasParameter ?hp>> pipeline:withParameterValue ?hpValue.
    }
    }
    ORDER BY ?hp DESC(?votes)   
    """
    
    ENDPOINT = 'http://localhost:5858'
    DATABASE_NAME = 'kglids_graphs_for_kgpip_training_datasets_full'
    QUERY = QUERY.replace('DATASET_NAME_PLACEHOLDER', dataset_name).replace('ESTIMATOR_URI_PLACEHOLDER', 
                                                                            estimator_uris[estimator])


    conn_details = {
        'endpoint': ENDPOINT,
        'username': 'admin',
        'password': 'admin'
    }
    conn = stardog.Connection(DATABASE_NAME, **conn_details)

    csv_results = conn.select(QUERY, content_type='text/csv')
    df = pd.read_csv(io.BytesIO(csv_results))
    return df


def main():
    
    FLAML_REGRESSORS = {'rf': 'RandomForestRegressor', 'xgboost': 'GradientBoostingRegressor'}
    FLAML_CLASSIFIERS = {'rf': 'RandomForestClassifier', 'xgboost': 'GradientBoostingClassifier'}
    FLAML_SUPPORTED_HPS = {'rf': ['n_estimators', 'max_leaf_nodes', 'criterion'],
                           'xgboost': ['n_estimators', 'max_leaf_nodes', 'min_weight_fraction_leaf', 'learning_rate', 
                                       'subsample', 'colsample_bylevel', 'colsample_bytree', 'reg_alpha', 'reg_lambda']}
    SKLEARN_TO_XGBOOST_MAP = {'max_leaf_nodes': 'max_leaves',
                              'min_weight_fraction_leaf': 'min_child_weight'}
        
    training_sets = pd.read_csv('kgpip_training_datasets.csv')
    
    dataset_hp_config = defaultdict(dict)
    for idx, row in tqdm(training_sets.iterrows(), total=len(training_sets)):
        if row['is_regression']:
            target_estimators = FLAML_REGRESSORS
        else:
            target_estimators = FLAML_CLASSIFIERS
        
        for estimator in target_estimators:
            records = query_stardog(row['dataset'], target_estimators[estimator])
            # select value by most common instead of most votes
            records = records[['hp', 'hpValue']].groupby('hp').agg(lambda x: x.mode().tolist()[0]).reset_index(drop=False)
            # hyper parameter configuration dict for this dataset
            hp_dict = {i['hp']: i['hpValue'] for _,i in records.iterrows()}
            
            # drop non-flaml hyperparameters
            for hp in list(hp_dict):
                if hp not in FLAML_SUPPORTED_HPS[estimator]:
                    hp_dict.pop(hp)
            # map sklearn hps to xgboost hps
            for hp in SKLEARN_TO_XGBOOST_MAP:
                if hp in hp_dict:
                    hp_dict[SKLEARN_TO_XGBOOST_MAP[hp]] = hp_dict[hp]
                    hp_dict.pop(hp)
            
            # convert to float if possible
            for hp in list(hp_dict):
                try:
                    val = float(hp_dict[hp])
                    # convert to int if possible
                    if float(val) == int(val):
                        val = int(val)
                except:
                    val = hp_dict[hp]
                hp_dict[hp] = val
            
            dataset_hp_config[row['dataset']][estimator] = hp_dict
    
    with open('dataset_hp_config_dict_from_kglids_full.pickle', 'wb') as f:
        pickle.dump(dataset_hp_config, f)


if __name__ == '__main__':
    main()