import glob
import json
import os
import pickle

import numpy as np
from tqdm import tqdm

from model.column_data_type import ColumnDataType
from config import profiler_config


def main():
    profiles_path = profiler_config.output_path
    profile_paths = glob.glob(os.path.join(profiles_path, '**/*.json'), recursive=True)
    
    embedding_types = [ColumnDataType.DATE.value, ColumnDataType.INT.value, ColumnDataType.FLOAT.value,
                       ColumnDataType.NATURAL_LANGUAGE_NAMED_ENTITY.value, ColumnDataType.NATURAL_LANGUAGE_TEXT.value,
                       ColumnDataType.STRING.value]
    table_embeddings_per_dtype = {}
    table_embeddings_count = {}
    for profile_path in tqdm(profile_paths):
        with open(profile_path, 'r') as f:
            profile = json.load(f)
        
        if profile['data_type'] == ColumnDataType.BOOLEAN.value:
            continue
        if profile['table_id'] not in table_embeddings_per_dtype:
            table_embeddings_per_dtype[profile['table_name']] = {dtype: np.zeros(len(profile['embedding'])) for dtype in embedding_types}
            table_embeddings_count[profile['table_name']] = {dtype: 0 for dtype in embedding_types}
            
        table_embeddings_per_dtype[profile['table_name']][profile['data_type']] += np.array(profile['embedding'])
        table_embeddings_count[profile['table_name']][profile['data_type']] += 1
    
    table_embeddings = {}
    for table in table_embeddings_per_dtype.keys():
        for dtype in embedding_types:
            if table_embeddings_count[table][dtype] != 0:
                table_embeddings_per_dtype[table][dtype] /= table_embeddings_count[table][dtype]
        table_embeddings[table] = np.concatenate([table_embeddings_per_dtype[table][dtype] for dtype in embedding_types])
    with open(os.path.expanduser('~/projects/kglids/storage/embeddings/smaller_real_table_embeddings.pickle'), 'wb') as f:
        pickle.dump(table_embeddings, f)


if __name__ == '__main__':
    main()