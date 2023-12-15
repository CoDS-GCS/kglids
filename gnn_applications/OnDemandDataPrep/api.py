from api.helpers.helper import *
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, FunctionTransformer


from gnn_applications.OnDemandDataPrep.Modeling.prepare_for_encoding import profile_to_csv, create_encoding_file
from gnn_applications.OnDemandDataPrep.Modeling.encoding import encode
from gnn_applications.OnDemandDataPrep.Modeling.embeddings_from_profile import *
from gnn_applications.OnDemandDataPrep.Modeling.create_cleaning_model import graphSaint as graphSaint_modeling_cleaning
from gnn_applications.OnDemandDataPrep.Modeling.create_scaling_model import graphSaint as graphSaint_modeling_scaler
from gnn_applications.OnDemandDataPrep.Modeling.create_unary_model import graphSaint as graphSaint_modeling_unary
from gnn_applications.OnDemandDataPrep.kglids import create_triplets
from gnn_applications.OnDemandDataPrep.script_transform_biokg_to_ogb_datasets import triplet_encoding
from gnn_applications.OnDemandDataPrep.inference_cleaning import graphSaint as graphSaint_cleaning
from gnn_applications.OnDemandDataPrep.inference_scaling import graphSaint as graphSaint_scaling
from gnn_applications.OnDemandDataPrep.inference_unary import graphSaint as graphSaint_unary
from gnn_applications.OnDemandDataPrep.apply_recommendation import apply_cleaning


class KGLiDSDataPrep:
    def __init__(self, endpoint: str = 'http://localhost:7200', db: str = 'kglids'):
        self.conn = connect_to_graphdb(endpoint, graphdb_repo=db)

    def build_cleaning_model(self, graph_name: str):
        # profile_to_csv(graph_name)
        create_encoding_file(graph_name, 'cleaning')
        encode(graph_name, "http://kglids.org/ontology/pipeline/HasCleaningOperation", "2Table")
        get_table_embeddings_cleaning(graph_name)
        get_column_embeddings(graph_name)
        graphSaint_modeling_cleaning(graph_name)

    def build_scaler_model(self, graph_name: str):
        # profile_to_csv(graph_name)
        # create_encoding_file(graph_name, 'scaler_transformation')
        # encode(graph_name, "http://kglids.org/ontology/pipeline/HasScalingTransformation", "2Table")
        get_table_embeddings_scaling(graph_name)
        get_column_embeddings(graph_name)
        graphSaint_modeling_scaler(graph_name)

    def build_unary_model(self, graph_name: str):
        # profile_to_csv(graph_name)
        # create_encoding_file(graph_name, 'unary_transformation')
        # encode(graph_name, "http://kglids.org/ontology/pipeline/HasUnaryTransformation", "1Column")
        get_column_embeddings_unary(graph_name)
        graphSaint_modeling_unary(graph_name)

    def recommend_cleaning_operations(self, table: pd.DataFrame, name: str = 'Cleaning_dataset'):
        cleaning_dict = {0: 'Fill', 1: 'Interpolate', 2: 'IterativeImputer', 3: 'KNNImputer', 4: 'SimpleImputer'}
        create_triplets(table, name)
        triplet_encoding(name, 'Table')
        cleaning_op = graphSaint_cleaning(table, name)
        return cleaning_dict[cleaning_op]

    def apply_cleaning_operations(self, operation: str, df: pd.DataFrame):
        clean_df = apply_cleaning(df, operation)
        return clean_df

    def recommend_transformation_operations(self, table: pd.DataFrame, name: str = 'Transformation_dataset'):
        scaling_dict = {0: 'MinMaxScaler', 1: 'RobustScaler', 2: 'StandardScaler'}
        unary_dict = {1: 'Log', 2: 'Sqrt', 0: 'NoUnary'}
        create_triplets(table, name)
        triplet_encoding(name, 'Table')
        scaling_op = graphSaint_scaling(name, table)
        triplet_encoding(name, 'Column')
        unary_op = graphSaint_unary(name, table)
        data = {'Recommended_transformation': [scaling_op],
                'Feature': [None]}
        recommended_scaling_transformations = pd.DataFrame(data)
        recommended_scaling_transformations['Recommended_transformation'] = recommended_scaling_transformations[
            'Recommended_transformation'].replace(scaling_dict)

        df_unary_rec = pd.DataFrame(unary_op, columns=['Recommended_transformation'])
        df_unary_col = pd.read_csv(
            'gnn_applications/OnDemandDataPrep/storage/' + name + '_gnn_Column/mapping/Column_entidx2name.csv')
        df_unary_join = df_unary_col.join(df_unary_rec)
        recommended_unary_transformations = df_unary_join.groupby('Recommended_transformation')['ent name'].apply(
            list).reset_index()
        recommended_unary_transformations.rename(columns={'ent name': 'Feature'}, inplace=True)
        recommended_unary_transformations['Recommended_transformation'] = recommended_unary_transformations[
            'Recommended_transformation'].replace(unary_dict)
        recommended_transformations = pd.concat(
            [recommended_scaling_transformations, recommended_unary_transformations], ignore_index=True)
        return recommended_transformations

    def apply_transformation_operations(self, X: pd.DataFrame, recommended_transformations: pd.DataFrame,
                                        label_name: str = 'None'):
        for n, recommendation in recommended_transformations.to_dict('index').items():
            if recommendation['Recommended_transformation'] != 'NoUnary':
                transformation = recommendation['Recommended_transformation']
                feature = recommendation['Feature']

                if transformation in {'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'QuantileTransformer',
                                      'PowerTransformer'}:
                    # if label_name != 'None': # If we want exclude the label from being transformed
                    #     X = X.drop(columns=[label_name])

                    print(f'Applying {transformation}')  # on {list(X.columns)}')
                    if transformation == 'StandardScaler':
                        scaler = StandardScaler()
                    elif transformation == 'MinMaxScaler':
                        scaler = MinMaxScaler()
                    elif transformation == 'RobustScaler':
                        scaler = RobustScaler()
                    else:
                        scaler = RobustScaler()
                    numerical_columns = X.select_dtypes(include=['int', 'float']).columns
                    numerical_columns_wo_label = [col for col in numerical_columns if col != label_name]
                    X[numerical_columns_wo_label] = scaler.fit_transform(X[numerical_columns_wo_label])

                elif transformation in {'Log', 'Sqrt', 'square'}:
                    print(f'Applying {transformation}')  # on {list(feature)}')
                    if transformation == 'Log':
                        def log_plus_const(x, const=0):
                            return np.log(x + np.abs(const) + 0.0001)

                        for f in tqdm(feature):
                            if str(f) != label_name and pd.api.types.is_numeric_dtype(X[str(f)].dtype):
                                min_neg_val = X[str(f)].min()
                                unary_transformation_model = FunctionTransformer(func=log_plus_const,
                                                                                 kw_args={'const': min_neg_val},
                                                                                 validate=True)

                                X[str(f)] = unary_transformation_model.fit_transform(
                                    X=np.array(X[str(f)]).reshape(-1, 1))

                    elif transformation == 'Sqrt':
                        def sqrt_plus_const(x, const=0):
                            return np.sqrt(x + np.abs(const) + 0.0001)

                        for f in tqdm(feature):
                            if str(f) != label_name and pd.api.types.is_numeric_dtype(X[str(f)].dtype):
                                min_neg_val = X[str(f)].min()
                                unary_transformation_model = FunctionTransformer(func=sqrt_plus_const,
                                                                                 kw_args={'const': min_neg_val},
                                                                                 validate=True)
                                X[str(f)] = unary_transformation_model.fit_transform(
                                    X=np.array(X[str(f)]).reshape(-1, 1))
                    else:
                        unary_transformation_model = FunctionTransformer(func=np.square, validate=True)
                        X[feature] = unary_transformation_model.fit_transform(X=X[feature])
                else:
                    raise ValueError(f'{transformation} not supported')

        return X