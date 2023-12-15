import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, FunctionTransformer

from api.template import *
from api.helpers.helper import *
from collections import Counter
from tqdm import tqdm
from OnDemandDataPrep.Modeling.prepare_for_encoding import profile_to_csv, create_encoding_file
from OnDemandDataPrep.Modeling.encoding import encode
from OnDemandDataPrep.Modeling.embeddings_from_profile import *
from OnDemandDataPrep.Modeling.create_cleaning_model import graphSaint as graphSaint_modeling_cleaning
from OnDemandDataPrep.Modeling.create_scaling_model import graphSaint as graphSaint_modeling_scaler
from OnDemandDataPrep.Modeling.create_unary_model import graphSaint as graphSaint_modeling_unary
from OnDemandDataPrep.kglids import create_triplets
from OnDemandDataPrep.script_transform_biokg_to_ogb_datasets import triplet_encoding
from OnDemandDataPrep.inference_cleaning import graphSaint as graphSaint_cleaning
from OnDemandDataPrep.inference_scaling import graphSaint as graphSaint_scaling
from OnDemandDataPrep.inference_unary import graphSaint as graphSaint_unary
from OnDemandDataPrep.apply_recommendation import apply_cleaning

class KGLiDS:
    def __init__(self, endpoint: str = 'http://localhost:7200', db: str = 'kglids'):
        self.conn = connect_to_graphdb(endpoint, graphdb_repo=db)
        

    def get_datasets_info(self, show_query: bool = False):
        return get_datasets_info(self.conn, show_query).sort_values('Dataset', ignore_index=True, ascending=True)

    def get_tables_info(self, dataset: str = '', show_query: bool = False):
        if not dataset:
            print('Showing all available table(s): ')
        else:
            print("Showing table(s) for '{}' dataset: ".format(dataset))
        return get_tables_info(self.conn, dataset, show_query).sort_values('Dataset', ignore_index=True, ascending=True)

    def recommend_k_joinable_tables(self, table: pd.Series, k: int = 5, show_query: bool = False):
        if not isinstance(table, pd.Series):
            raise ValueError("table needs to be a type 'pd.Series'")
        # if not isinstance(dataset, str):
        #     raise ValueError("dataset needs to be a type 'str'")
        if not isinstance(k, int):
            raise ValueError("k needs to be a type 'int'")
        elif isinstance(table, pd.Series) and isinstance(k, int):
            dataset = table["Dataset"]
            table = table["Table"]
            recommendations = recommend_tables(self.conn, dataset, table, k, 'data:hasContentSimilarity',
                                               show_query)
            recommendations['Score'] = list(map(lambda x: round(x, 2), [float(i) / max(recommendations['Score'].
                                                       tolist()) for i in (recommendations['Score'].tolist())]))

            print('Showing the top-{} joinable table recommendations:'.format(len(recommendations)))
            return recommendations

    def recommend_k_unionable_tables(self, table: pd.Series, k: int = 5, show_query: bool = False):
        if not isinstance(table, pd.Series):
            raise ValueError("table needs to be a type 'pd.Series'")
        # if not isinstance(dataset, str):
        #     raise ValueError("dataset needs to be a type 'str'")
        if not isinstance(k, int):
            raise ValueError("k needs to be a type 'int'")
        elif isinstance(table, pd.Series) and isinstance(k, int):
            dataset = table["Dataset"]
            table = table["Table"]
            recommendations = recommend_tables(self.conn, dataset, table, k, 'data:hasLabelSimilarity',
                                               show_query)
            recommendations['Score'] = list(map(lambda x: round(x, 2), [float(i) / max(recommendations['Score'].
                                                tolist()) for i in (recommendations['Score'].tolist())]))

            print('Showing the top-{} unionable table recommendations:'.format(len(recommendations)))
            return recommendations

    def get_table_info(self, table, show_query: bool = False):
        dataset = table["Dataset"]
        if 'Recommended_table' in table.keys():
            table = table["Recommended_table"]
        else:
            table = table["Table"]
        return get_table_info(self.conn, dataset, table, show_query)

    def show_graph_info(self, show_query: bool = False):
        print('Information captured: ')
        return show_graph_info(self.conn, show_query)

    def search_tables_on(self, conditions: list, show_query: bool = False):

        def parsed_conditions(user_conditions):
            error_message = 'conditions need to be in encapsulated in list.\n' \
                            'lists in the list are associated by an \"and\" condition.\n' \
                            'String in each tuple will be joined by an \"or\" condition.\n' \
                            ' For instance [[a,b],[c]]'
            if not isinstance(user_conditions, list):
                raise TypeError(error_message)
            else:
                for l in user_conditions:
                    if not isinstance(l, list):
                        raise TypeError(error_message)
                    else:
                        for s in l:
                            if not isinstance(s, str):
                                raise TypeError(error_message)

            i = 1
            filters = []
            statements = []
            for t in user_conditions:
                sts = '?column' + str(i) + ' rdf:type kglids:Column.' \
                                           '\n?column' + str(i) + ' kglids:isPartOf ?table.' \
                                                                  '\n?column' + str(
                    i) + ' rdfs:label ?label' + str(i) + '.'

                statements.append(sts)
                or_conditions = '|'.join(t)
                regex = 'regex(?label' + str(i) + ', "' + or_conditions + '", "i")'
                filters.append(regex)
                i += 1
            return '\n'.join(statements), ' && '.join(filters)

        data = search_tables_on(self.conn, parsed_conditions(conditions), show_query)
        print('Showing recommendations as per the following conditions:\nCondition = ', conditions)
        df = pd.DataFrame(list(data), columns=['Dataset', 'Table', 'Number_of_columns',
                                                 'Number_of_rows', 'Path_to_table']).sort_values('Number_of_rows',
                                                                                                 ignore_index=True,
                                                                                               ascending=False)
        df['Number_of_rows'] = df['Number_of_rows'].apply(lambda x: int(x))
        df['Number_of_columns'] = df['Number_of_columns'].apply(lambda x: int(x))
        return df

    def get_path_between_tables(self, source_table: pd.Series, target_table: pd.Series, hops: int,
                                relation: str = 'data:hasContentSimilarity', show_query: bool = False):
        return get_path_between_tables(self.conn, source_table, target_table, hops, relation, show_query)

    def query(self, rdf_query: str):
        return query_kglids(self.conn, rdf_query)

    def get_top_scoring_ml_model(self, dataset: str = '', show_query=False):
        return get_top_scoring_ml_model(self.conn, dataset, show_query)

    def get_pipelines_info(self, author: str = '', show_query=False):
        return get_pipelines_info(self.conn, author, show_query).sort_values('Number_of_votes', ignore_index=True,
                                                                             ascending=False)

    def get_most_recent_pipeline(self, dataset: str = '', show_query=False):
        return get_most_recent_pipeline(self.conn, dataset, show_query)

    def get_top_k_scoring_pipelines_for_dataset(self, dataset: str = '', k: int = None, show_query=False):
        return get_top_k_scoring_pipelines_for_dataset(self.conn, dataset, k, show_query)

    def get_most_popular_parameters(self, library: str, parameters='all'):
        pass

    def search_classifier(self, dataset: str = '', show_query=False):
        return search_classifier(self.conn, dataset, show_query)

    def get_hyperparameters(self, classifier: pd.Series, show_query=False):
        pipeline_name = classifier['Pipeline']
        classifier = classifier['Classifier']
        return get_hyperparameters(self.conn, pipeline_name, classifier, show_query)

    def get_top_k_library_used(self, dataset: str = '', k: int = 5, show_query=False):
        return get_library_usage(self.conn, dataset, k, show_query)

    def get_top_used_libraries(self, k: int = 5, task: str = 'classification', show_query: bool = False):
        supported_tasks = ['classification', 'regression', 'visualization', 'clustering']
        if task not in supported_tasks:
            raise ValueError(' invalid task, try using one of the following tasks: \n'
                             'classification, regression, visualization or clustering!')
        else:
            library_info = get_top_used_libraries(self.conn, task, show_query)
            if len(library_info) == 0:
                print('No library found for {}'.format(task))
                return
            library_info['Module'] = library_info['Module'].apply(lambda module: module.replace('/', '.'))
            # fetch top k libraries by maximum occurrence
            libraries = library_info['Library']
            library_count = Counter(libraries)
            if k > len(library_count):
                k = len(library_count)
                if k == 1:
                    print('Single library was found for {}: '.format(task))
                else:
                    print('Maximum {} libraries were found for {}: '.format(k, task))
            else:
                if k == 1:
                    print('Showing the top used library for {}: '.format(task))
                else:
                    print('Showing the top {} libraries for {}: '.format(k, task))
            libraries = sorted(library_count, key=library_count.get, reverse=True)[:k]

            for i in libraries[:len(libraries)-1]:
                print(i + ',', end=' ')
            print(libraries[-1])
            for k, v in library_info.to_dict('index').items():
                if v['Library'] not in libraries:
                    library_info = library_info.drop(k)
            return library_info.sort_values(by=['Library']).reset_index(drop=True)

    def get_pipelines_calling_libraries(self, components: list, show_query: bool = False):
        return get_pipelines_calling_libraries(self.conn, components, show_query)

    def get_pipelines_for_deep_learning(self, show_query: bool = False):
        return get_pipelines_for_deep_learning(self.conn, show_query)

    def recommend_transformations(self, show_query: bool = False):
        return recommend_transformations(self.conn, show_query)

    def get_pipelines_by_tags(self, tag: str = '', show_query: bool = False):
        return get_pipelines_by_tags(self.conn, tag, show_query)

    def show_pipeline_usage_by_task(self, show_query: bool = False):
        usage = dict()
        usage['classification'] = sum(get_pipelines_by_tags(self.conn, tag='classification', show_query=show_query)['Number_of_pipelines'].tolist())
        usage['clustering'] = sum(get_pipelines_by_tags(self.conn, tag='clustering', show_query=show_query)['Number_of_pipelines'].tolist())
        usage['visualization'] = sum(get_pipelines_by_tags(self.conn, tag='visualization', show_query=show_query)['Number_of_pipelines'].tolist())
        usage['cleaning'] = sum(get_pipelines_by_tags(self.conn, tag='cleaning', show_query=show_query)['Number_of_pipelines'].tolist())
        usage['regression'] = sum(get_pipelines_by_tags(self.conn, tag='regression', show_query=show_query)['Number_of_pipelines'].tolist())
        usage['deep learning'] = sum(get_pipelines_by_tags(self.conn, tag='deep learning', show_query=show_query)['Number_of_pipelines'].tolist()) + \
                        sum(get_pipelines_by_tags(self.conn, tag='neural networks', show_query=show_query)['Number_of_pipelines'].tolist())

        tasks = list(usage.keys())
        data = list(usage.values())
        colors = ("red", "orange", "yellow", "limegreen", "seagreen", "dodgerblue")

        def func(pct):
            return "{:.1f}%".format(pct)

        wp = {'linewidth': 0, 'edgecolor': "black"}
        label_data = [str(i) + " pipelines" for i in data]
        fig, ax = plt.subplots(figsize=(10, 8))
        wedges, texts, autotexts = ax.pie(data,
                                          autopct=lambda pct: func(pct),
                                          colors=colors,
                                          labels=label_data,
                                          textprops=dict(color="black"),
                                          wedgeprops=wp)
        ax.legend(wedges, tasks,
                  loc="center left",
                  bbox_to_anchor=(1.2, 0, 0.5, 1), fontsize=15)

        plt.setp(autotexts, size=10, weight='bold')
        plt.title("Pipeline usage by tasks", fontsize=15)
        plt.show()

    def show_top_k_models_by_task(self, task: str, k: int = 5, show_query: bool = False):
        if task == 'classification':
            plot_top_k_classifiers(self.conn, k, show_query)

        elif task == 'regression':
            plot_top_k_regressors(self.conn, k, show_query)

        else:
            raise ValueError(' invalid task, try using one of the following tasks: \n'
                                 'classification or regression')

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
        create_triplets(table,name)
        triplet_encoding(name, 'Table')
        cleaning_op = graphSaint_cleaning(table, name)
        return cleaning_dict[cleaning_op]

    def apply_cleaning_operations(self, operation: str, df: pd.DataFrame):
        clean_df = apply_cleaning(df,operation)
        return clean_df

    def recommend_transformation_operations(self, table: pd.DataFrame, name: str = 'Transformation_dataset'):
        scaling_dict = {0: 'MinMaxScaler', 1: 'RobustScaler', 2: 'StandardScaler'}
        unary_dict = {1: 'Log', 2: 'Sqrt', 0: 'NoUnary'}
        create_triplets(table, name)
        triplet_encoding(name,'Table')
        scaling_op = graphSaint_scaling(name, table)
        triplet_encoding(name, 'Column')
        unary_op = graphSaint_unary(name, table)
        data = {'Recommended_transformation': [scaling_op],
                'Feature': [None]}
        recommended_scaling_transformations = pd.DataFrame(data)
        recommended_scaling_transformations['Recommended_transformation'] = recommended_scaling_transformations[
            'Recommended_transformation'].replace(scaling_dict)

        df_unary_rec = pd.DataFrame(unary_op, columns=['Recommended_transformation'])
        df_unary_col = pd.read_csv('OnDemandDataPrep/storage/' + name + '_gnn_Column/mapping/Column_entidx2name.csv')
        df_unary_join = df_unary_col.join(df_unary_rec)
        recommended_unary_transformations = df_unary_join.groupby('Recommended_transformation')['ent name'].apply(
            list).reset_index()
        recommended_unary_transformations.rename(columns={'ent name': 'Feature'}, inplace=True)
        recommended_unary_transformations['Recommended_transformation'] = recommended_unary_transformations[
            'Recommended_transformation'].replace(unary_dict)
        recommended_transformations = pd.concat(
            [recommended_scaling_transformations, recommended_unary_transformations], ignore_index=True)
        return recommended_transformations

    def apply_transformation_operations(self, X: pd.DataFrame, recommended_transformations: pd.DataFrame, label_name: str = 'None'):
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
                                                                                 kw_args={'const': min_neg_val}, validate=True)

                                X[str(f)] = unary_transformation_model.fit_transform(X=np.array(X[str(f)]).reshape(-1, 1))

                    elif transformation == 'Sqrt':
                        def sqrt_plus_const(x, const=0):
                            return np.sqrt(x + np.abs(const) + 0.0001)

                        for f in tqdm(feature):
                            if str(f) != label_name and pd.api.types.is_numeric_dtype(X[str(f)].dtype):
                                min_neg_val = X[str(f)].min()
                                unary_transformation_model = FunctionTransformer(func=sqrt_plus_const,
                                                                                 kw_args={'const': min_neg_val}, validate=True)
                                X[str(f)] = unary_transformation_model.fit_transform(X=np.array(X[str(f)]).reshape(-1, 1))
                    else:
                        unary_transformation_model = FunctionTransformer(func=np.square, validate=True)
                        X[feature] = unary_transformation_model.fit_transform(X=X[feature])
                else:
                    raise ValueError(f'{transformation} not supported')

        return X
