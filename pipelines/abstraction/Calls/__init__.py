class Call:
    name = ''  # name of the function/class
    library_path = ''  # the library import path. e.g. for sklearn.svm.SVC it will be sklearn.svm
    parameters = {}  # contains the names and default values for the first 5 params
    is_class_def = None  # whether this Call is a class (or function)
    return_types = []  # the return types of this call. For classes, the same object is returned
    is_relevant = True  # whether this call is relevant to the analysis (e.g. plotting functions aren't)

    def __init__(self, name, library_path, parameters, is_class_def, return_types=None, is_relevant=True):
        self.name = name
        self.library_path = library_path
        self.parameters = parameters
        self.is_class_def = is_class_def
        if self.is_class_def:
            self.return_types = [self]
        else:
            self.return_types = return_types


class File:
    __slots__ = ['id', 'filename', 'path']

    def __init__(self, id, filename, path):
        self.id = id
        self.filename = filename
        self.path = path


pd_dataframe = Call('DataFrame',
                    'pandas',
                    {'data': None,
                     'index': None,
                     'columns': None,
                     'dtype': None,
                     'copy': None},
                    True)
read_csv_call = Call('read_csv',
                     'pandas',
                     {'filepath_or_buffer': None,
                      'sep': 'NoDefault.no_default',
                      'delimiter': None,
                      'header': 'infer'},
                     False,
                     [pd_dataframe])  # pd_dataframe is the Call object for pandas.DataFrame
# dataframe_info_call = Call('info',
#                            'pandas.DataFrame',)
# dataframe_head = Call('head',
#                       'pandas.DataFrame',
#                       )
dataframe_drop = Call('drop',
                      'pandas.DataFrame',
                      {'labels': None,
                       'axis': 0,
                       'index': None,
                       'columns': None,
                       'level': None,
                       'inplace': False,
                       'errors': 'raise'},
                      False,
                      [pd_dataframe])
dataframe_sort_values = Call('sort_values',
                             'pandas.DataFrame',
                             {'by': None,
                              'axis': 0,
                              'inplace': False,
                              'kind': 'quicksort',
                              'na_position': 'last',
                              'ignore_index': False,
                              'key': None},
                             False,
                             [pd_dataframe])
dataframe_nunique = Call('nunique',
                         'pandas.DataFrame',
                         {'axis': 0,
                          'dropna': True},
                         False,
                         [pd_dataframe])
dataframe_value_counts = Call('value_counts',
                              'pandas.DataFrame',
                              {'subset': None,
                               'normalize': False,
                               'sort': True,
                               'ascending': False,
                               'dropna': True},
                              False,
                              [pd_dataframe])

dataframe_group_by = Call('groupby',
                          'pandas.DataFrame',
                          {'by': None,
                           'axis': 0,
                           'level': None,
                           'as_index': True,
                           'sort': True,
                           'group_keys': True,
                           'squeeze': None,
                           'observed': False,
                           'dropna': True},
                          False,
                          [pd_dataframe])
pandas_get_dummies = Call('get_dummies',
                          'pandas',
                          {'data': None,
                           'prefix': None,
                           'prefix_sep': '_',
                           'dummy_na': False,
                           'columns': None,
                           'sparse': False,
                           'drop_first': False,
                           'dtype': None},
                          False,
                          [pd_dataframe])
# Start
dataframe_rename = Call('rename',
                        'pandas.DataFrame',
                        {'mapper': None,
                         'index': None,
                         'columns': None,
                         'axis': None,
                         'copy': True,
                         'inplace': False,
                         'level': None,
                         'errors': 'ignore'},
                        False,
                        [pd_dataframe])
dataframe_replace = Call('replace',
                         'pandas.DataFrame',
                         {'to_replace': None,
                          'value': None,
                          'inplace': False,
                          'limit': None,
                          'regex': False,
                          'method': 'pad'},
                         False,
                         [pd_dataframe])
dataframe_reset_index = Call('reset_index',
                             'pandas.DataFrame',
                             {'level': None,
                              'drop': False,
                              'inplace': False,
                              'col_level': 0,
                              'col_fill': ''},
                             False,
                             [pd_dataframe])
dataframe_sample = Call('sample',
                        'pandas.DataFrame',
                        {'n': None,
                         'frac': None,
                         'replace': False,
                         'weights': None,
                         'random_state': None,
                         'axis': None,
                         'ignore_index': False},
                        False,
                        [pd_dataframe])
dataframe_sort_index = Call('sort_index',
                            'pandas.DataFrame',
                            {'axis': 0,
                             'level': None,
                             'ascending': True,
                             'inplace': False,
                             'kind': 'quicksort',
                             'na_position': 'last',
                             'sort_remaining': True,
                             'ignore_index': False,
                             'key': None},
                            False,
                            [pd_dataframe])
dataframe_transpose = Call('transpose',
                           'pandas.DataFrame',
                           {'args': (),  # TODO: This is a tuple, should appear like so
                            'copy': False},
                           False,
                           [pd_dataframe])
dataframe_t = Call('T',
                   'pandas.DataFrame',
                   dataframe_transpose.parameters,
                   False,
                   dataframe_transpose.return_types)
dataframe_drop_duplicates = Call('drop_duplicates',
                                 'pandas.DataFrame',
                                 {'subset': None,
                                  'keep': 'first',
                                  'inplace': False,
                                  'ignore_index': False},
                                 False,
                                 [pd_dataframe])
dataframe_dropna = Call('dropna',
                        'pandas.DataFrame',
                        {'axis': 0,
                         'how': 'any',
                         'thresh': None,
                         'subset': None,
                         'inplace': False},
                        False,
                        [pd_dataframe])
dataframe_fillna = Call('fillna',
                        'pandas.DataFrame',
                        {'value': None,
                         'method': None,
                         'axis': None,
                         'inplace': False,
                         'limit': None,
                         'downcast': None},
                        False,
                        [pd_dataframe])
dataframe_from_dict = Call('from_dict',
                           'pandas.DataFrame',
                           {'data': None,
                            'orient': 'columns',
                            'dtype': None,
                            'columns': None},
                           False,
                           [pd_dataframe])
dataframe_copy = Call('copy',
                      'pandas.DataFrame',
                      {'deep': True},
                      False,
                      [pd_dataframe])
dataframe_apply = Call('apply',
                       'pandas.DataFrame',
                       {'func': None,
                        'axis': 0,
                        'raw': False,
                        'result_type': None,
                        'args': (),
                        'kwargs': {}  # TODO: This is a dictionary, it should reflect it
                        },
                       False,
                       [pd_dataframe])
dataframe_astype = Call('astype',
                        'pandas.DataFrame',
                        {'dtype': None,
                         'copy': True,
                         'errors': 'raise'},
                        False,
                        [])  # TODO: What is this return type ?? CASTED
dataframe_isnull = Call('isnull',
                        'pandas.DataFrame',
                        {},
                        False,
                        [pd_dataframe])
dataframe_isna = Call('isna',
                      'pandas.DataFrame',
                      {},
                      False,
                      [pd_dataframe])

dataframe_to_csv = Call('to_csv',
                        'pandas.DataFrame',
                        {'path_or_buf': None,
                         'sep': ',',
                         'na_rep': '',
                         'float_format': None,
                         'columns': None,
                         'header': True,
                         'index': True,
                         'index_label': None,
                         'mode': 'w',
                         'encoding': None,
                         'compression': 'infer',
                         'quoting': None,
                         'quotechar': '"',
                         'line_terminator': None,
                         'chunksize': None,
                         'date_format': None,
                         'doublequote': True,
                         'escapechar': None,
                         'decimal': '.',
                         'errors': 'strict',
                         'storage_options': None},
                        False,
                        ['str'])

pandas_merge = Call('merge',
                    'pandas',
                    {'left': None,
                     'right': None,
                     'how': 'inner',
                     'on': None,
                     'left_on': None,
                     'right_on': None,
                     'left_index': False,
                     'right_index': False,
                     'sort': False,
                     'suffixes': ('_x', '_y'),
                     'copy': True,
                     'indicator': False,
                     'validate': None},
                    False,
                    [pd_dataframe])
pandas_concat = Call('concat',
                     'pandas',
                     {'objs': None,
                      'axis': 0,
                      'join': 'outer',
                      'ignore_index': False,
                      'keys': None,
                      'levels': None,
                      'names': None,
                      'verify_integrity': False,
                      'sort': False,
                      'copy': True},
                     False,
                     [pd_dataframe])
dataframe_sum = Call('sum',
                     'pandas.DataFrame',
                     {'axis': None,
                      'skipna': True,
                      'level': None,
                      'numeric_only': None,
                      'min_count': 0},
                     False,
                     [pd_dataframe])

dataframe_min = Call('min',
                     'pandas.DataFrame',
                     {'axis': None,
                      'skipna': None,
                      'level': None,
                      'numeric_only': None},
                     False,
                     [pd_dataframe])
dataframe_median = Call('median',
                     'pandas.DataFrame',
                     {'axis': None,
                      'skipna': None,
                      'level': None,
                      'numeric_only': None},
                     False,
                     [pd_dataframe])
dataframe_mean = Call('mean',
                     'pandas.DataFrame',
                     {'axis': None,
                      'skipna': None,
                      'level': None,
                      'numeric_only': None},
                     False,
                     [pd_dataframe])
dataframe_max = Call('max',
                     'pandas.DataFrame',
                     {'axis': None,
                      'skipna': None,
                      'level': None,
                      'numeric_only': None},
                     False,
                     [pd_dataframe])
dataframe_std = Call('min',
                     'pandas.DataFrame',
                     {'axis': None,
                      'skipna': None,
                      'level': None,
                      'ddof': 1,
                      'numeric_only': None},
                     False,
                     [pd_dataframe])

dataframe_transform = Call('transform',
                           'pandas.DataFrame',
                           {'func': None,
                            'axis': 0},
                           False,
                           [pd_dataframe])

# preprocessing = Call('preprocessing',
#                      'sklearn', )
label_encoder = Call('LabelEncoder',
                     'sklearn.preprocessing',
                     {'classes_': None},
                     True)
label_encoder_fit = Call('fit',
                         'sklearn.preprocessing.LabelEncoder',
                         {'y': None},
                         False,
                         [label_encoder])

label_encoder_fit_transform = Call('fit_transform',
                                   'sklearn.preprocessing.LabelEncoder',
                                   {'y': None},
                                   False,
                                   [pd_dataframe])

label_encoder_transform = Call('transform',
                               'sklearn.preprocessing.LabelEncoder',
                               {'y': None},
                               False,
                               [pd_dataframe])

train_test_split_call = Call('train_test_split',
                             'sklearn.model_selection',
                             {'*arrays': None,
                              'test_size': None,
                              'train_size': None,
                              'random_state': None,
                              'shuffle': True,
                              'stratify': None},
                             False,
                             [pd_dataframe, pd_dataframe, pd_dataframe, pd_dataframe])

accuracy_score = Call('accuracy_score',
                      'sklearn.metrics',
                      {'y_true': None,
                       'y_pred': None,
                       'normalize': True,
                       'sample_weight': None},
                      False,
                      'float')  # TODO Determine the return type of not classes

cross_val_score = Call('cross_val_score',
                       'sklearn.mode_selection',
                       {'estimator': None,
                        'X': None,
                        'y': None,
                        'groups': None,
                        'scoring': None,
                        'cv': None,
                        'n_jobs': None,
                        'verbose': 0,
                        'fit_params': None,
                        'pre_dispatch': '2*n_jobs',
                        'error_score': 'nan'},
                       False,
                       [pd_dataframe])

random_forest_classifier = Call('RandomForestClassifier',
                                'sklearn.ensemble',
                                {'n_estimators': 100},
                                True)

random_forest_classifier_fit = Call('fit',
                                    'sklearn.ensemble.RandomForestClassifier',
                                    {'X': None,
                                     'y': None,
                                     'sample_weight': None},
                                    False,
                                    [random_forest_classifier])

random_forest_classifier_predict = Call('predict',
                                        'sklearn.ensemble.RandomForestClassifier',
                                        {'X': None},
                                        False,
                                        [pd_dataframe])

gradient_boosting_classifier = Call('GradientBoostingClassifier',
                                    'sklearn.ensemble',
                                    {},
                                    True)
gradient_boosting_classifier_fit = Call('fit',
                                        'sklearn.ensemble.GradientBoostingClassifier',
                                        {'X': None,
                                         'y': None,
                                         'sample_weight': None,
                                         'monitor': None},
                                        False,
                                        [gradient_boosting_classifier])
gradient_boosting_classifier_predict = Call('predict',
                                            'sklearn.ensemble.GradientBoostingClassifier',
                                            {'X': None},
                                            False,
                                            [pd_dataframe])

logistic_regression = Call('LogisticRegression',
                           'sklearn.linear_model',
                           {'penalty': 'l2'},
                           True)
logistic_regression_fit = Call('fit',
                               'sklearn.linear_model.LogisticRegression',
                               {'X': None,
                                'y': None,
                                'sample_weight': None},
                               False,
                               [logistic_regression])
logistic_regression_predict = Call('predict',
                                   'sklearn.linear_model.LogisticRegression',
                                   {'X': None},
                                   False,
                                   [pd_dataframe])

sgd_classifier = Call('SGDClassifier',
                      'sklearn.linear_model',
                      {'loss': 'hinge'},
                      True)
sgd_classifier_fit = Call('fit',
                          'sklearn.linear_model.SGDClassifier',
                          {'X': None,
                           'y': None,
                           'coef_init': None,
                           'intercept_init': None,
                           'sample_weight': None},
                          False,
                          [sgd_classifier])
sgd_classifier_predict = Call('predict',
                              'sklearn.linear_model.SGDClassifier',
                              {'X': None},
                              False,
                              [pd_dataframe])

svc = Call('SVC',
           'sklearn.svm',
           {},
           True)
svc_fit = Call('fit',
               'sklearn.svm.SVC',
               {'X': None,
                'y': None,
                'sample_weight': None},
               False,
               [svc])
svc_predict = Call('predict',
                   'sklearn.svm.SVC',
                   {'X': None},
                   False,
                   [pd_dataframe])

# ada_boost_classifier =
# ada_boost_regressor =
# bagging_classifier =
# bagging_regressor =
# extra_trees_classifier =
# extra_trees_regressor =
# random_trees_embedding =
# stacking_classifier =
# stacking_regressor =
# voting_classifier =
# voting_regressor =
# hist_gradient_boosting_classifier =
# hist_gradient_boosting_regressor =
# gaussian_process_classifier =
# gaussian_process_regressor =
# logistic_regression_CV =
# passive_aggressive_classifier =
# ridge_classifier =
# ridge_classifier_CV =
# linear_regression =
# ridge =
# ridge_CV =
# SGDRegressor =
# elastic_net =
# elastic_net_CV =
# lasso =
# lasso_CV =
# ARDRegression =
# passive_aggressive_regressor =
# bernoulliN =
# categorical_NB =
# gaussian_NB =
# KNeighbors_classifier =
# KNeighbors_regressor =
# MLP_classifier =
# MLP_regressor =
# LinearSVC =
# LinearSVR =
# SVR =
# DecisionTreeClassifier =
# DecisionTreeRegressor =
# ColumnTransformer =
# FastICA =
# IncrementalPCA =
# KernelPCA =
# PCA =
# TruncatedSVD =
# OneHotEncoder =
# CountVectorizer =
# HashingVectorizer =
# TfidfVectorizer =
# SimpleImputer =
# IterativeImputer =
# KNNImputer =


Elements = [svc, svc_fit, pd_dataframe, pandas_get_dummies, read_csv_call, dataframe_drop,
            dataframe_sort_values, dataframe_group_by, label_encoder, label_encoder_fit,
            train_test_split_call, accuracy_score, cross_val_score, random_forest_classifier,
            random_forest_classifier_fit, gradient_boosting_classifier, gradient_boosting_classifier_fit,
            logistic_regression, logistic_regression_fit, sgd_classifier, dataframe_to_csv,
            sgd_classifier_fit, dataframe_nunique, dataframe_value_counts,
            dataframe_rename, dataframe_replace, dataframe_reset_index, dataframe_sample, dataframe_sort_index,
            dataframe_transpose, dataframe_t, dataframe_drop_duplicates, dataframe_dropna, dataframe_fillna,
            dataframe_from_dict, dataframe_apply, dataframe_copy, dataframe_astype, pandas_merge, pandas_concat,
            label_encoder_fit_transform, label_encoder_transform, random_forest_classifier_predict,
            gradient_boosting_classifier_predict, logistic_regression_predict, sgd_classifier_predict,
            svc_predict, dataframe_sum, dataframe_isnull, dataframe_isna, dataframe_mean, dataframe_median,
            dataframe_min, dataframe_max, dataframe_std, dataframe_transform]
