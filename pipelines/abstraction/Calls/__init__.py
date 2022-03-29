class Call:
    name = ''  # name of the function/class
    library_path = ''  # the library import path. e.g. for sklearn.svm.SVC it will be sklearn.svm
    parameters = {}  # contains the names and default values for the first 5 params
    is_class_def = None  # whether this Call is a class (or function)
    return_types = []  # the return types of this call. For classes, the same object is returned
    is_relevant = True  # whether this call is relevant to the analysis (e.g. plotting functions aren't)

    def __init__(self, name='', library_path='', parameters=None, is_class_def=False, return_types=None, is_relevant=True):
        if parameters is None:
            parameters = {}
        self.name = name
        self.library_path = library_path
        self.parameters = parameters
        self.is_class_def = is_class_def
        if self.is_class_def:
            self.return_types = [self]
        else:
            self.return_types = return_types
        self.is_relevant = is_relevant

    def full_path(self):
        return f'{self.library_path}.{self.name}'


class File:
    __slots__ = ['filename']

    def __init__(self, filename):
        self.filename = filename


packages = {}

pd_dataframe = Call('DataFrame',
                    'pandas',
                    {'data': None,
                     'index': None,
                     'columns': None,
                     'dtype': None,
                     'copy': None},
                    True)
packages[f'{pd_dataframe.library_path}.{pd_dataframe.name}'] = pd_dataframe

read_csv_call = Call('read_csv',
                     'pandas',
                     {'filepath_or_buffer': None,
                      'sep': ',',
                      'delimiter': None,
                      'header': 'infer'},
                     False,
                     [pd_dataframe])
packages[f'{read_csv_call.library_path}.{read_csv_call.name}'] = read_csv_call

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
packages[f'{dataframe_drop.library_path}.{dataframe_drop.name}'] = dataframe_drop

dataframe_sort_values = Call('sort_values',
                             'pandas.DataFrame',
                             {'by': None,
                              'axis': 0,
                              'ascending': True,
                              'inplace': False,
                              'kind': 'quicksort',
                              'na_position': 'last',
                              'ignore_index': False,
                              'key': None},
                             False,
                             [pd_dataframe])
packages[f'{dataframe_sort_values.library_path}.{dataframe_sort_values.name}'] = dataframe_sort_values

dataframe_nunique = Call('nunique',
                         'pandas.DataFrame',
                         {'axis': 0,
                          'dropna': True},
                         False,
                         [pd_dataframe])
packages[f'{dataframe_nunique.library_path}.{dataframe_nunique.name}'] = dataframe_nunique

dataframe_value_counts = Call('value_counts',
                              'pandas.DataFrame',
                              {'subset': None,
                               'normalize': False,
                               'sort': True,
                               'ascending': False,
                               'dropna': True},
                              False,
                              [pd_dataframe])
packages[f'{dataframe_value_counts.library_path}.{dataframe_value_counts.name}'] = dataframe_value_counts

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
packages[f'{dataframe_group_by.library_path}.{dataframe_group_by.name}'] = dataframe_group_by

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
packages[f'{pandas_get_dummies.library_path}.{pandas_get_dummies.name}'] = pandas_get_dummies

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
packages[f'{dataframe_rename.library_path}.{dataframe_rename.name}'] = dataframe_rename

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
packages[f'{dataframe_replace.library_path}.{dataframe_replace.name}'] = dataframe_replace

dataframe_reset_index = Call('reset_index',
                             'pandas.DataFrame',
                             {'level': None,
                              'drop': False,
                              'inplace': False,
                              'col_level': 0,
                              'col_fill': ''},
                             False,
                             [pd_dataframe])
packages[f'{dataframe_reset_index.library_path}.{dataframe_reset_index.name}'] = dataframe_reset_index

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
packages[f'{dataframe_sample.library_path}.{dataframe_sample.name}'] = dataframe_sample

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
packages[f'{dataframe_sort_index.library_path}.{dataframe_sort_index.name}'] = dataframe_sort_index

dataframe_transpose = Call('transpose',
                           'pandas.DataFrame',
                           {'args': (),  # TODO: This is a tuple, should appear like so
                            'copy': False},
                           False,
                           [pd_dataframe])
packages[f'{dataframe_transpose.library_path}.{dataframe_transpose.name}'] = dataframe_transpose

dataframe_t = Call('T',
                   'pandas.DataFrame',
                   dataframe_transpose.parameters,
                   False,
                   dataframe_transpose.return_types)
packages[f'{dataframe_t.library_path}.{dataframe_t.name}'] = dataframe_t

dataframe_drop_duplicates = Call('drop_duplicates',
                                 'pandas.DataFrame',
                                 {'subset': None,
                                  'keep': 'first',
                                  'inplace': False,
                                  'ignore_index': False},
                                 False,
                                 [pd_dataframe])
packages[f'{dataframe_drop_duplicates.library_path}.{dataframe_drop_duplicates.name}'] = dataframe_drop_duplicates

dataframe_dropna = Call('dropna',
                        'pandas.DataFrame',
                        {'axis': 0,
                         'how': 'any',
                         'thresh': None,
                         'subset': None,
                         'inplace': False},
                        False,
                        [pd_dataframe])
packages[f'{dataframe_dropna.library_path}.{dataframe_dropna.name}'] = dataframe_dropna

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
packages[f'{dataframe_fillna.library_path}.{dataframe_fillna.name}'] = dataframe_fillna

dataframe_from_dict = Call('from_dict',
                           'pandas.DataFrame',
                           {'data': None,
                            'orient': 'columns',
                            'dtype': None,
                            'columns': None},
                           False,
                           [pd_dataframe])
packages[f'{dataframe_from_dict.library_path}.{dataframe_from_dict.name}'] = dataframe_from_dict

dataframe_copy = Call('copy',
                      'pandas.DataFrame',
                      {'deep': True},
                      False,
                      [pd_dataframe])
packages[f'{dataframe_copy.library_path}.{dataframe_copy.name}'] = dataframe_copy

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
packages[f'{dataframe_apply.library_path}.{dataframe_apply.name}'] = dataframe_apply

dataframe_astype = Call('astype',
                        'pandas.DataFrame',
                        {'dtype': None,
                         'copy': True,
                         'errors': 'raise'},
                        False,
                        [])  # TODO: What is this return type ?? CASTED
packages[f'{dataframe_astype.library_path}.{dataframe_astype.name}'] = dataframe_astype

dataframe_isnull = Call('isnull',
                        'pandas.DataFrame',
                        {},
                        False,
                        [pd_dataframe])
packages[f'{dataframe_isnull.library_path}.{dataframe_isnull.name}'] = dataframe_isnull

dataframe_isna = Call('isna',
                      'pandas.DataFrame',
                      {},
                      False,
                      [pd_dataframe])
packages[f'{dataframe_isna.library_path}.{dataframe_isna.name}'] = dataframe_isna

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
packages[f'{dataframe_to_csv.library_path}.{dataframe_to_csv.name}'] = dataframe_to_csv

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
packages[f'{pandas_merge.library_path}.{pandas_merge.name}'] =  pandas_merge

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
packages[f'{pandas_concat.library_path}.{pandas_concat.name}'] = pandas_concat

dataframe_sum = Call('sum',
                     'pandas.DataFrame',
                     {'axis': None,
                      'skipna': True,
                      'level': None,
                      'numeric_only': None,
                      'min_count': 0},
                     False,
                     [pd_dataframe])
packages[f'{dataframe_sum.library_path}.{dataframe_sum.name}'] = dataframe_sum

dataframe_min = Call('min',
                     'pandas.DataFrame',
                     {'axis': None,
                      'skipna': None,
                      'level': None,
                      'numeric_only': None},
                     False,
                     [pd_dataframe])
packages[f'{dataframe_min.library_path}.{dataframe_min.name}'] = dataframe_min

dataframe_median = Call('median',
                        'pandas.DataFrame',
                        {'axis': None,
                         'skipna': None,
                         'level': None,
                         'numeric_only': None},
                        False,
                        [pd_dataframe])
packages[f'{dataframe_median.library_path}.{dataframe_median.name}'] = dataframe_median

dataframe_mean = Call('mean',
                      'pandas.DataFrame',
                      {'axis': None,
                       'skipna': None,
                       'level': None,
                       'numeric_only': None},
                      False,
                      [pd_dataframe])
packages[f'{dataframe_mean.library_path}.{dataframe_mean.name}'] = dataframe_mean

dataframe_max = Call('max',
                     'pandas.DataFrame',
                     {'axis': None,
                      'skipna': None,
                      'level': None,
                      'numeric_only': None},
                     False,
                     [pd_dataframe])
packages[f'{dataframe_max.library_path}.{dataframe_max.name}'] = dataframe_max

dataframe_std = Call('min',
                     'pandas.DataFrame',
                     {'axis': None,
                      'skipna': None,
                      'level': None,
                      'ddof': 1,
                      'numeric_only': None},
                     False,
                     [pd_dataframe])
packages[f'{dataframe_std.library_path}.{dataframe_std.name}'] = dataframe_std

dataframe_transform = Call('transform',
                           'pandas.DataFrame',
                           {'func': None,
                            'axis': 0},
                           False,
                           [pd_dataframe])
packages[f'{dataframe_transform.library_path}.{dataframe_transform.name}'] = dataframe_transform

# preprocessing = Call('preprocessing',
#                      'sklearn', )
label_encoder = Call('LabelEncoder',
                     'sklearn.preprocessing',
                     {'classes_': None},
                     True)
packages[f'{label_encoder.library_path}.{label_encoder.name}'] = label_encoder

label_encoder_fit = Call('fit',
                         'sklearn.preprocessing.LabelEncoder',
                         {'y': None},
                         False,
                         [label_encoder])
packages[f'{label_encoder_fit.library_path}.{label_encoder_fit.name}'] = label_encoder_fit

label_encoder_fit_transform = Call('fit_transform',
                                   'sklearn.preprocessing.LabelEncoder',
                                   {'y': None},
                                   False,
                                   [pd_dataframe])
packages[f'{label_encoder_fit_transform.library_path}.{label_encoder_fit_transform.name}'] = label_encoder_fit_transform

label_encoder_transform = Call('transform',
                               'sklearn.preprocessing.LabelEncoder',
                               {'y': None},
                               False,
                               [pd_dataframe])
packages[f'{label_encoder_transform.library_path}.{label_encoder_transform.name}'] = label_encoder_transform

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
packages[f'{train_test_split_call.library_path}.{train_test_split_call.name}'] = train_test_split_call

accuracy_score = Call('accuracy_score',
                      'sklearn.metrics',
                      {'y_true': None,
                       'y_pred': None,
                       'normalize': True,
                       'sample_weight': None},
                      False,
                      [])  # TODO Determine the return type of not classes
packages[f'{accuracy_score.library_path}.{accuracy_score.name}'] = accuracy_score

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
packages[f'{cross_val_score.library_path}.{cross_val_score.name}'] = cross_val_score

random_forest_classifier = Call('RandomForestClassifier',
                                'sklearn.ensemble',
                                {'n_estimators': 100},
                                True)
packages[f'{random_forest_classifier.library_path}.{random_forest_classifier.name}'] = random_forest_classifier

random_forest_classifier_fit = Call('fit',
                                    'sklearn.ensemble.RandomForestClassifier',
                                    {'X': None,
                                     'y': None,
                                     'sample_weight': None},
                                    False,
                                    [random_forest_classifier])
packages[f'{random_forest_classifier_fit.library_path}.{random_forest_classifier_fit.name}'] = random_forest_classifier_fit

random_forest_classifier_predict = Call('predict',
                                        'sklearn.ensemble.RandomForestClassifier',
                                        {'X': None},
                                        False,
                                        [pd_dataframe])
packages[f'{random_forest_classifier_predict.library_path}.{random_forest_classifier_predict.name}'] = random_forest_classifier_predict

gradient_boosting_classifier = Call('GradientBoostingClassifier',
                                    'sklearn.ensemble',
                                    {},
                                    True)
packages[f'{gradient_boosting_classifier.library_path}.{gradient_boosting_classifier.name}'] = gradient_boosting_classifier

gradient_boosting_classifier_fit = Call('fit',
                                        'sklearn.ensemble.GradientBoostingClassifier',
                                        {'X': None,
                                         'y': None,
                                         'sample_weight': None,
                                         'monitor': None},
                                        False,
                                        [gradient_boosting_classifier])
packages[f'{gradient_boosting_classifier_fit.library_path}.{gradient_boosting_classifier_fit.name}'] = gradient_boosting_classifier_fit

gradient_boosting_classifier_predict = Call('predict',
                                            'sklearn.ensemble.GradientBoostingClassifier',
                                            {'X': None},
                                            False,
                                            [pd_dataframe])
packages[f'{gradient_boosting_classifier_predict.library_path}.{gradient_boosting_classifier_predict.name}'] = gradient_boosting_classifier_predict

logistic_regression = Call('LogisticRegression',
                           'sklearn.linear_model',
                           {'penalty': 'l2'},
                           True)
packages[f'{logistic_regression.library_path}.{logistic_regression.name}'] = logistic_regression

logistic_regression_fit = Call('fit',
                               'sklearn.linear_model.LogisticRegression',
                               {'X': None,
                                'y': None,
                                'sample_weight': None},
                               False,
                               [logistic_regression])
packages[f'{logistic_regression_fit.library_path}.{logistic_regression_fit.name}'] = logistic_regression_fit

logistic_regression_predict = Call('predict',
                                   'sklearn.linear_model.LogisticRegression',
                                   {'X': None},
                                   False,
                                   [pd_dataframe])
packages[f'{logistic_regression_predict.library_path}.{logistic_regression_predict.name}'] = logistic_regression_predict

sgd_classifier = Call('SGDClassifier',
                      'sklearn.linear_model',
                      {'loss': 'hinge'},
                      True)
packages[f'{sgd_classifier.library_path}.{sgd_classifier.name}'] = sgd_classifier

sgd_classifier_fit = Call('fit',
                          'sklearn.linear_model.SGDClassifier',
                          {'X': None,
                           'y': None,
                           'coef_init': None,
                           'intercept_init': None,
                           'sample_weight': None},
                          False,
                          [sgd_classifier])
packages[f'{sgd_classifier_fit.library_path}.{sgd_classifier_fit.name}'] = sgd_classifier_fit

sgd_classifier_predict = Call('predict',
                              'sklearn.linear_model.SGDClassifier',
                              {'X': None},
                              False,
                              [pd_dataframe])
packages[f'{sgd_classifier_predict.library_path}.{sgd_classifier_predict.name}'] = sgd_classifier_predict

svc = Call('SVC',
           'sklearn.svm',
           {},
           True)
packages[f'{svc.library_path}.{svc.name}'] = svc

svc_fit = Call('fit',
               'sklearn.svm.SVC',
               {'X': None,
                'y': None,
                'sample_weight': None},
               False,
               [svc])
packages[f'{svc_fit.library_path}.{svc_fit.name}'] = svc_fit

svc_predict = Call('predict',
                   'sklearn.svm.SVC',
                   {'X': None},
                   False,
                   [pd_dataframe])
packages[f'{svc_predict.library_path}.{svc_predict.name}'] = svc_predict

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
