from enum import Enum


class CallType(Enum):
    CLASS = 'http://kglids.org/ontology/Class'
    FUNCTION = 'http://kglids.org/ontology/Function'
    LIBRARY = 'http://kglids.org/ontology/Library'
    PACKAGE = 'http://kglids.org/ontology/Package'
    NONE = None


class Call:
    name = ''  # name of the function/class
    library_path = ''  # the library import path. e.g. for sklearn.svm.SVC it will be sklearn.svm
    parameters = {}  # contains the names and default values for the first 5 params
    is_class_def = None  # whether this Call is a class (or function)
    call_type = None  # ['Class', 'Function', 'Library', 'Package']
    return_types = []  # the return types of this call. For classes, the same object is returned
    is_relevant = True  # whether this call is relevant to the analysis (e.g. plotting functions aren't)
    count = 0

    def __init__(self, name='', library_path='', parameters=None, is_class_def=False, call_type=CallType.NONE,
                 return_types=None, is_relevant=True):
        if parameters is None:
            parameters = {}
        self.name = name
        self.library_path = library_path
        self.parameters = parameters
        self.is_class_def = is_class_def
        self.call_type = call_type
        if self.is_class_def:
            self.return_types = [self]
        else:
            self.return_types = return_types
        self.is_relevant = is_relevant

    def full_path(self):
        self.count += 1
        return f'{self.library_path}.{self.name}'


class File:
    __slots__ = ['filename']

    def __init__(self, filename):
        self.filename = filename


packages = dict()
# Pandas
packages['pandas'] = Call(name='pandas', call_type=CallType.LIBRARY)

# pandas # DataFrame
pd_dataframe = Call('DataFrame',
                    'pandas',
                    {'data': None,
                     'index': None,
                     'columns': None,
                     'dtype': None,
                     'copy': None},
                    True,
                    CallType.CLASS)
packages[f'{pd_dataframe.library_path}.{pd_dataframe.name}'] = pd_dataframe

dataframe_iloc = Call('iloc',
                      'pandas.DataFrame',
                      {},
                      False,
                      CallType.NONE,
                      [pd_dataframe])  # Note: Should not appear in the libraries output
packages[f'{dataframe_iloc.library_path}.{dataframe_iloc.name}'] = dataframe_iloc

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
                      CallType.FUNCTION,
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
                             CallType.FUNCTION,
                             [pd_dataframe])
packages[f'{dataframe_sort_values.library_path}.{dataframe_sort_values.name}'] = dataframe_sort_values

dataframe_nunique = Call('nunique',
                         'pandas.DataFrame',
                         {'axis': 0,
                          'dropna': True},
                         False,
                         CallType.FUNCTION,
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
                              CallType.FUNCTION,
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
                          CallType.FUNCTION,
                          [pd_dataframe])
packages[f'{dataframe_group_by.library_path}.{dataframe_group_by.name}'] = dataframe_group_by

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
                        CallType.FUNCTION,
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
                         CallType.FUNCTION,
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
                             CallType.FUNCTION,
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
                        CallType.FUNCTION,
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
                            CallType.FUNCTION,
                            [pd_dataframe])
packages[f'{dataframe_sort_index.library_path}.{dataframe_sort_index.name}'] = dataframe_sort_index

dataframe_transpose = Call('transpose',
                           'pandas.DataFrame',
                           {'args': tuple(),
                            'copy': False},
                           False,
                           CallType.FUNCTION,
                           [pd_dataframe])
packages[f'{dataframe_transpose.library_path}.{dataframe_transpose.name}'] = dataframe_transpose

dataframe_t = Call('T',
                   'pandas.DataFrame',
                   dataframe_transpose.parameters,
                   False,
                   CallType.FUNCTION,
                   dataframe_transpose.return_types)
packages[f'{dataframe_t.library_path}.{dataframe_t.name}'] = dataframe_t

dataframe_drop_duplicates = Call('drop_duplicates',
                                 'pandas.DataFrame',
                                 {'subset': None,
                                  'keep': 'first',
                                  'inplace': False,
                                  'ignore_index': False},
                                 False,
                                 CallType.FUNCTION,
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
                        CallType.FUNCTION,
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
                        CallType.FUNCTION,
                        [pd_dataframe])
packages[f'{dataframe_fillna.library_path}.{dataframe_fillna.name}'] = dataframe_fillna

# pandas # dataframe # interpolate
dataframe_interpolate = Call('interpolate',
                             'pandas.DataFrame',
                             {'method': 'linear',
                              'axis': 0,
                              'limit': None,
                              'inplace': False,
                              'limit_direction': None,
                              'limit_area': None,
                              'downcast': None},
                             False,
                             CallType.FUNCTION,
                             [pd_dataframe])
packages[f'{dataframe_interpolate.library_path}.{dataframe_interpolate.name}'] = dataframe_interpolate

dataframe_from_dict = Call('from_dict',
                           'pandas.DataFrame',
                           {'data': None,
                            'orient': 'columns',
                            'dtype': None,
                            'columns': None},
                           False,
                           CallType.FUNCTION,
                           [pd_dataframe])
packages[f'{dataframe_from_dict.library_path}.{dataframe_from_dict.name}'] = dataframe_from_dict

dataframe_copy = Call('copy',
                      'pandas.DataFrame',
                      {'deep': True},
                      False,
                      CallType.FUNCTION,
                      [pd_dataframe])
packages[f'{dataframe_copy.library_path}.{dataframe_copy.name}'] = dataframe_copy

dataframe_apply = Call('apply',
                       'pandas.DataFrame',
                       {'func': None,
                        'axis': 0,
                        'raw': False,
                        'result_type': None,
                        'args': (),
                        'kwargs': dict()
                        },
                       False,
                       CallType.FUNCTION,
                       [pd_dataframe])
packages[f'{dataframe_apply.library_path}.{dataframe_apply.name}'] = dataframe_apply

dataframe_astype = Call('astype',
                        'pandas.DataFrame',
                        {'dtype': None,
                         'copy': True,
                         'errors': 'raise'},
                        False,
                        CallType.FUNCTION,
                        [])  # TODO: What is this return type ?? CASTED
packages[f'{dataframe_astype.library_path}.{dataframe_astype.name}'] = dataframe_astype

dataframe_isnull = Call('isnull',
                        'pandas.DataFrame',
                        {},
                        False,
                        CallType.FUNCTION,
                        [pd_dataframe])
packages[f'{dataframe_isnull.library_path}.{dataframe_isnull.name}'] = dataframe_isnull

dataframe_isna = Call('isna',
                      'pandas.DataFrame',
                      {},
                      False,
                      CallType.FUNCTION,
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
                        CallType.FUNCTION,
                        ['str'])
packages[f'{dataframe_to_csv.library_path}.{dataframe_to_csv.name}'] = dataframe_to_csv

dataframe_sum = Call('sum',
                     'pandas.DataFrame',
                     {'axis': None,
                      'skipna': True,
                      'level': None,
                      'numeric_only': None,
                      'min_count': 0},
                     False,
                     CallType.FUNCTION,
                     [pd_dataframe])
packages[f'{dataframe_sum.library_path}.{dataframe_sum.name}'] = dataframe_sum

dataframe_min = Call('min',
                     'pandas.DataFrame',
                     {'axis': None,
                      'skipna': None,
                      'level': None,
                      'numeric_only': None},
                     False,
                     CallType.FUNCTION,
                     [pd_dataframe])
packages[f'{dataframe_min.library_path}.{dataframe_min.name}'] = dataframe_min

dataframe_median = Call('median',
                        'pandas.DataFrame',
                        {'axis': None,
                         'skipna': None,
                         'level': None,
                         'numeric_only': None},
                        False,
                        CallType.FUNCTION,
                        [pd_dataframe])
packages[f'{dataframe_median.library_path}.{dataframe_median.name}'] = dataframe_median

dataframe_mean = Call('mean',
                      'pandas.DataFrame',
                      {'axis': None,
                       'skipna': None,
                       'level': None,
                       'numeric_only': None},
                      False,
                      CallType.FUNCTION,
                      [pd_dataframe])
packages[f'{dataframe_mean.library_path}.{dataframe_mean.name}'] = dataframe_mean

dataframe_max = Call('max',
                     'pandas.DataFrame',
                     {'axis': None,
                      'skipna': None,
                      'level': None,
                      'numeric_only': None},
                     False,
                     CallType.FUNCTION,
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
                     CallType.FUNCTION,
                     [pd_dataframe])
packages[f'{dataframe_std.library_path}.{dataframe_std.name}'] = dataframe_std

dataframe_transform = Call('transform',
                           'pandas.DataFrame',
                           {'func': None,
                            'axis': 0},
                           False,
                           CallType.FUNCTION,
                           [pd_dataframe])
packages[f'{dataframe_transform.library_path}.{dataframe_transform.name}'] = dataframe_transform

# pandas # read_csv
read_csv_call = Call('read_csv',
                     'pandas',
                     {'filepath_or_buffer': None,
                      'sep': ',',
                      'delimiter': None,
                      'header': 'infer'},
                     False,
                     CallType.FUNCTION,
                     [pd_dataframe])
packages[f'{read_csv_call.library_path}.{read_csv_call.name}'] = read_csv_call

# pandas # get_dummies
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
                          CallType.FUNCTION,
                          [pd_dataframe])
packages[f'{pandas_get_dummies.library_path}.{pandas_get_dummies.name}'] = pandas_get_dummies

# pandas # merge
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
                    CallType.FUNCTION,
                    [pd_dataframe])
packages[f'{pandas_merge.library_path}.{pandas_merge.name}'] = pandas_merge

# pandas # concat
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
                     CallType.FUNCTION,
                     [pd_dataframe])
packages[f'{pandas_concat.library_path}.{pandas_concat.name}'] = pandas_concat

# ##### SKLEARN #######
packages['sklearn'] = Call(name='sklearn', call_type=CallType.LIBRARY)

# sklearn # preprocessing
packages['sklearn.preprocessing'] = Call(name='preprocessing', library_path='sklearn', call_type=CallType.PACKAGE)

# sklearn # preprocessing # LabelEncoder
label_encoder = Call('LabelEncoder',
                     'sklearn.preprocessing',
                     {'classes_': None},
                     True,
                     CallType.CLASS)
packages[f'{label_encoder.library_path}.{label_encoder.name}'] = label_encoder

label_encoder_get_params = Call('get_params',
                                'sklearn.preprocessing.LabelEncoder',
                                {'deep': True},
                                False,
                                CallType.FUNCTION,
                                [dataframe_from_dict])
packages[
    f'{label_encoder_get_params.library_path}.{label_encoder_get_params.name}'] = label_encoder_get_params

label_encoder_set_params = Call('set_params',
                                'sklearn.preprocessing.LabelEncoder',
                                {'**params': None},
                                False,
                                CallType.FUNCTION,
                                [label_encoder])
packages[
    f'{label_encoder_set_params.library_path}.{label_encoder_set_params.name}'] = label_encoder_set_params

label_encoder_fit = Call('fit',
                         'sklearn.preprocessing.LabelEncoder',
                         {'y': None},
                         False,
                         CallType.FUNCTION,
                         [label_encoder])
packages[f'{label_encoder_fit.library_path}.{label_encoder_fit.name}'] = label_encoder_fit

label_encoder_fit_transform = Call('fit_transform',
                                   'sklearn.preprocessing.LabelEncoder',
                                   {'y': None},
                                   False,
                                   CallType.FUNCTION,
                                   [pd_dataframe])
packages[f'{label_encoder_fit_transform.library_path}.{label_encoder_fit_transform.name}'] = label_encoder_fit_transform

label_encoder_transform = Call('transform',
                               'sklearn.preprocessing.LabelEncoder',
                               {'y': None},
                               False,
                               CallType.FUNCTION,
                               [pd_dataframe])
packages[f'{label_encoder_transform.library_path}.{label_encoder_transform.name}'] = label_encoder_transform

label_encoder_inverse_transform = Call('inverse_transform',
                                       'sklearn.preprocessing.LabelEncoder',
                                       {'y': None},
                                       False,
                                       CallType.FUNCTION,
                                       [pd_dataframe])
packages[
    f'{label_encoder_inverse_transform.library_path}.{label_encoder_inverse_transform.name}'] = label_encoder_inverse_transform

# sklearn # preprocessing # Binarizer
binarizer = Call('Binarizer',
                 'sklearn.preprocessing',
                 {'threshold': 0.0,
                  'copy': True},
                 True,
                 CallType.CLASS)
packages[f'{binarizer.library_path}.{binarizer.name}'] = binarizer

binarizer_get_params = Call('get_params',
                            'sklearn.preprocessing.Binarizer',
                            {'deep': True},
                            False,
                            CallType.FUNCTION,
                            [dataframe_from_dict])
packages[
    f'{binarizer_get_params.library_path}.{binarizer_get_params.name}'] = binarizer_get_params

binarizer_set_params = Call('set_params',
                            'sklearn.preprocessing.Binarizer',
                            {'**params': None},
                            False,
                            CallType.FUNCTION,
                            [binarizer])
packages[
    f'{binarizer_set_params.library_path}.{binarizer_set_params.name}'] = binarizer_set_params

binarizer_get_feature_names_out = Call('get_feature_names_out',
                                       'sklearn.preprocessing.Binarizer',
                                       {'input_features': None},
                                       False,
                                       CallType.FUNCTION,
                                       [pd_dataframe])
packages[
    f'{binarizer_get_feature_names_out.library_path}.{binarizer_get_feature_names_out.name}'] = binarizer_get_feature_names_out

binarizer_fit_transform = Call('fit_transform',
                               'sklearn.preprocessing.Binarizer',
                               {'X': None,
                                'y': None,
                                '**fit_params': None},
                               False,
                               CallType.FUNCTION,
                               [pd_dataframe])
packages[
    f'{binarizer_fit_transform.library_path}.{binarizer_fit_transform.name}'] = binarizer_fit_transform

binarizer_transform = Call('transform',
                           'sklearn.preprocessing.Binarizer',
                           {'X': None,
                            'copy': None},
                           False,
                           CallType.FUNCTION,
                           [pd_dataframe])
packages[f'{binarizer_transform.library_path}.{binarizer_transform.name}'] = binarizer_transform

binarizer_fit = Call('fit',
                     'sklearn.preprocessing.Binarizer',
                     {'X': None,
                      'y': None, },
                     False,
                     CallType.FUNCTION,
                     [binarizer])
packages[f'{binarizer_fit.library_path}.{binarizer_fit.name}'] = binarizer_fit

# sklearn # preprocessing # FunctionTransformer
function_transformer = Call('FunctionTransformer',
                            'sklearn.preprocessing',
                            {'func': None,
                             'inverse_func': None,
                             'validate': False,
                             'accept_sparse': False,
                             'check_inverse': True,
                             'feature_names_out': None,
                             'kw_args': None,
                             'inv_kw_args': None},
                            True,
                            CallType.CLASS)
packages[f'{function_transformer.library_path}.{function_transformer.name}'] = function_transformer

function_transformer_get_params = Call('get_params',
                                       'sklearn.preprocessing.FunctionTransformer',
                                       {'deep': True},
                                       False,
                                       CallType.FUNCTION,
                                       [dataframe_from_dict])
packages[
    f'{function_transformer_get_params.library_path}.{function_transformer_get_params.name}'] = function_transformer_get_params

function_transformer_set_params = Call('set_params',
                                       'sklearn.preprocessing.FunctionTransformer',
                                       {'**params': None},
                                       False,
                                       CallType.FUNCTION,
                                       [function_transformer])
packages[
    f'{binarizer_set_params.library_path}.{binarizer_set_params.name}'] = binarizer_set_params

function_transformer_get_feature_names_out = Call('get_feature_names_out',
                                                  'sklearn.preprocessing.FunctionTransformer',
                                                  {'input_features': None},
                                                  False,
                                                  CallType.FUNCTION,
                                                  [pd_dataframe])
packages[
    f'{function_transformer_get_feature_names_out.library_path}.{function_transformer_get_feature_names_out.name}'] = function_transformer_get_feature_names_out

function_transformer_fit_transform = Call('fit_transform',
                                          'sklearn.preprocessing.FunctionTransformer',
                                          {'X': None,
                                           'y': None,
                                           '**fit_params': None},
                                          False,
                                          CallType.FUNCTION,
                                          [pd_dataframe])
packages[
    f'{function_transformer_fit_transform.library_path}.{function_transformer_fit_transform.name}'] = function_transformer_fit_transform

function_transformer_transform = Call('transform',
                                      'sklearn.preprocessing.FunctionTransformer',
                                      {'X': None},
                                      False,
                                      CallType.FUNCTION,
                                      [pd_dataframe])
packages[
    f'{function_transformer_transform.library_path}.{function_transformer_transform.name}'] = function_transformer_transform

function_transformer_fit = Call('fit',
                                'sklearn.preprocessing.FunctionTransformer',
                                {'X': None,
                                 'y': None, },
                                False,
                                CallType.FUNCTION,
                                [function_transformer])
packages[f'{function_transformer_fit.library_path}.{function_transformer_fit.name}'] = function_transformer_fit

function_transformer_inverse_transform = Call('inverse_transform',
                                              'sklearn.preprocessing.FunctionTransformer',
                                              {'X': None},
                                              False,
                                              CallType.FUNCTION,
                                              [pd_dataframe])
packages[
    f'{function_transformer_inverse_transform.library_path}.{function_transformer_inverse_transform.name}'] = function_transformer_inverse_transform

# sklearn # preprocessing # KernelCenterer
kernel_centerer = Call('KernelCenterer',
                       'sklearn.preprocessing',
                       {'K_fit_rows_': None,
                        'K_fit_all_': None,
                        'n_features_in_': None,
                        'feature_names_in_': None},
                       True,
                       CallType.CLASS)
packages[f'{kernel_centerer.library_path}.{kernel_centerer.name}'] = kernel_centerer

kernel_centerer_get_params = Call('get_params',
                                  'sklearn.preprocessing.KernelCenterer',
                                  {'deep': True},
                                  False,
                                  CallType.FUNCTION,
                                  [dataframe_from_dict])
packages[
    f'{kernel_centerer_get_params.library_path}.{kernel_centerer_get_params.name}'] = kernel_centerer_get_params

kernel_centerer_set_params = Call('set_params',
                                  'sklearn.preprocessing.KernelCenterer',
                                  {'**params': None},
                                  False,
                                  CallType.FUNCTION,
                                  [kernel_centerer])
packages[
    f'{kernel_centerer_set_params.library_path}.{kernel_centerer_set_params.name}'] = kernel_centerer_set_params

kernel_centerer_get_feature_names_out = Call('get_feature_names_out',
                                             'sklearn.preprocessing.KernelCenterer',
                                             {'input_features': None},
                                             False,
                                             CallType.FUNCTION,
                                             [pd_dataframe])
packages[
    f'{kernel_centerer_get_feature_names_out.library_path}.{kernel_centerer_get_feature_names_out.name}'] = kernel_centerer_get_feature_names_out

kernel_centerer_fit_transform = Call('fit_transform',
                                     'sklearn.preprocessing.KernelCenterer',
                                     {'X': None,
                                      'y': None,
                                      '**fit_params': None},
                                     False,
                                     CallType.FUNCTION,
                                     [pd_dataframe])
packages[
    f'{kernel_centerer_fit_transform.library_path}.{kernel_centerer_fit_transform.name}'] = kernel_centerer_fit_transform

kernel_centerer_transform = Call('transform',
                                 'sklearn.preprocessing.KernelCenterer',
                                 {'K': None,
                                  'copy': True},
                                 False,
                                 CallType.FUNCTION,
                                 [kernel_centerer])
packages[f'{kernel_centerer_transform.library_path}.{kernel_centerer_transform.name}'] = kernel_centerer_transform

kernel_centerer_fit = Call('fit',
                           'sklearn.preprocessing.KernelCenterer',
                           {'K': None,
                            'y': None, },
                           False,
                           CallType.FUNCTION,
                           [kernel_centerer])
packages[f'{kernel_centerer_fit.library_path}.{kernel_centerer_fit.name}'] = kernel_centerer_fit

# sklearn # preprocessing # MultiLabelBinarizer
multi_label_binarizer = Call('MultiLabelBinarizer',
                             'sklearn.preprocessing',
                             {'classes': None,
                              'sparse_output': False},
                             True,
                             CallType.CLASS)
packages[f'{multi_label_binarizer.library_path}.{multi_label_binarizer.name}'] = multi_label_binarizer

multi_label_binarizer_get_params = Call('get_params',
                                        'sklearn.preprocessing.MultiLabelBinarizer',
                                        {'deep': True},
                                        False,
                                        CallType.FUNCTION,
                                        [dataframe_from_dict])
packages[
    f'{multi_label_binarizer_get_params.library_path}.{multi_label_binarizer_get_params.name}'] = multi_label_binarizer_get_params

multi_label_binarizer_set_params = Call('set_params',
                                        'sklearn.preprocessing.MultiLabelBinarizer',
                                        {'**params': None},
                                        False,
                                        CallType.FUNCTION,
                                        [multi_label_binarizer])
packages[
    f'{multi_label_binarizer_set_params.library_path}.{multi_label_binarizer_set_params.name}'] = multi_label_binarizer_set_params

multi_label_binarizer_fit_transform = Call('fit_transform',
                                           'sklearn.preprocessing.MultiLabelBinarizer',
                                           {'y': None},
                                           False,
                                           CallType.FUNCTION,
                                           [pd_dataframe])
packages[
    f'{multi_label_binarizer_fit_transform.library_path}.{multi_label_binarizer_fit_transform.name}'] = multi_label_binarizer_fit_transform

multi_label_binarizer_transform = Call('transform',
                                       'sklearn.preprocessing.MultiLabelBinarizer',
                                       {'y': None},
                                       False,
                                       CallType.FUNCTION,
                                       [pd_dataframe])
packages[
    f'{multi_label_binarizer_transform.library_path}.{multi_label_binarizer_transform.name}'] = multi_label_binarizer_transform

multi_label_binarizer_fit = Call('fit',
                                 'sklearn.preprocessing.MultiLabelBinarizer',
                                 {'y': None},
                                 False,
                                 CallType.FUNCTION,
                                 [multi_label_binarizer])
packages[f'{multi_label_binarizer_fit.library_path}.{multi_label_binarizer_fit.name}'] = multi_label_binarizer_fit

multi_label_binarizer_inverse_transform = Call('inverse_transform',
                                               'sklearn.preprocessing.MultiLabelBinarizer',
                                               {'yt': None},
                                               False,
                                               CallType.FUNCTION,
                                               [pd_dataframe])
packages[
    f'{multi_label_binarizer_inverse_transform.library_path}.{multi_label_binarizer_inverse_transform.name}'] = multi_label_binarizer_inverse_transform

# sklearn # preprocessing # StandardScaler
standard_scaler = Call('StandardScaler',
                       'sklearn.preprocessing',
                       {'copy': True,
                        'with_mean': True,
                        'with_std': True},
                       True,
                       CallType.CLASS)
packages[f'{standard_scaler.library_path}.{standard_scaler.name}'] = standard_scaler

standard_scaler_fit_transform = Call('fit_transform',
                                     'sklearn.preprocessing.StandardScaler',
                                     {'X': None,
                                      'y': None,
                                      '**fit_params': None},
                                     False,
                                     CallType.FUNCTION,
                                     [pd_dataframe])
packages[
    f'{standard_scaler_fit_transform.library_path}.{standard_scaler_fit_transform.name}'] = standard_scaler_fit_transform

standard_scaler_transform = Call('transform',
                                 'sklearn.preprocessing.StandardScaler',
                                 {'X': None,
                                  'copy': None},
                                 False,
                                 CallType.FUNCTION,
                                 [pd_dataframe])
packages[f'{standard_scaler_transform.library_path}.{standard_scaler_transform.name}'] = standard_scaler_transform

standard_scaler_fit = Call('fit',
                           'sklearn.preprocessing.StandardScaler',
                           {'X': None,
                            'y': None,
                            'sample_weight': None},
                           False,
                           CallType.FUNCTION,
                           [standard_scaler])
packages[f'{standard_scaler_fit.library_path}.{standard_scaler_fit.name}'] = standard_scaler_fit

standard_scaler_get_params = Call('get_params',
                                  'sklearn.preprocessing.StandardScaler',
                                  {'deep': True},
                                  False,
                                  CallType.FUNCTION,
                                  [dataframe_from_dict])
packages[
    f'{standard_scaler_get_params.library_path}.{standard_scaler_get_params.name}'] = standard_scaler_get_params

standard_scaler_set_params = Call('set_params',
                                  'sklearn.preprocessing.StandardScaler',
                                  {'**params': None},
                                  False,
                                  CallType.FUNCTION,
                                  [standard_scaler])
packages[
    f'{standard_scaler_set_params.library_path}.{standard_scaler_set_params.name}'] = standard_scaler_set_params

standard_scaler_inverse_transform = Call('inverse_transform',
                                         'sklearn.preprocessing.StandardScaler',
                                         {'X': None,
                                          'copy': None},
                                         False,
                                         CallType.FUNCTION,
                                         [pd_dataframe])
packages[
    f'{standard_scaler_inverse_transform.library_path}.{standard_scaler_inverse_transform.name}'] = standard_scaler_inverse_transform

standard_scaler_partial_fit = Call('partial_fit',
                                   'sklearn.preprocessing.StandardScaler',
                                   {'X': None,
                                    'y': None,
                                    'sample_weight': None},
                                   False,
                                   CallType.FUNCTION,
                                   [standard_scaler])
packages[f'{standard_scaler_partial_fit.library_path}.{standard_scaler_partial_fit.name}'] = standard_scaler_partial_fit

standard_scaler_get_feature_names_out = Call('get_feature_names_out',
                                             'sklearn.preprocessing.StandardScaler',
                                             {'input_features': None},
                                             False,
                                             CallType.FUNCTION,
                                             [pd_dataframe])
packages[
    f'{standard_scaler_get_feature_names_out.library_path}.{standard_scaler_get_feature_names_out.name}'] = standard_scaler_get_feature_names_out

# sklearn # preprocessing # LabelBinarizer
label_binarizer = Call('LabelBinarizer',
                       'sklearn.preprocessing',
                       {'neg_label': 0,
                        'pos_label': 1,
                        'sparse_output': False},
                       True,
                       CallType.CLASS)
packages[f'{label_binarizer.library_path}.{label_binarizer.name}'] = label_binarizer

label_binarizer_get_params = Call('get_params',
                                  'sklearn.preprocessing.LabelBinarizer',
                                  {'deep': True},
                                  False,
                                  CallType.FUNCTION,
                                  [dataframe_from_dict])
packages[
    f'{label_binarizer_get_params.library_path}.{label_binarizer_get_params.name}'] = label_binarizer_get_params

label_binarizer_set_params = Call('set_params',
                                  'sklearn.preprocessing.LabelBinarizer',
                                  {'**params'},
                                  False,
                                  CallType.FUNCTION,
                                  [label_binarizer])
packages[
    f'{label_binarizer_set_params.library_path}.{label_binarizer_set_params.name}'] = label_binarizer_set_params

label_binarizer_fit_transform = Call('fit_transform',
                                     'sklearn.preprocessing.LabelBinarizer',
                                     {'y': None},
                                     False,
                                     CallType.FUNCTION,
                                     [pd_dataframe])
packages[
    f'{label_binarizer_fit_transform.library_path}.{label_binarizer_fit_transform.name}'] = label_binarizer_fit_transform

label_binarizer_transform = Call('transform',
                                 'sklearn.preprocessing.LabelBinarizer',
                                 {'y': None},
                                 False,
                                 CallType.FUNCTION,
                                 [pd_dataframe])
packages[f'{label_binarizer_transform.library_path}.{label_binarizer_transform.name}'] = label_binarizer_transform

label_binarizer_inverse_transform = Call('inverse_transform',
                                         'sklearn.preprocessing.LabelBinarizer',
                                         {'y': None,
                                          'threshold': None},
                                         False,
                                         CallType.FUNCTION,
                                         [pd_dataframe])
packages[
    f'{label_binarizer_inverse_transform.library_path}.{label_binarizer_inverse_transform.name}'] = label_binarizer_inverse_transform

label_binarizer_fit = Call('fit',
                           'sklearn.preprocessing.LabelBinarizer',
                           {'y': None, },
                           False,
                           CallType.FUNCTION,
                           [label_binarizer])
packages[f'{label_binarizer_fit.library_path}.{label_binarizer_fit.name}'] = label_binarizer_fit

# sklearn # preprocessing # StandardScaler
standard_scaler = Call('StandardScaler',
                       'sklearn.preprocessing',
                       {'copy': True,
                        'with_mean': True,
                        'with_std': True},
                       True,
                       CallType.CLASS)
packages[f'{standard_scaler.library_path}.{standard_scaler.name}'] = standard_scaler

standard_scaler_fit_transform = Call('fit_transform',
                                     'sklearn.preprocessing.StandardScaler',
                                     {'X': None,
                                      'y': None},
                                     False,
                                     CallType.FUNCTION,
                                     [pd_dataframe])
packages[
    f'{standard_scaler_fit_transform.library_path}.{standard_scaler_fit_transform.name}'] = standard_scaler_fit_transform

standard_scaler_transform = Call('transform',
                                 'sklearn.preprocessing.StandardScaler',
                                 {'X': None,
                                  'copy': None},
                                 False,
                                 CallType.FUNCTION,
                                 [pd_dataframe])
packages[f'{standard_scaler_transform.library_path}.{standard_scaler_transform.name}'] = standard_scaler_transform

standard_scaler_fit = Call('fit',
                           'sklearn.preprocessing.StandardScaler',
                           {'X': None,
                            'y': None,
                            'sample_weight': None},
                           False,
                           CallType.FUNCTION,
                           [standard_scaler])
packages[f'{standard_scaler_fit.library_path}.{standard_scaler_fit.name}'] = standard_scaler_fit

# sklearn # preprocessing # add_dummy_feature
add_dummy_feature = Call('add_dummy_feature',
                         'sklearn.preprocessing',
                         {'X': None,
                          'value': None},
                         False,
                         CallType.FUNCTION,
                         [pd_dataframe])
packages[f'{add_dummy_feature.library_path}.{add_dummy_feature.name}'] = add_dummy_feature

# sklearn # preprocessing # binarize
binarize = Call('binarize',
                'sklearn.preprocessing',
                {'X': None,
                 'threshold': 0.0,
                 'copy': True},
                False,
                CallType.FUNCTION,
                [pd_dataframe])
packages[f'{binarize.library_path}.{binarize.name}'] = binarize

# sklearn # preprocessing # label_binarize
label_binarize = Call('label_binarize',
                      'sklearn.preprocessing',
                      {'y': None,
                       'classes': None,
                       'neg_label': 0,
                       'pos_label': 1,
                       'sparse_output': False},
                      False,
                      CallType.FUNCTION,
                      [pd_dataframe])
packages[f'{label_binarize.library_path}.{label_binarize.name}'] = label_binarize

# sklearn # preprocessing # maxabs_scale
maxabs_scale = Call('maxabs_scale',
                    'sklearn.preprocessing',
                    {'X': None,
                     'axis': 0,
                     'copy': True},
                    False,
                    CallType.FUNCTION,
                    [pd_dataframe])
packages[f'{maxabs_scale.library_path}.{maxabs_scale.name}'] = maxabs_scale

# sklearn # preprocessing # maxabs_scale
minmax_scale = Call('minmax_scale',
                    'sklearn.preprocessing',
                    {'X': None,
                     'feature_range': (0, 1),
                     'axis': 0,
                     'copy': True},
                    False,
                    CallType.FUNCTION,
                    [pd_dataframe])
packages[f'{minmax_scale.library_path}.{minmax_scale.name}'] = minmax_scale

# sklearn # preprocessing # normalize
normalize = Call('normalize',
                 'sklearn.preprocessing',
                 {'X': None,
                  'norm': 'l2',
                  'axis': 1,
                  'copy': True,
                  'return_norm': False},
                 False,
                 CallType.FUNCTION,
                 [pd_dataframe])
packages[f'{normalize.library_path}.{normalize.name}'] = normalize

# sklearn # preprocessing # quantile_transform
quantile_transform = Call('quantile_transform',
                          'sklearn.preprocessing',
                          {'X': None,
                           'axis': 0,
                           'n_quantiles': 1000,
                           'output_distribution': 'uniform',
                           'ignore_implicit_zeros': False,
                           'subsample': 1e5,
                           'random_state': None,
                           'copy': True},
                          False,
                          CallType.FUNCTION,
                          [pd_dataframe])
packages[f'{quantile_transform.library_path}.{quantile_transform.name}'] = quantile_transform

# sklearn # preprocessing # robust_scale
robust_scale = Call('robust_scale',
                    'sklearn.preprocessing',
                    {'X': None,
                     'axis': 0,
                     'with_centering': True,
                     'with_scaling': True,
                     'quantile_range': (25.0, 75.0),
                     'copy': True,
                     'unit_variance': False},
                    False,
                    CallType.FUNCTION,
                    [pd_dataframe])
packages[f'{robust_scale.library_path}.{robust_scale.name}'] = robust_scale

# sklearn # preprocessing # scale
scale = Call('scale',
             'sklearn.preprocessing',
             {'X': None,
              'axis': 0,
              'with_mean': True,
              'with_std': True,
              'copy': True},
             False,
             CallType.FUNCTION,
             [pd_dataframe])
packages[f'{scale.library_path}.{scale.name}'] = scale

# sklearn # preprocessing # power_transform
power_transform = Call('power_transform',
                       'sklearn.preprocessing',
                       {'X': None,
                        'method': 'yeo-johnson',
                        'standardize': True,
                        'copy': True},
                       False,
                       CallType.FUNCTION,
                       [pd_dataframe])
packages[f'{power_transform.library_path}.{power_transform.name}'] = power_transform

# sklearn # preprocessing # MinMaxScaler
min_max_scaler = Call('MinMaxScaler',
                      'sklearn.preprocessing',
                      {'feature_range': (0, 1),
                       'copy': True,
                       'clip': False},
                      True,
                      CallType.CLASS)
packages[f'{min_max_scaler.library_path}.{min_max_scaler.name}'] = min_max_scaler

min_max_scaler_fit_transform = Call('fit_transform',
                                    'sklearn.preprocessing.MinMaxScaler',
                                    {'X': None,
                                     'y': None},
                                    False,
                                    CallType.FUNCTION,
                                    [pd_dataframe])
packages[
    f'{min_max_scaler_fit_transform.library_path}.{min_max_scaler_fit_transform.name}'] = min_max_scaler_fit_transform

min_max_scaler_transform = Call('transform',
                                'sklearn.preprocessing.MinMaxScaler',
                                {'X': None},
                                False,
                                CallType.FUNCTION,
                                [pd_dataframe])
packages[f'{min_max_scaler_transform.library_path}.{min_max_scaler_transform.name}'] = min_max_scaler_transform

min_max_scaler_fit = Call('fit',
                          'sklearn.preprocessing.MinMaxScaler',
                          {'X': None,
                           'y': None},
                          False,
                          CallType.FUNCTION,
                          [min_max_scaler])
packages[f'{min_max_scaler_fit.library_path}.{min_max_scaler_fit.name}'] = min_max_scaler_fit

min_max_scaler_get_params = Call('get_params',
                                 'sklearn.preprocessing.MinMaxScaler',
                                 {'deep': True},
                                 False,
                                 CallType.FUNCTION,
                                 [dataframe_from_dict])
packages[
    f'{min_max_scaler_get_params.library_path}.{min_max_scaler_get_params.name}'] = min_max_scaler_get_params

min_max_scaler_set_params = Call('set_params',
                                 'sklearn.preprocessing.MinMaxScaler',
                                 {'**params': None},
                                 False,
                                 CallType.FUNCTION,
                                 [min_max_scaler])
packages[
    f'{min_max_scaler_set_params.library_path}.{min_max_scaler_set_params.name}'] = min_max_scaler_set_params

min_max_scaler_inverse_transform = Call('inverse_transform',
                                        'sklearn.preprocessing.MinMaxScaler',
                                        {'X': None},
                                        False,
                                        CallType.FUNCTION,
                                        [pd_dataframe])
packages[
    f'{min_max_scaler_inverse_transform.library_path}.{min_max_scaler_inverse_transform.name}'] = min_max_scaler_inverse_transform

min_max_scaler_partial_fit = Call('partial_fit',
                                  'sklearn.preprocessing.MinMaxScaler',
                                  {'X': None,
                                   'y': None},
                                  False,
                                  CallType.FUNCTION,
                                  [min_max_scaler])
packages[f'{min_max_scaler_partial_fit.library_path}.{min_max_scaler_partial_fit.name}'] = min_max_scaler_partial_fit

min_max_scaler_get_feature_names_out = Call('get_feature_names_out',
                                            'sklearn.preprocessing.MinMaxScaler',
                                            {'input_features': None},
                                            False,
                                            CallType.FUNCTION,
                                            [pd_dataframe])
packages[
    f'{min_max_scaler_get_feature_names_out.library_path}.{min_max_scaler_get_feature_names_out.name}'] = min_max_scaler_get_feature_names_out

# sklearn # preprocessing # Normalizer
normalizer = Call('Normalizer',
                  'sklearn.preprocessing',
                  {'norm': 'l2',
                   'copy': True},
                  True,
                  CallType.CLASS)
packages[f'{normalizer.library_path}.{normalizer.name}'] = normalizer

normalizer_fit_transform = Call('fit_transform',
                                'sklearn.preprocessing.Normalizer',
                                {'X': None,
                                 'y': None,
                                 '**fit_params': None},
                                False,
                                CallType.FUNCTION,
                                [pd_dataframe])
packages[
    f'{normalizer_fit_transform.library_path}.{normalizer_fit_transform.name}'] = normalizer_fit_transform

normalizer_transform = Call('transform',
                            'sklearn.preprocessing.Normalizer',
                            {'X': None,
                             'copy': None},
                            False,
                            CallType.FUNCTION,
                            [pd_dataframe])
packages[f'{normalizer_transform.library_path}.{normalizer_transform.name}'] = normalizer_transform

normalizer_fit = Call('fit',
                      'sklearn.preprocessing.Normalizer',
                      {'X': None,
                       'y': None},
                      False,
                      CallType.FUNCTION,
                      [normalizer])
packages[f'{normalizer_fit.library_path}.{normalizer_fit.name}'] = normalizer_fit

normalizer_get_params = Call('get_params',
                             'sklearn.preprocessing.Normalizer',
                             {'deep': True},
                             False,
                             CallType.FUNCTION,
                             [dataframe_from_dict])
packages[
    f'{normalizer_get_params.library_path}.{normalizer_get_params.name}'] = normalizer_get_params

normalizer_set_params = Call('set_params',
                             'sklearn.preprocessing.Normalizer',
                             {'**params': None},
                             False,
                             CallType.FUNCTION,
                             [normalizer])
packages[
    f'{normalizer_set_params.library_path}.{normalizer_set_params.name}'] = normalizer_set_params

normalizer_get_feature_names_out = Call('get_feature_names_out',
                                        'sklearn.preprocessing.Normalizer',
                                        {'input_features': None},
                                        False,
                                        CallType.FUNCTION,
                                        [pd_dataframe])
packages[
    f'{normalizer_get_feature_names_out.library_path}.{normalizer_get_feature_names_out.name}'] = normalizer_get_feature_names_out

# sklearn # preprocessing # OneHotEncoder
one_hot_encoder = Call('OneHotEncoder',
                       'sklearn.preprocessing',
                       {'categories': 'auto',
                        'drop': None,
                        'sparse': True,
                        'dtype': "<class 'numpy.float64'>",
                        'handle_unknown': 'error',
                        'min_frequency': None,
                        'max_categories': None},
                       True,
                       CallType.CLASS)
packages[f'{one_hot_encoder.library_path}.{one_hot_encoder.name}'] = one_hot_encoder

one_hot_encoder_fit_transform = Call('fit_transform',
                                     'sklearn.preprocessing.OneHotEncoder',
                                     {'X': None,
                                      'y': None},
                                     False,
                                     CallType.FUNCTION,
                                     [pd_dataframe])
packages[
    f'{one_hot_encoder_fit_transform.library_path}.{one_hot_encoder_fit_transform.name}'] = one_hot_encoder_fit_transform

one_hot_encoder_inverse_transform = Call('inverse_transform',
                                         'sklearn.preprocessing.OneHotEncoder',
                                         {'X': None},
                                         False,
                                         CallType.FUNCTION,
                                         [pd_dataframe])
packages[
    f'{one_hot_encoder_inverse_transform.library_path}.{one_hot_encoder_inverse_transform.name}'] = one_hot_encoder_inverse_transform

one_hot_encoder_transform = Call('transform',
                                 'sklearn.preprocessing.OneHotEncoder',
                                 {'X': None},
                                 False,
                                 CallType.FUNCTION,
                                 [pd_dataframe])
packages[f'{one_hot_encoder_transform.library_path}.{one_hot_encoder_transform.name}'] = one_hot_encoder_transform

one_hot_encoder_fit = Call('fit',
                           'sklearn.preprocessing.OneHotEncoder',
                           {'X': None,
                            'y': None},
                           False,
                           CallType.FUNCTION,
                           [one_hot_encoder])
packages[f'{one_hot_encoder_fit.library_path}.{one_hot_encoder_fit.name}'] = one_hot_encoder_fit

one_hot_encoder_get_params = Call('get_params',
                                  'sklearn.preprocessing.OneHotEncoder',
                                  {'deep': True},
                                  False,
                                  CallType.FUNCTION,
                                  [dataframe_from_dict])
packages[
    f'{one_hot_encoder_get_params.library_path}.{one_hot_encoder_get_params.name}'] = one_hot_encoder_get_params

one_hot_encoder_set_params = Call('set_params',
                                  'sklearn.preprocessing.OneHotEncoder',
                                  {'**params': None},
                                  False,
                                  CallType.FUNCTION,
                                  [one_hot_encoder])
packages[
    f'{one_hot_encoder_set_params.library_path}.{one_hot_encoder_set_params.name}'] = one_hot_encoder_set_params

one_hot_encoder_get_feature_names_out = Call('get_feature_names_out',
                                             'sklearn.preprocessing.OneHotEncoder',
                                             {'input_features': None},
                                             False,
                                             CallType.FUNCTION,
                                             [pd_dataframe])
packages[
    f'{one_hot_encoder_get_feature_names_out.library_path}.{one_hot_encoder_get_feature_names_out.name}'] = one_hot_encoder_get_feature_names_out

one_hot_encoder_get_feature_names = Call('get_feature_names',
                                         'sklearn.preprocessing.OneHotEncoder',
                                         {'input_features': None},
                                         False,
                                         CallType.FUNCTION,
                                         [pd_dataframe])
packages[
    f'{one_hot_encoder_get_feature_names.library_path}.{one_hot_encoder_get_feature_names.name}'] = one_hot_encoder_get_feature_names

# sklearn # preprocessing # OrdinalEncoder
ordinal_encoder = Call('OrdinalEncoder',
                       'sklearn.preprocessing',
                       {'categories': 'auto',
                        'dtype': "<class 'numpy.float64'>",
                        'handle_unknown': 'error',
                        'unknown_value': None,
                        'encoded_missing_value': 'np.nan'},
                       True,
                       CallType.CLASS)
packages[f'{ordinal_encoder.library_path}.{ordinal_encoder.name}'] = ordinal_encoder

ordinal_encoder_fit_transform = Call('fit_transform',
                                     'sklearn.preprocessing.OrdinalEncoder',
                                     {'X': None,
                                      'y': None,
                                      '**fit_params': None},
                                     False,
                                     CallType.FUNCTION,
                                     [pd_dataframe])
packages[
    f'{ordinal_encoder_fit_transform.library_path}.{ordinal_encoder_fit_transform.name}'] = ordinal_encoder_fit_transform

ordinal_encoder_inverse_transform = Call('inverse_transform',
                                         'sklearn.preprocessing.OrdinalEncoder',
                                         {'X': None},
                                         False,
                                         CallType.FUNCTION,
                                         [pd_dataframe])
packages[
    f'{ordinal_encoder_inverse_transform.library_path}.{ordinal_encoder_inverse_transform.name}'] = ordinal_encoder_inverse_transform

ordinal_encoder_transform = Call('transform',
                                 'sklearn.preprocessing.OrdinalEncoder',
                                 {'X': None},
                                 False,
                                 CallType.FUNCTION,
                                 [pd_dataframe])
packages[f'{ordinal_encoder_transform.library_path}.{ordinal_encoder_transform.name}'] = ordinal_encoder_transform

ordinal_encoder_fit = Call('fit',
                           'sklearn.preprocessing.OrdinalEncoder',
                           {'X': None,
                            'y': None},
                           False,
                           CallType.FUNCTION,
                           [ordinal_encoder])
packages[f'{ordinal_encoder_fit.library_path}.{ordinal_encoder_fit.name}'] = ordinal_encoder_fit

ordinal_encoder_get_params = Call('get_params',
                                  'sklearn.preprocessing.OrdinalEncoder',
                                  {'deep': True},
                                  False,
                                  CallType.FUNCTION,
                                  [dataframe_from_dict])
packages[
    f'{ordinal_encoder_get_params.library_path}.{ordinal_encoder_get_params.name}'] = ordinal_encoder_get_params

ordinal_encoder_set_params = Call('set_params',
                                  'sklearn.preprocessing.OrdinalEncoder',
                                  {'**params': None},
                                  False,
                                  CallType.FUNCTION,
                                  [ordinal_encoder])
packages[
    f'{ordinal_encoder_set_params.library_path}.{ordinal_encoder_set_params.name}'] = ordinal_encoder_set_params

ordinal_encoder_get_feature_names_out = Call('get_feature_names_out',
                                             'sklearn.preprocessing.OrdinalEncoder',
                                             {'input_features': None},
                                             False,
                                             CallType.FUNCTION,
                                             [pd_dataframe])
packages[
    f'{ordinal_encoder_get_feature_names_out.library_path}.{ordinal_encoder_get_feature_names_out.name}'] = ordinal_encoder_get_feature_names_out

# sklearn # preprocessing # PolynomialFeatures
polynomial_features = Call('PolynomialFeatures',
                           'sklearn.preprocessing',
                           {'degree': 2,
                            'interaction_only': False,
                            'include_bias': True,
                            'order': 'C'},
                           True,
                           CallType.CLASS)
packages[f'{polynomial_features.library_path}.{polynomial_features.name}'] = polynomial_features

polynomial_features_fit_transform = Call('fit_transform',
                                         'sklearn.preprocessing.PolynomialFeatures',
                                         {'X': None,
                                          'y': None,
                                          '**fit_params': None},
                                         False,
                                         CallType.FUNCTION,
                                         [pd_dataframe])
packages[
    f'{polynomial_features_fit_transform.library_path}.{polynomial_features_fit_transform.name}'] = polynomial_features_fit_transform

polynomial_features_transform = Call('transform',
                                     'sklearn.preprocessing.PolynomialFeatures',
                                     {'X': None},
                                     False,
                                     CallType.FUNCTION,
                                     [pd_dataframe])
packages[
    f'{polynomial_features_transform.library_path}.{polynomial_features_transform.name}'] = polynomial_features_transform

polynomial_features_fit = Call('fit',
                               'sklearn.preprocessing.PolynomialFeatures',
                               {'X': None,
                                'y': None},
                               False,
                               CallType.FUNCTION,
                               [polynomial_features])
packages[f'{polynomial_features_fit.library_path}.{polynomial_features_fit.name}'] = polynomial_features_fit

polynomial_features_get_params = Call('get_params',
                                      'sklearn.preprocessing.PolynomialFeatures',
                                      {'deep': True},
                                      False,
                                      CallType.FUNCTION,
                                      [dataframe_from_dict])
packages[
    f'{polynomial_features_get_params.library_path}.{polynomial_features_get_params.name}'] = polynomial_features_get_params

polynomial_features_set_params = Call('set_params',
                                      'sklearn.preprocessing.PolynomialFeatures',
                                      {'**params': None},
                                      False,
                                      CallType.FUNCTION,
                                      [polynomial_features])
packages[
    f'{polynomial_features_set_params.library_path}.{polynomial_features_set_params.name}'] = polynomial_features_set_params

polynomial_features_get_feature_names_out = Call('get_feature_names_out',
                                                 'sklearn.preprocessing.PolynomialFeatures',
                                                 {'input_features': None},
                                                 False,
                                                 CallType.FUNCTION,
                                                 [pd_dataframe])
packages[
    f'{polynomial_features_get_feature_names_out.library_path}.{polynomial_features_get_feature_names_out.name}'] = polynomial_features_get_feature_names_out

# sklearn # preprocessing # PowerTransformer
power_transformer = Call('PowerTransformer',
                         'sklearn.preprocessing',
                         {'method': 'yeo-johnson',
                          'standardize': True,
                          'copy': True},
                         True,
                         CallType.CLASS)
packages[f'{power_transformer.library_path}.{power_transformer.name}'] = power_transformer

power_transformer_fit_transform = Call('fit_transform',
                                       'sklearn.preprocessing.PowerTransformer',
                                       {'X': None,
                                        'y': None},
                                       False,
                                       CallType.FUNCTION,
                                       [pd_dataframe])
packages[
    f'{power_transformer_fit_transform.library_path}.{power_transformer_fit_transform.name}'] = power_transformer_fit_transform

power_transformer_transform = Call('transform',
                                   'sklearn.preprocessing.PowerTransformer',
                                   {'X': None},
                                   False,
                                   CallType.FUNCTION,
                                   [pd_dataframe])
packages[f'{power_transformer_transform.library_path}.{power_transformer_transform.name}'] = power_transformer_transform

power_transformer_fit = Call('fit',
                             'sklearn.preprocessing.PowerTransformer',
                             {'X': None,
                              'y': None},
                             False,
                             CallType.FUNCTION,
                             [power_transformer])
packages[f'{power_transformer_fit.library_path}.{power_transformer_fit.name}'] = power_transformer_fit

power_transformer_inverse_transform = Call('inverse_transform',
                                           'sklearn.preprocessing.PowerTransformer',
                                           {'X': None},
                                           False,
                                           CallType.FUNCTION,
                                           [pd_dataframe])
packages[
    f'{power_transformer_inverse_transform.library_path}.{power_transformer_inverse_transform.name}'] = power_transformer_inverse_transform

power_transformer_get_params = Call('get_params',
                                    'sklearn.preprocessing.PowerTransformer',
                                    {'deep': True},
                                    False,
                                    CallType.FUNCTION,
                                    [dataframe_from_dict])
packages[
    f'{power_transformer_get_params.library_path}.{power_transformer_get_params.name}'] = power_transformer_get_params

power_transformer_set_params = Call('set_params',
                                    'sklearn.preprocessing.PowerTransformer',
                                    {'**params': None},
                                    False,
                                    CallType.FUNCTION,
                                    [power_transformer])
packages[
    f'{power_transformer_set_params.library_path}.{power_transformer_set_params.name}'] = power_transformer_set_params

power_transformer_get_feature_names_out = Call('get_feature_names_out',
                                               'sklearn.preprocessing.PowerTransformer',
                                               {'input_features': None},
                                               False,
                                               CallType.FUNCTION,
                                               [pd_dataframe])
packages[
    f'{power_transformer_get_feature_names_out.library_path}.{power_transformer_get_feature_names_out.name}'] = power_transformer_get_feature_names_out

# sklearn # preprocessing # QuantileTransformer
quantile_transformer = Call('QuantileTransformer',
                            'sklearn.preprocessing',
                            {'n_quantiles': 1000,
                             'output_distribution': 'uniform',
                             'ignore_implicit_zeros': False,
                             'subsample': 1e5,
                             'random_state': None,
                             'copy': True},
                            True,
                            CallType.CLASS)
packages[f'{quantile_transformer.library_path}.{quantile_transformer.name}'] = quantile_transformer

quantile_transformer_fit_transform = Call('fit_transform',
                                          'sklearn.preprocessing.QuantileTransformer',
                                          {'X': None,
                                           'y': None,
                                           '**fit_params': None},
                                          False,
                                          CallType.FUNCTION,
                                          [pd_dataframe])
packages[
    f'{quantile_transformer_fit_transform.library_path}.{quantile_transformer_fit_transform.name}'] = quantile_transformer_fit_transform

quantile_transformer_transform = Call('transform',
                                      'sklearn.preprocessing.QuantileTransformer',
                                      {'X': None},
                                      False,
                                      CallType.FUNCTION,
                                      [pd_dataframe])
packages[
    f'{quantile_transformer_transform.library_path}.{quantile_transformer_transform.name}'] = quantile_transformer_transform

quantile_transformer_fit = Call('fit',
                                'sklearn.preprocessing.QuantileTransformer',
                                {'X': None,
                                 'y': None},
                                False,
                                CallType.FUNCTION,
                                [quantile_transformer])
packages[f'{quantile_transformer_fit.library_path}.{quantile_transformer_fit.name}'] = quantile_transformer_fit

quantile_transformer_inverse_transform = Call('inverse_transform',
                                              'sklearn.preprocessing.QuantileTransformer',
                                              {'X': None},
                                              False,
                                              CallType.FUNCTION,
                                              [pd_dataframe])
packages[
    f'{quantile_transformer_inverse_transform.library_path}.{quantile_transformer_inverse_transform.name}'] = quantile_transformer_inverse_transform

quantile_transformer_get_params = Call('get_params',
                                       'sklearn.preprocessing.QuantileTransformer',
                                       {'deep': True},
                                       False,
                                       CallType.FUNCTION,
                                       [dataframe_from_dict])
packages[
    f'{quantile_transformer_get_params.library_path}.{quantile_transformer_get_params.name}'] = quantile_transformer_get_params

quantile_transformer_set_params = Call('set_params',
                                       'sklearn.preprocessing.QuantileTransformer',
                                       {'**params': None},
                                       False,
                                       CallType.FUNCTION,
                                       [quantile_transformer])
packages[
    f'{quantile_transformer_set_params.library_path}.{quantile_transformer_set_params.name}'] = quantile_transformer_set_params

quantile_transformer_get_feature_names_out = Call('get_feature_names_out',
                                                  'sklearn.preprocessing.QuantileTransformer',
                                                  {'input_features': None},
                                                  False,
                                                  CallType.FUNCTION,
                                                  [pd_dataframe])
packages[
    f'{quantile_transformer_get_feature_names_out.library_path}.{quantile_transformer_get_feature_names_out.name}'] = quantile_transformer_get_feature_names_out

# sklearn # preprocessing # RobustScaler
robust_scaler = Call('RobustScaler',
                     'sklearn.preprocessing',
                     {'with_centering': True,
                      'with_scaling': True,
                      'quantile_range': (25.0, 75.0),
                      'copy': True,
                      'unit_variance': False},
                     True,
                     CallType.CLASS)
packages[f'{robust_scaler.library_path}.{robust_scaler.name}'] = robust_scaler

robust_scaler_fit_transform = Call('fit_transform',
                                   'sklearn.preprocessing.RobustScaler',
                                   {'X': None,
                                    'y': None},
                                   False,
                                   CallType.FUNCTION,
                                   [pd_dataframe])
packages[
    f'{robust_scaler_fit_transform.library_path}.{robust_scaler_fit_transform.name}'] = robust_scaler_fit_transform

robust_scaler_transform = Call('transform',
                               'sklearn.preprocessing.RobustScaler',
                               {'X': None},
                               False,
                               CallType.FUNCTION,
                               [pd_dataframe])
packages[f'{robust_scaler_transform.library_path}.{robust_scaler_transform.name}'] = robust_scaler_transform

robust_scaler_fit = Call('fit',
                         'sklearn.preprocessing.RobustScaler',
                         {'X': None,
                          'y': None},
                         False,
                         CallType.FUNCTION,
                         [robust_scaler])
packages[f'{robust_scaler_fit.library_path}.{robust_scaler_fit.name}'] = robust_scaler_fit

robust_scaler_inverse_transform = Call('inverse_transform',
                                       'sklearn.preprocessing.RobustScaler',
                                       {'X': None},
                                       False,
                                       CallType.FUNCTION,
                                       [pd_dataframe])
packages[
    f'{robust_scaler_inverse_transform.library_path}.{robust_scaler_inverse_transform.name}'] = robust_scaler_inverse_transform

robust_scaler_get_params = Call('get_params',
                                'sklearn.preprocessing.RobustScaler',
                                {'deep': True},
                                False,
                                CallType.FUNCTION,
                                [dataframe_from_dict])
packages[
    f'{robust_scaler_get_params.library_path}.{robust_scaler_get_params.name}'] = robust_scaler_get_params

robust_scaler_set_params = Call('set_params',
                                'sklearn.preprocessing.RobustScaler',
                                {'**params': None},
                                False,
                                CallType.FUNCTION,
                                [robust_scaler])
packages[
    f'{robust_scaler_set_params.library_path}.{robust_scaler_set_params.name}'] = robust_scaler_set_params

robust_scaler_get_feature_names_out = Call('get_feature_names_out',
                                           'sklearn.preprocessing.RobustScaler',
                                           {'input_features': None},
                                           False,
                                           CallType.FUNCTION,
                                           [pd_dataframe])
packages[
    f'{robust_scaler_get_feature_names_out.library_path}.{robust_scaler_get_feature_names_out.name}'] = robust_scaler_get_feature_names_out

# sklearn # preprocessing # SplineTransformer
spline_transformer = Call('SplineTransformer',
                          'sklearn.preprocessing',
                          {'n_knots': 5,
                           'degree': 3,
                           'knots': 'uniform',
                           'extrapolation': 'constant',
                           'include_bias': True,
                           'order': 'C'},
                          True,
                          CallType.CLASS)
packages[f'{spline_transformer.library_path}.{spline_transformer.name}'] = spline_transformer

spline_transformer_fit_transform = Call('fit_transform',
                                        'sklearn.preprocessing.SplineTransformer',
                                        {'X': None,
                                         'y': None,
                                         '**fit_params': None},
                                        False,
                                        CallType.FUNCTION,
                                        [pd_dataframe])
packages[
    f'{spline_transformer_fit_transform.library_path}.{spline_transformer_fit_transform.name}'] = spline_transformer_fit_transform

spline_transformer_transform = Call('transform',
                                    'sklearn.preprocessing.SplineTransformer',
                                    {'X': None},
                                    False,
                                    CallType.FUNCTION,
                                    [pd_dataframe])
packages[
    f'{spline_transformer_transform.library_path}.{spline_transformer_transform.name}'] = spline_transformer_transform

spline_transformer_fit = Call('fit',
                              'sklearn.preprocessing.SplineTransformer',
                              {'X': None,
                               'y': None,
                               'sample_weight': None},
                              False,
                              CallType.FUNCTION,
                              [spline_transformer])
packages[f'{spline_transformer_fit.library_path}.{spline_transformer_fit.name}'] = spline_transformer_fit

spline_transformer_get_params = Call('get_params',
                                     'sklearn.preprocessing.SplineTransformer',
                                     {'deep': True},
                                     False,
                                     CallType.FUNCTION,
                                     [dataframe_from_dict])
packages[
    f'{spline_transformer_get_params.library_path}.{spline_transformer_get_params.name}'] = spline_transformer_get_params

spline_transformer_set_params = Call('set_params',
                                     'sklearn.preprocessing.SplineTransformer',
                                     {'**params': None},
                                     False,
                                     CallType.FUNCTION,
                                     [spline_transformer])
packages[
    f'{spline_transformer_set_params.library_path}.{spline_transformer_set_params.name}'] = spline_transformer_set_params

spline_transformer_get_feature_names_out = Call('get_feature_names_out',
                                                'sklearn.preprocessing.SplineTransformer',
                                                {'input_features': None},
                                                False,
                                                CallType.FUNCTION,
                                                [pd_dataframe])
packages[
    f'{spline_transformer_get_feature_names_out.library_path}.{spline_transformer_get_feature_names_out.name}'] = spline_transformer_get_feature_names_out

# sklearn.preprocessing.MaxAbsScaler
max_abs_scaler = Call('MaxAbsScaler',
                      'sklearn.preprocessing',
                      {'copy': True},
                      True,
                      CallType.CLASS)
packages[f'{max_abs_scaler.library_path}.{max_abs_scaler.name}'] = max_abs_scaler

max_abs_scaler_get_params = Call('get_params',
                                 'sklearn.preprocessing.MaxAbsScaler',
                                 {'deep': True},
                                 False,
                                 CallType.FUNCTION,
                                 [dataframe_from_dict])
packages[
    f'{max_abs_scaler_get_params.library_path}.{max_abs_scaler_get_params.name}'] = max_abs_scaler_get_params

max_abs_scaler_set_params = Call('set_params',
                                 'sklearn.preprocessing.MaxAbsScaler',
                                 {'**params': None},
                                 False,
                                 CallType.FUNCTION,
                                 [max_abs_scaler])
packages[
    f'{max_abs_scaler_set_params.library_path}.{max_abs_scaler_set_params.name}'] = max_abs_scaler_set_params

max_abs_scaler_inverse_transform = Call('inverse_transform',
                                        'sklearn.preprocessing.MaxAbsScaler',
                                        {'X': None},
                                        False,
                                        CallType.FUNCTION,
                                        [pd_dataframe])
packages[
    f'{max_abs_scaler_inverse_transform.library_path}.{max_abs_scaler_inverse_transform.name}'] = max_abs_scaler_inverse_transform

max_abs_scaler_fit_transform = Call('fit_transform',
                                    'sklearn.preprocessing.MaxAbsScaler',
                                    {'X': None,
                                     'y': None},
                                    False,
                                    CallType.FUNCTION,
                                    [pd_dataframe])
packages[
    f'{max_abs_scaler_fit_transform.library_path}.{max_abs_scaler_fit_transform.name}'] = max_abs_scaler_fit_transform

max_abs_scaler_transform = Call('transform',
                                'sklearn.preprocessing.MaxAbsScaler',
                                {'X': None},
                                False,
                                CallType.FUNCTION,
                                [pd_dataframe])
packages[f'{max_abs_scaler_transform.library_path}.{max_abs_scaler_transform.name}'] = max_abs_scaler_transform

max_abs_scaler_fit = Call('fit',
                          'sklearn.preprocessing.MaxAbsScaler',
                          {'X': None,
                           'y': None},
                          False,
                          CallType.FUNCTION,
                          [max_abs_scaler])
packages[f'{max_abs_scaler_fit.library_path}.{max_abs_scaler_fit.name}'] = max_abs_scaler_fit

max_abs_scaler_partial_fit = Call('partial_fit',
                                  'sklearn.preprocessing.MaxAbsScaler',
                                  {'X': None,
                                   'y': None},
                                  False,
                                  CallType.FUNCTION,
                                  [max_abs_scaler])
packages[f'{max_abs_scaler_partial_fit.library_path}.{max_abs_scaler_partial_fit.name}'] = max_abs_scaler_partial_fit

max_abs_scaler_get_feature_names_out = Call('get_feature_names_out',
                                            'sklearn.preprocessing.MaxAbsScaler',
                                            {'input_features': None},
                                            False,
                                            CallType.FUNCTION,
                                            [pd_dataframe])
packages[
    f'{max_abs_scaler_get_feature_names_out.library_path}.{max_abs_scaler_get_feature_names_out.name}'] = max_abs_scaler_get_feature_names_out

# sklearn # preprocessing # scale
preprocessing_scale = Call('scale',
                           'sklearn.preprocessing',
                           {'X': None,
                            'axis': 0,
                            'with_mean': True,
                            'with_std': True,
                            'copy': True},
                           False,
                           CallType.FUNCTION,
                           [pd_dataframe])
packages[f'{preprocessing_scale.library_path}.{preprocessing_scale.name}'] = preprocessing_scale

# sklearn # model_selection
model_selection = Call(name='model_selection', library_path='sklearn', call_type=CallType.PACKAGE)
packages['sklearn.model_selection'] = model_selection

train_test_split_call = Call('train_test_split',
                             'sklearn.model_selection',
                             {'*arrays': None,
                              'test_size': None,
                              'train_size': None,
                              'random_state': None,
                              'shuffle': True,
                              'stratify': None},
                             False,
                             CallType.FUNCTION,
                             [pd_dataframe, pd_dataframe, pd_dataframe, pd_dataframe])
packages[f'{train_test_split_call.library_path}.{train_test_split_call.name}'] = train_test_split_call

cross_val_score = Call('cross_val_score',
                       'sklearn.model_selection',
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
                       CallType.FUNCTION,
                       [pd_dataframe])
packages[f'{cross_val_score.library_path}.{cross_val_score.name}'] = cross_val_score

# sklearn # metrics
packages['sklearn.metrics'] = Call(name='metrics', library_path='sklearn', call_type=CallType.PACKAGE)

accuracy_score = Call('accuracy_score',
                      'sklearn.metrics',
                      {'y_true': None,
                       'y_pred': None,
                       'normalize': True,
                       'sample_weight': None},
                      False,
                      CallType.FUNCTION,
                      [])  # TODO Determine the return type of not classes
packages[f'{accuracy_score.library_path}.{accuracy_score.name}'] = accuracy_score

# sklearn # ensemble
packages['sklearn.ensemble'] = Call(name='ensemble', library_path='sklearn', call_type=CallType.PACKAGE)

# sklearn # ensemble # RandomForestClassifier
random_forest_classifier = Call('RandomForestClassifier',
                                'sklearn.ensemble',
                                {'n_estimators': 100},
                                True,
                                CallType.CLASS)
packages[f'{random_forest_classifier.library_path}.{random_forest_classifier.name}'] = random_forest_classifier

random_forest_classifier_fit = Call('fit',
                                    'sklearn.ensemble.RandomForestClassifier',
                                    {'X': None,
                                     'y': None,
                                     'sample_weight': None},
                                    False,
                                    CallType.FUNCTION,
                                    [random_forest_classifier])
packages[
    f'{random_forest_classifier_fit.library_path}.{random_forest_classifier_fit.name}'] = random_forest_classifier_fit

random_forest_classifier_predict = Call('predict',
                                        'sklearn.ensemble.RandomForestClassifier',
                                        {'X': None},
                                        False,
                                        CallType.FUNCTION,
                                        [pd_dataframe])
packages[
    f'{random_forest_classifier_predict.library_path}.{random_forest_classifier_predict.name}'] = random_forest_classifier_predict

# sklearn # ensemble # GradientBoostingClassifier
gradient_boosting_classifier = Call('GradientBoostingClassifier',
                                    'sklearn.ensemble',
                                    {},
                                    True,
                                    CallType.CLASS)
packages[
    f'{gradient_boosting_classifier.library_path}.{gradient_boosting_classifier.name}'] = gradient_boosting_classifier

gradient_boosting_classifier_fit = Call('fit',
                                        'sklearn.ensemble.GradientBoostingClassifier',
                                        {'X': None,
                                         'y': None,
                                         'sample_weight': None,
                                         'monitor': None},
                                        False,
                                        CallType.FUNCTION,
                                        [gradient_boosting_classifier])
packages[
    f'{gradient_boosting_classifier_fit.library_path}.{gradient_boosting_classifier_fit.name}'] = gradient_boosting_classifier_fit

gradient_boosting_classifier_predict = Call('predict',
                                            'sklearn.ensemble.GradientBoostingClassifier',
                                            {'X': None},
                                            False,
                                            CallType.FUNCTION,
                                            [pd_dataframe])
packages[
    f'{gradient_boosting_classifier_predict.library_path}.{gradient_boosting_classifier_predict.name}'] = gradient_boosting_classifier_predict

# sklearn # linear_model
packages['sklearn.linear_model'] = Call(name='linear_model', library_path='sklearn', call_type=CallType.PACKAGE)

# sklearn # linear_model # LogisticRegression
logistic_regression = Call('LogisticRegression',
                           'sklearn.linear_model',
                           {'penalty': 'l2'},
                           True,
                           CallType.CLASS)
packages[f'{logistic_regression.library_path}.{logistic_regression.name}'] = logistic_regression

logistic_regression_fit = Call('fit',
                               'sklearn.linear_model.LogisticRegression',
                               {'X': None,
                                'y': None,
                                'sample_weight': None},
                               False,
                               CallType.FUNCTION,
                               [logistic_regression])
packages[f'{logistic_regression_fit.library_path}.{logistic_regression_fit.name}'] = logistic_regression_fit

logistic_regression_predict = Call('predict',
                                   'sklearn.linear_model.LogisticRegression',
                                   {'X': None},
                                   False,
                                   CallType.FUNCTION,
                                   [pd_dataframe])
packages[f'{logistic_regression_predict.library_path}.{logistic_regression_predict.name}'] = logistic_regression_predict

# sklearn # linear_model # SGDClassifier
sgd_classifier = Call('SGDClassifier',
                      'sklearn.linear_model',
                      {'loss': 'hinge'},
                      True,
                      CallType.CLASS)
packages[f'{sgd_classifier.library_path}.{sgd_classifier.name}'] = sgd_classifier

sgd_classifier_fit = Call('fit',
                          'sklearn.linear_model.SGDClassifier',
                          {'X': None,
                           'y': None,
                           'coef_init': None,
                           'intercept_init': None,
                           'sample_weight': None},
                          False,
                          CallType.FUNCTION,
                          [sgd_classifier])
packages[f'{sgd_classifier_fit.library_path}.{sgd_classifier_fit.name}'] = sgd_classifier_fit

sgd_classifier_predict = Call('predict',
                              'sklearn.linear_model.SGDClassifier',
                              {'X': None},
                              False,
                              CallType.FUNCTION,
                              [pd_dataframe])
packages[f'{sgd_classifier_predict.library_path}.{sgd_classifier_predict.name}'] = sgd_classifier_predict

# sklearn # svm
packages['sklearn.svm'] = Call(name='svm', library_path='sklearn', call_type=CallType.PACKAGE)

# sklearn # svm # SVC
svc = Call('SVC',
           'sklearn.svm',
           {'C': 1.0,
            'kernel': 'rbf',
            'degree': 3,
            'gamma': 'scale',
            'coef0': 0.0,
            'shrinking': True,
            'probability': False,
            'tol': 0.001,
            'cache_size': 200,
            'class_weight': None,
            'verbose': False,
            'max_iter': -1,
            'decision_function_shape': 'ovr',
            'break_ties': False,
            'random_state': None},
           True,
           CallType.CLASS)
packages[f'{svc.library_path}.{svc.name}'] = svc

svc_fit = Call('fit',
               'sklearn.svm.SVC',
               {'X': None,
                'y': None,
                'sample_weight': None},
               False,
               CallType.FUNCTION,
               [svc])
packages[f'{svc_fit.library_path}.{svc_fit.name}'] = svc_fit

svc_predict = Call('predict',
                   'sklearn.svm.SVC',
                   {'X': None},
                   False,
                   CallType.FUNCTION,
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

# {"uri": "http://kglids.org/resource/library/sklearn", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/base", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/base/BaseEstimator", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/base/TransformerMixin", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/base/RegressorMixin", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/base/ClassifierMixin", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/base/clone", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/pipeline", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/pipeline/Pipeline", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/pipeline/FeatureUnion", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/pipeline/make_pipeline", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/pipeline/make_union", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/preprocessing", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/preprocessing/StandardScaler", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/preprocessing/LabelBinarizer", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/preprocessing/Imputer", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/preprocessing/MinMaxScaler", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/preprocessing/PolynomialFeatures", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/preprocessing/Normalizer", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/preprocessing/RobustScaler", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/preprocessing/binarize", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/preprocessing/normalize", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/preprocessing/LabelEncoder", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/preprocessing/LabelEncoder/fit_transform", "contain": [], "type": "http://kglids.org/ontology/Function"}, {"uri": "http://kglids.org/resource/library/sklearn/preprocessing/LabelEncoder/fit", "contain": [], "type": "http://kglids.org/ontology/Function"}, {"uri": "http://kglids.org/resource/library/sklearn/preprocessing/LabelEncoder/transform", "contain": [], "type": "http://kglids.org/ontology/Function"}], "type": "http://kglids.org/ontology/Class"}, {"uri": "http://kglids.org/resource/library/sklearn/preprocessing/OneHotEncoder", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/preprocessing/MaxAbsScaler", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/preprocessing/FunctionTransformer", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/preprocessing/KBinsDiscretizer", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/preprocessing/scale", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/preprocessing/OrdinalEncoder", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/preprocessing/QuantileTransformer", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/preprocessing/PowerTransformer", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/preprocessing/minmax_scale", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/preprocessing/MultiLabelBinarizer", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/preprocessing/power_transform", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/preprocessing/*", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/preprocessing/label_binarize", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/preprocessing/Binarizer", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/preprocessing/data", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/preprocessing/data/QuantileTransformer", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/preprocessing/robust_scale", "contain": [], "type": null}], "type": "http://kglids.org/ontology/Package"}, {"uri": "http://kglids.org/resource/library/sklearn/svm", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/svm/SVR", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/svm/SVC", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/svm/SVC/fit", "contain": [], "type": "http://kglids.org/ontology/Function"}, {"uri": "http://kglids.org/resource/library/sklearn/svm/SVC/predict", "contain": [], "type": "http://kglids.org/ontology/Function"}], "type": "http://kglids.org/ontology/Class"}, {"uri": "http://kglids.org/resource/library/sklearn/svm/LinearSVC", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/svm/NuSVC", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/svm/LinearSVR", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/svm/OneClassSVM", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/svm/*", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/svm/NuSVR", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/svm/classes", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/svm/classes/SVR", "contain": [], "type": null}], "type": null}], "type": "http://kglids.org/ontology/Package"}, {"uri": "http://kglids.org/resource/library/sklearn/model_selection", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/model_selection/GridSearchCV", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/model_selection/cross_val_score", "contain": [], "type": "http://kglids.org/ontology/Function"}, {"uri": "http://kglids.org/resource/library/sklearn/model_selection/train_test_split", "contain": [], "type": "http://kglids.org/ontology/Function"}, {"uri": "http://kglids.org/resource/library/sklearn/model_selection/RandomizedSearchCV", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/model_selection/KFold", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/model_selection/learning_curve", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/model_selection/StratifiedKFold", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/model_selection/StratifiedShuffleSplit", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/model_selection/cross_val_predict", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/model_selection/*", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/model_selection/GroupKFold", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/model_selection/ShuffleSplit", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/model_selection/cross_validate", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/model_selection/RepeatedStratifiedKFold", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/model_selection/RepeatedKFold", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/model_selection/TimeSeriesSplit", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/model_selection/validation_curve", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/model_selection/ParameterGrid", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/model_selection/LeaveOneOut", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/model_selection/GroupShuffleSplit", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/model_selection/LeaveOneGroupOut", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/model_selection/ParameterSampler", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/model_selection/PredefinedSplit", "contain": [], "type": null}], "type": "http://kglids.org/ontology/Package"}, {"uri": "http://kglids.org/resource/library/sklearn/ensemble", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/ensemble/RandomForestRegressor", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/ensemble/RandomForestClassifier", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/ensemble/RandomForestClassifier/fit", "contain": [], "type": "http://kglids.org/ontology/Function"}, {"uri": "http://kglids.org/resource/library/sklearn/ensemble/RandomForestClassifier/predict", "contain": [], "type": "http://kglids.org/ontology/Function"}], "type": "http://kglids.org/ontology/Class"}, {"uri": "http://kglids.org/resource/library/sklearn/ensemble/AdaBoostClassifier", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/ensemble/GradientBoostingClassifier", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/ensemble/GradientBoostingClassifier/fit", "contain": [], "type": "http://kglids.org/ontology/Function"}, {"uri": "http://kglids.org/resource/library/sklearn/ensemble/GradientBoostingClassifier/predict", "contain": [], "type": "http://kglids.org/ontology/Function"}], "type": "http://kglids.org/ontology/Class"}, {"uri": "http://kglids.org/resource/library/sklearn/ensemble/ExtraTreesClassifier", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/ensemble/GradientBoostingRegressor", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/ensemble/BaggingRegressor", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/ensemble/RandomTreesEmbedding", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/ensemble/BaggingClassifier", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/ensemble/VotingClassifier", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/ensemble/StackingClassifier", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/ensemble/AdaBoostRegressor", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/ensemble/IsolationForest", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/ensemble/ExtraTreesRegressor", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/ensemble/forest", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/ensemble/forest/ExtraTreesClassifier", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/ensemble/forest/RandomForestRegressor", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/ensemble/gradient_boosting", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/ensemble/gradient_boosting/GradientBoostingClassifier", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/ensemble/VotingRegressor", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/ensemble/StackingRegressor", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/ensemble/HistGradientBoostingClassifier", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/ensemble/bagging", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/ensemble/bagging/BaggingClassifier", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/ensemble/*", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/ensemble/HistGradientBoostingRegressor", "contain": [], "type": null}], "type": "http://kglids.org/ontology/Package"}, {"uri": "http://kglids.org/resource/library/sklearn/metrics", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/metrics/mean_squared_error", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/mean_absolute_error", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/r2_score", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/precision_score", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/recall_score", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/accuracy_score", "contain": [], "type": "http://kglids.org/ontology/Function"}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/classification_report", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/confusion_matrix", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/average_precision_score", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/precision_recall_curve", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/f1_score", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/roc_curve", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/roc_auc_score", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/auc", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/pairwise", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/metrics/pairwise/cosine_similarity", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/pairwise/linear_kernel", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/pairwise/sigmoid_kernel", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/pairwise/euclidean_distances", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/pairwise/pairwise_distances", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/pairwise/cosine_distances", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/pairwise/rbf_kernel", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/make_scorer", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/mean_squared_log_error", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/pairwise_distances", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/explained_variance_score", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/plot_confusion_matrix", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/plot_precision_recall_curve", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/balanced_accuracy_score", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/log_loss", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/hamming_loss", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/silhouette_score", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/fbeta_score", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/cohen_kappa_score", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/plot_roc_curve", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/*", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/median_absolute_error", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/jaccard_similarity_score", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/matthews_corrcoef", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/precision_recall_fscore_support", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/silhouette_samples", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/ConfusionMatrixDisplay", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/brier_score_loss", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/jaccard_score", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/multilabel_confusion_matrix", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/mean_absolute_percentage_error", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/scorer", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/cluster", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/metrics/cluster/normalized_mutual_info_score", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/cluster/adjusted_rand_score", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/adjusted_rand_score", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/davies_bouldin_score", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/max_error", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/label_ranking_average_precision_score", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/label_ranking_loss", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/coverage_error", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/normalized_mutual_info_score", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/adjusted_mutual_info_score", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/mutual_info_score", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/mean_poisson_deviance", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/mean_gamma_deviance", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/pairwise_distances_argmin", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/classification", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/metrics/classification/accuracy_score", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/classification/log_loss", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/completeness_score", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/fowlkes_mallows_score", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/homogeneity_score", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/v_measure_score", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/RocCurveDisplay", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/rand_score", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/euclidean_distances", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/metrics/pairwise_distances_argmin_min", "contain": [], "type": null}], "type": "http://kglids.org/ontology/Package"}, {"uri": "http://kglids.org/resource/library/sklearn/decomposition", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/decomposition/LatentDirichletAllocation", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/decomposition/FastICA", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/decomposition/MiniBatchDictionaryLearning", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/decomposition/TruncatedSVD", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/decomposition/IncrementalPCA", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/decomposition/KernelPCA", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/decomposition/SparsePCA", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/decomposition/PCA", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/decomposition/NMF", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/decomposition/FactorAnalysis", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/decomposition/RandomizedPCA", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/decomposition/pca", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/decomposition/MiniBatchSparsePCA", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/feature_extraction", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/feature_extraction/text", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/feature_extraction/text/TfidfVectorizer", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/feature_extraction/text/CountVectorizer", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/feature_extraction/text/TfidfTransformer", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/feature_extraction/text/HashingVectorizer", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/feature_extraction/text/ENGLISH_STOP_WORDS", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/feature_extraction/DictVectorizer", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/feature_extraction/stop_words", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/feature_extraction/stop_words/ENGLISH_STOP_WORDS", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/feature_extraction/FeatureHasher", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/linear_model/LinearRegression", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/Perceptron", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/LogisticRegression", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/linear_model/LogisticRegression/fit", "contain": [], "type": "http://kglids.org/ontology/Function"}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/LogisticRegression/predict", "contain": [], "type": "http://kglids.org/ontology/Function"}], "type": "http://kglids.org/ontology/Class"}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/SGDClassifier", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/linear_model/SGDClassifier/fit", "contain": [], "type": "http://kglids.org/ontology/Function"}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/SGDClassifier/predict", "contain": [], "type": "http://kglids.org/ontology/Function"}], "type": "http://kglids.org/ontology/Class"}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/OrthogonalMatchingPursuit", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/RANSACRegressor", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/ElasticNetCV", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/HuberRegressor", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/Ridge", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/Lasso", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/LassoCV", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/Lars", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/BayesianRidge", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/LogisticRegressionCV", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/RidgeClassifier", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/RidgeCV", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/ElasticNet", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/LassoLarsCV", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/SGDRegressor", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/PassiveAggressiveClassifier", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/LassoLarsIC", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/TheilSenRegressor", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/RidgeClassifierCV", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/PassiveAggressiveRegressor", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/ARDRegression", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/ridge_regression", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/LassoLars", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/RandomizedLogisticRegression", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/PoissonRegressor", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/MultiTaskElasticNet", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/*", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/logistic", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/linear_model/logistic/LogisticRegression", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/stochastic_gradient", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/linear_model/stochastic_gradient/SGDRegressor", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/ridge", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/linear_model/ridge/Ridge", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/RandomizedLasso", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/linear_model/LarsCV", "contain": [], "type": null}], "type": "http://kglids.org/ontology/Package"}, {"uri": "http://kglids.org/resource/library/sklearn/naive_bayes", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/naive_bayes/GaussianNB", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/naive_bayes/MultinomialNB", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/naive_bayes/CategoricalNB", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/naive_bayes/BernoulliNB", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/naive_bayes/ComplementNB", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/naive_bayes/*", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/neighbors", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/neighbors/KNeighborsClassifier", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/neighbors/KNeighborsRegressor", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/neighbors/LocalOutlierFactor", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/neighbors/NearestNeighbors", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/neighbors/KDTree", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/neighbors/NearestCentroid", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/neighbors/RadiusNeighborsClassifier", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/neighbors/KernelDensity", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/neighbors/kneighbors_graph", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/neighbors/NeighborhoodComponentsAnalysis", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/neighbors/RadiusNeighborsRegressor", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/neighbors/BallTree", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/neighbors/*", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/neighbors/nearest_centroid", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/neighbors/nearest_centroid/NearestCentroid", "contain": [], "type": null}], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/tree", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/tree/DecisionTreeClassifier", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/tree/DecisionTreeRegressor", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/tree/export_graphviz", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/tree/ExtraTreeClassifier", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/tree/plot_tree", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/tree/export_text", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/tree/ExtraTreeRegressor", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/tree/export", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/tree/export/export_text", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/tree/tree", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/tree/*", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/random_projection", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/random_projection/SparseRandomProjection", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/random_projection/GaussianRandomProjection", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/random_projection/sparse_random_matrix", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/manifold", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/manifold/TSNE", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/manifold/LocallyLinearEmbedding", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/manifold/MDS", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/manifold/Isomap", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/manifold/SpectralEmbedding", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/feature_selection", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/feature_selection/SelectKBest", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/feature_selection/f_regression", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/feature_selection/chi2", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/feature_selection/GenericUnivariateSelect", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/feature_selection/f_classif", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/feature_selection/SelectFromModel", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/feature_selection/VarianceThreshold", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/feature_selection/RFECV", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/feature_selection/RFE", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/feature_selection/mutual_info_regression", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/feature_selection/mutual_info_classif", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/feature_selection/univariate_selection", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/feature_selection/univariate_selection/SelectKBest", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/feature_selection/univariate_selection/chi2", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/feature_selection/univariate_selection/f_classif", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/feature_selection/SelectPercentile", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/feature_selection/*", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/feature_selection/SequentialFeatureSelector", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/feature_selection/SelectFpr", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/neural_network", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/neural_network/MLPClassifier", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/neural_network/MLPRegressor", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/neural_network/*", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/neural_network/BernoulliRBM", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/discriminant_analysis", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/discriminant_analysis/LinearDiscriminantAnalysis", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/discriminant_analysis/QuadraticDiscriminantAnalysis", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/gaussian_process", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/gaussian_process/kernels", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/gaussian_process/kernels/RBF", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/gaussian_process/kernels/DotProduct", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/gaussian_process/kernels/Sum", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/gaussian_process/kernels/ConstantKernel", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/gaussian_process/kernels/WhiteKernel", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/gaussian_process/kernels/Matern", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/gaussian_process/kernels/RationalQuadratic", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/gaussian_process/kernels/ExpSineSquared", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/gaussian_process/GaussianProcessClassifier", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/gaussian_process/GaussianProcessRegressor", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/cluster", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/cluster/KMeans", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/cluster/Birch", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/cluster/DBSCAN", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/cluster/AffinityPropagation", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/cluster/AgglomerativeClustering", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/cluster/MiniBatchKMeans", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/cluster/*", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/cluster/SpectralClustering", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/cluster/MeanShift", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/cluster/estimate_bandwidth", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/cluster/OPTICS", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/cluster/k_means", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/cluster/FeatureAgglomeration", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/cross_validation", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/cross_validation/train_test_split", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/cross_validation/KFold", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/cross_validation/StratifiedKFold", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/cross_validation/ShuffleSplit", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/cross_validation/*", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/cross_validation/cross_val_score", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/cross_validation/StratifiedShuffleSplit", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/grid_search", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/grid_search/GridSearchCV", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/grid_search/RandomizedSearchCV", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/externals", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/externals/joblib", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/externals/six", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/externals/six/StringIO", "contain": [], "type": null}], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/impute", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/impute/SimpleImputer", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/impute/MissingIndicator", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/impute/KNNImputer", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/impute/IterativeImputer", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/impute/_iterative", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/impute/_iterative/IterativeImputer", "contain": [], "type": null}], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/compose", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/compose/ColumnTransformer", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/compose/make_column_transformer", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/compose/make_column_selector", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/compose/TransformedTargetRegressor", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/multioutput", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/multioutput/MultiOutputClassifier", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/multioutput/RegressorChain", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/multioutput/MultiOutputRegressor", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/multiclass", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/multiclass/OneVsRestClassifier", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/multiclass/OneVsOneClassifier", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/utils", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/utils/shuffle", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/utils/resample", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/utils/multiclass", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/utils/multiclass/type_of_target", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/utils/multiclass/unique_labels", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/utils/class_weight", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/utils/class_weight/compute_sample_weight", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/utils/class_weight/compute_class_weight", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/utils/extmath", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/utils/extmath/safe_sparse_dot", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/utils/extmath/density", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/utils/extmath/randomized_svd", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/utils/check_X_y", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/utils/check_array", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/utils/optimize", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/utils/optimize/_check_optimize_result", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/utils/fixes", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/utils/fixes/signature", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/utils/testing", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/utils/testing/ignore_warnings", "contain": [], "type": null}], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/datasets", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/datasets/load_boston", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/datasets/samples_generator", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/datasets/samples_generator/make_blobs", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/datasets/samples_generator/make_regression", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/datasets/make_blobs", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/datasets/make_classification", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/datasets/load_digits", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/datasets/make_regression", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/datasets/load_iris", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/datasets/load_breast_cancer", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/datasets/load_wine", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/datasets/load_diabetes", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/datasets/make_moons", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/datasets/make_circles", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/datasets/fetch_mldata", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/datasets/fetch_20newsgroups", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/datasets/fetch_20newsgroups_vectorized", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/datasets/make_gaussian_quantiles", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/datasets/fetch_openml", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/datasets/make_multilabel_classification", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/datasets/load_files", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/datasets/make_friedman1", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/covariance", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/covariance/EllipticEnvelope", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/mixture", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/mixture/GaussianMixture", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/mixture/BayesianGaussianMixture", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/dummy", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/dummy/DummyClassifier", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/dummy/DummyRegressor", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/inspection", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/inspection/permutation_importance", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/inspection/plot_partial_dependence", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/inspection/partial_dependence", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/experimental", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/experimental/enable_iterative_imputer", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/experimental/enable_hist_gradient_boosting", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/calibration", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/calibration/CalibratedClassifierCV", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/calibration/calibration_curve", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/set_config", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/exceptions", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/exceptions/ConvergenceWarning", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/exceptions/NotFittedError", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/exceptions/DataConversionWarning", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/kernel_ridge", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/kernel_ridge/KernelRidge", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/cross_decomposition", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/cross_decomposition/PLSRegression", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/cross_decomposition/PLSSVD", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/*", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/semi_supervised", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/semi_supervised/LabelPropagation", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/lda", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/re", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/learning_curve", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/learning_curve/learning_curve", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/sklearn/isotonic", "contain": [{"uri": "http://kglids.org/resource/library/sklearn/isotonic/IsotonicRegression", "contain": [], "type": null}], "type": null}], "type": "http://kglids.org/ontology/Library"},

# ####### matplotlib ##########
packages['matplotlib'] = Call(name='matplotlib', call_type=CallType.LIBRARY)

# matplotlib # figure
packages['matplotlib.figure'] = Call(name='figure', library_path='matplotlib', call_type=CallType.PACKAGE)

mpl_figure = Call('Figure',
                  'matplotlib.figure',
                  {'figsize': None,
                   'dpi': None,
                   'facecolor': None,
                   'edgecolor': None,
                   'linewidth': 0.0,
                   'frameon': None,
                   'subplotpars': None,
                   'tight_layout': None,
                   'constrained_layout': None,
                   'layout': None},  # Note: There is a * and **kwargs
                  True,
                  CallType.CLASS)
packages[f'{mpl_figure.library_path}.{mpl_figure.name}'] = mpl_figure

# matplotlib # collections
packages['matplotlib.collections'] = Call(name='collections', library_path='matplotlib', call_type=CallType.PACKAGE)

mpl_collect_PathCollection = Call('Figure',
                                  'matplotlib.collections',
                                  {'paths': None,
                                   'sizes': None},  # Note: There is a * and **kwargs
                                  True,
                                  CallType.CLASS)
packages[f'{mpl_collect_PathCollection.library_path}.{mpl_collect_PathCollection.name}'] = mpl_collect_PathCollection

# matplotlib # pyplot
pyplot = Call('pyplot',
              'matplotlib',
              {},
              True,
              CallType.CLASS)
packages[f'{pyplot.library_path}.{pyplot.name}'] = pyplot

pyplot_figure = Call('figure',
                     'matplotlib.pyplot',
                     {'num': None,
                      'figsize': None,
                      'dpi': None,
                      'facecolor': None,
                      'edgecolor': None,
                      'frameon': True,
                      'FigureClass': "<class 'matplotlib.figure.Figure'>",
                      'clear': False},
                     False,
                     CallType.FUNCTION,
                     [mpl_figure])
packages[f'{pyplot_figure.library_path}.{pyplot_figure.name}'] = pyplot_figure

pyplot_scatter = Call('scatter',
                      'matplotlib.pyplot',
                      {'x': None,
                       'y': None,
                       's': None,
                       'c': None,
                       'marker': None,
                       'cmap': None,
                       'norm': None,
                       'vmin': None,
                       'vmax': None,
                       'alpha': None,
                       'linewidths': None,
                       'edgecolors': None,
                       'plotnonfinite': False,
                       'data': None},  # Note: there is a * and **kwargs
                      False,
                      CallType.FUNCTION,
                      [mpl_figure])
packages[f'{pyplot_scatter.library_path}.{pyplot_scatter.name}'] = pyplot_scatter

# matplotlib # axes
packages['matplotlib.collections'] = Call(name='collections', library_path='matplotlib', call_type=CallType.PACKAGE)

# matplotlib # axes # Axes
mpl_axes_Axes = Call('Axes',
                     'matplotlib.axes',
                     {'fig': None,
                      'rect': None,
                      'facecolor': None,
                      'frameon': True,
                      'sharex': None,
                      'sharey': None,
                      'label': '',
                      'xscale': None,
                      'yscale': None,
                      'box_aspect': None},
                     True,
                     CallType.CLASS)
packages[f'{mpl_axes_Axes.library_path}.{mpl_axes_Axes.name}'] = mpl_axes_Axes

# {"uri": "http://kglids.org/resource/library/matplotlib", "contain": [{"uri": "http://kglids.org/resource/library/matplotlib/pyplot", "contain": [{"uri": "http://kglids.org/resource/library/matplotlib/pyplot/figure", "contain": [], "type": "http://kglids.org/ontology/Function"}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/scatter", "contain": [], "type": "http://kglids.org/ontology/Function"}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/pie", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/axis", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/show", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/stackplot", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/imread", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/xticks", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/plot", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/savefig", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/xlim", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/ylim", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/legend", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/boxplot", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/setp", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/axes", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/imshow", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/hist", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/rcParams", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/MaxNLocator", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/FuncFormatter", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/subplots", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/ylabel", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/xlabel", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/subplot", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/suptitle", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/rc_context", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/yticks", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/*", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/title", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/cm", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/colorbar", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/pyplot/barh", "contain": [], "type": null}], "type": "http://kglids.org/ontology/Class"}, {"uri": "http://kglids.org/resource/library/matplotlib/gridspec", "contain": [{"uri": "http://kglids.org/resource/library/matplotlib/gridspec/GridSpec", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/lines", "contain": [{"uri": "http://kglids.org/resource/library/matplotlib/lines/Line2D", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/ticker", "contain": [{"uri": "http://kglids.org/resource/library/matplotlib/ticker/MultipleLocator", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/ticker/MaxNLocator", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/ticker/FuncFormatter", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/ticker/PercentFormatter", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/ticker/NullFormatter", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/ticker/LinearLocator", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/ticker/FormatStrFormatter", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/ticker/ScalarFormatter", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/ticker/StrMethodFormatter", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/ticker/AutoMinorLocator", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/ticker/FixedLocator", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/ticker/FixedFormatter", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/patches", "contain": [{"uri": "http://kglids.org/resource/library/matplotlib/patches/Rectangle", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/patches/Patch", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/patches/Polygon", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/patches/PathPatch", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/patches/FancyArrowPatch", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/patches/Circle", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/patches/RegularPolygon", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/patches/Ellipse", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/patches/Wedge", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/colors", "contain": [{"uri": "http://kglids.org/resource/library/matplotlib/colors/LinearSegmentedColormap", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/colors/ListedColormap", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/colors/DivergingNorm", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/colors/Normalize", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/colors/rgb2hex", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/colors/LogNorm", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/colors/colorConverter", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/rcParams", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/animation", "contain": [{"uri": "http://kglids.org/resource/library/matplotlib/animation/FuncAnimation", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/style", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/cm", "contain": [{"uri": "http://kglids.org/resource/library/matplotlib/cm/ScalarMappable", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/cm/hsv", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/cm/get_cmap", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/offsetbox", "contain": [{"uri": "http://kglids.org/resource/library/matplotlib/offsetbox/AnnotationBbox", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/offsetbox/OffsetImage", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/offsetbox/TextArea", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/offsetbox/DrawingArea", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/offsetbox/AnchoredText", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/font_manager", "contain": [{"uri": "http://kglids.org/resource/library/matplotlib/font_manager/FontProperties", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/dates", "contain": [{"uri": "http://kglids.org/resource/library/matplotlib/dates/DateFormatter", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/dates/date2num", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/dates/num2date", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/dates/MonthLocator", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/dates/AutoDateLocator", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/finance", "contain": [{"uri": "http://kglids.org/resource/library/matplotlib/finance/candlestick_ohlc", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/rc", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/image", "contain": [{"uri": "http://kglids.org/resource/library/matplotlib/image/imread", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/pylab", "contain": [{"uri": "http://kglids.org/resource/library/matplotlib/pylab/rcParams", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/collections", "contain": [{"uri": "http://kglids.org/resource/library/matplotlib/collections/PatchCollection", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/collections/PolyCollection", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/collections/QuadMesh", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/collections/LineCollection", "contain": [], "type": null}], "type": "http://kglids.org/ontology/Package"}, {"uri": "http://kglids.org/resource/library/matplotlib/*", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/projections", "contain": [{"uri": "http://kglids.org/resource/library/matplotlib/projections/register_projection", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/projections/polar", "contain": [{"uri": "http://kglids.org/resource/library/matplotlib/projections/polar/PolarAxes", "contain": [], "type": null}], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/spines", "contain": [{"uri": "http://kglids.org/resource/library/matplotlib/spines/Spine", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/path", "contain": [{"uri": "http://kglids.org/resource/library/matplotlib/path/Path", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/figure", "contain": [{"uri": "http://kglids.org/resource/library/matplotlib/figure/Figure", "contain": [], "type": "http://kglids.org/ontology/Class"}], "type": "http://kglids.org/ontology/Package"}, {"uri": "http://kglids.org/resource/library/matplotlib/mlab", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/text", "contain": [{"uri": "http://kglids.org/resource/library/matplotlib/text/Text", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/text/Annotation", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/cbook", "contain": [{"uri": "http://kglids.org/resource/library/matplotlib/cbook/boxplot_stats", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/cbook/get_sample_data", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/interactive", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/markers", "contain": [{"uri": "http://kglids.org/resource/library/matplotlib/markers/TICKDOWN", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/backends", "contain": [{"uri": "http://kglids.org/resource/library/matplotlib/backends/backend_pdf", "contain": [{"uri": "http://kglids.org/resource/library/matplotlib/backends/backend_pdf/PdfPages", "contain": [], "type": null}], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/legend_handler", "contain": [{"uri": "http://kglids.org/resource/library/matplotlib/legend_handler/HandlerLine2D", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/rc_params_from_file", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/rcParamsDefault", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/transforms", "contain": [{"uri": "http://kglids.org/resource/library/matplotlib/transforms/Affine2D", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/patheffects", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/widgets", "contain": [{"uri": "http://kglids.org/resource/library/matplotlib/widgets/CheckButtons", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/widgets/RadioButtons", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/rcsetup", "contain": [{"uri": "http://kglids.org/resource/library/matplotlib/rcsetup/cycler", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/axes", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/matplotlib/cycler", "contain": [], "type": null}], "type": "http://kglids.org/ontology/Library"},

# ####### PyTorch ##########
packages['torch'] = Call(name='torch', call_type=CallType.LIBRARY)

# torch # Tensor
torch_Tensor = Call('Tensor',
                    'torch',
                    {},
                    True,
                    CallType.CLASS)
packages[f'{torch_Tensor.library_path}.{torch_Tensor.name}'] = torch_Tensor

torch_Tensor_clone = Call('clone',
                          'torch.Tensor',
                          {'memory_format': 'torch.preserve_format'},
                          False,
                          CallType.FUNCTION,
                          [torch_Tensor])
packages[f'{torch_Tensor_clone.library_path}.{torch_Tensor_clone.name}'] = torch_Tensor_clone

# torch # nn
packages['torch.nn'] = Call(name='nn', library_path='torch', call_type=CallType.PACKAGE)

# torch # nn # Flatten
torch_nn_Flatten = Call('Flatten',
                        'torch.nn',
                        {'start_dim': 1,
                         'end_dim': -1},
                        True,
                        CallType.CLASS)
packages[f'{torch_nn_Flatten.library_path}.{torch_nn_Flatten.name}'] = torch_nn_Flatten

# torch # nn # Linear
torch_nn_Linear = Call('Linear',
                       'torch.nn',
                       {'in_features': None,
                        'out_features': None,
                        'bias': True,
                        'device': None,
                        'dtype': None},
                       True,
                       CallType.CLASS)
packages[f'{torch_nn_Linear.library_path}.{torch_nn_Linear.name}'] = torch_nn_Linear

# torch # nn # ReLU
torch_nn_ReLU = Call('ReLU',
                     'torch.nn',
                     {'inplace': False},
                     True,
                     CallType.CLASS)
packages[f'{torch_nn_ReLU.library_path}.{torch_nn_ReLU.name}'] = torch_nn_ReLU

torch_tensor_fct = Call('tensor',
                        'torch',
                        {'data': None,
                         'dtype': None,
                         'device': None,
                         'requires_grad': False,
                         'pin_memory': False},
                        False,
                        CallType.FUNCTION,
                        [torch_Tensor])
packages[f'{torch_tensor_fct.library_path}.{torch_tensor_fct.name}'] = torch_tensor_fct

torch_as_tensor = Call('as_tensor',
                       'torch',
                       {'data': None,
                        'dtype': None,
                        'device': None},
                       False,
                       CallType.FUNCTION,
                       [torch_Tensor])
packages[f'{torch_as_tensor.library_path}.{torch_as_tensor.name}'] = torch_as_tensor

torch_from_numpy = Call('from_numpy',
                        'torch',
                        {'ndarray': None},
                        False,
                        CallType.FUNCTION,
                        [torch_Tensor])
packages[f'{torch_from_numpy.library_path}.{torch_from_numpy.name}'] = torch_from_numpy

torch_linspace = Call('linspace',
                      'torch',
                      {'start': None,
                       'end': None,
                       'steps': None,
                       'out': None,
                       'dtype': None,
                       'layout': 'torch.strided',
                       'device': None,
                       'requires_grad': False},
                      False,
                      CallType.FUNCTION,
                      [torch_Tensor])
packages[f'{torch_linspace.library_path}.{torch_linspace.name}'] = torch_linspace

torch_eye = Call('eye',
                 'torch',
                 {'n': None,
                  'm': None,
                  'out': None,
                  'dtype': None,
                  'layout': 'torch.strided',
                  'device': None,
                  'requires_grad': False},
                 False,
                 CallType.FUNCTION,
                 [torch_Tensor])
packages[f'{torch_eye.library_path}.{torch_eye.name}'] = torch_eye

torch_full = Call('full',
                  'torch',
                  {'size': None,
                   'fill_value': None,
                   'out': None,
                   'dtype': None,
                   'layout': 'torch.strided',
                   'device': None,
                   'requires_grad': False},
                  False,
                  CallType.FUNCTION,
                  [torch_Tensor])
packages[f'{torch_full.library_path}.{torch_full.name}'] = torch_full

torch_cat = Call('cat',
                 'torch',
                 {'tensors': None,
                  'dim': 0,
                  'out': None},
                 False,
                 CallType.FUNCTION,
                 [torch_Tensor])
packages[f'{torch_cat.library_path}.{torch_cat.name}'] = torch_cat

torch_take = Call('take',
                  'torch',
                  {'input': None,
                   'index': None},
                  False,
                  CallType.FUNCTION,
                  [torch_Tensor])
packages[f'{torch_take.library_path}.{torch_take.name}'] = torch_take

torch_unbind = Call('unbind',
                    'torch',
                    {'input': None,
                     'dim': 0},
                    False,
                    CallType.FUNCTION,
                    [tuple()])  # Note: This is a tuple of Tensor
packages[f'{torch_unbind.library_path}.{torch_unbind.name}'] = torch_unbind

# ####### PyTorch ##########
packages['seaborn'] = Call(name='seaborn', call_type=CallType.LIBRARY)

sns_scatterplot = Call('scatterplot',
                       'seaborn',
                       {'x': None,
                        'y': None,
                        'hue': None,
                        'style': None,
                        'size': None,
                        'data': None,
                        'palette': None,
                        'hue_order': None,
                        'hue_norm': None,
                        'sizes': None,
                        'size_order': None,
                        'size_norm': None,
                        'markers': True,
                        'style_order': None,
                        'x_bins': None,
                        'y_bins': None,
                        'units': None,
                        'estimator': None,
                        'ci': 95,
                        'n_boot': 1000,
                        'alpha': None,
                        'x_jitter': None,
                        'y_jitter': None,
                        'legend': 'auto',
                        'ax': None},
                       False,
                       CallType.FUNCTION,
                       [mpl_axes_Axes])
packages[f'{sns_scatterplot.library_path}.{sns_scatterplot.name}'] = sns_scatterplot

sns_lineplot = Call('lineplot',
                    'seaborn',
                    {'x': None,
                     'y': None,
                     'hue': None,
                     'style': None,
                     'size': None,
                     'data': None,
                     'palette': None,
                     'hue_order': None,
                     'hue_norm': None,
                     'sizes': None,
                     'size_order': None,
                     'size_norm': None,
                     'dashes': True,
                     'markers': True,
                     'style_order': None,
                     'units': None,
                     'estimator': 'mean',
                     'ci': 95,  #
                     'n_boot': 1000,
                     'seed': None,
                     'sort': True,
                     'err_style': 'band',
                     'err_kws': None,
                     'legend': 'auto',
                     'ax': None},
                    False,
                    CallType.FUNCTION,
                    [mpl_axes_Axes])
packages[f'{sns_lineplot.library_path}.{sns_lineplot.name}'] = sns_lineplot

sns_histplot = Call('histplot',
                    'seaborn',
                    {'data': None,
                     'x': None,
                     'y': None,
                     'hue': None,
                     'weights': None,
                     'stat': 'count',
                     'bins': 'auto',
                     'binwidth': None,
                     'binrange': None,
                     'discrete': None,
                     'cumulative': False,
                     'common_bins': True,
                     'common_norm': True,
                     'multiple': 'layer',
                     'element': 'bars',
                     'fill': True,
                     'shrink': 1,
                     'kde': False,
                     'kde_kws': None,
                     'line_kws': None,
                     'thresh': 0,
                     'pthresh': None,
                     'pmax': None,
                     'cbar': False,
                     'cbar_ax': None,
                     'cbar_kws': None,
                     'palette': None,
                     'hue_order': None,
                     'hue_norm': None,
                     'color': None,
                     'log_scale': None,
                     'legend': True,
                     'ax': None},
                    False,
                    CallType.FUNCTION,
                    [mpl_axes_Axes])
packages[f'{sns_histplot.library_path}.{sns_histplot.name}'] = sns_histplot

sns_heatmap = Call('heatmap',
                   'seaborn',
                   {'data': None,
                    'vmin': None,
                    'vmax': None,
                    'cmap': None,
                    'center': None,
                    'robust': False,
                    'annot': None,
                    'fmt': '.2g',
                    'annot_kws': None,
                    'linewidths': 0,
                    'linecolor': 'white',
                    'cbar': True,
                    'cbar_kws': None,
                    'cbar_ax': None,
                    'square': False,
                    'xticklabels': 'auto',
                    'yticklabels': 'auto',
                    'mask': None,
                    'ax': None},
                   False,
                   CallType.FUNCTION,
                   [mpl_axes_Axes])
packages[f'{sns_heatmap.library_path}.{sns_heatmap.name}'] = sns_heatmap

# ###### NUMPY #######
packages['numpy'] = Call(name='numpy', call_type=CallType.LIBRARY)

# numpy # ndarray
np_ndarray = Call('ndarray',
                  'numpy',
                  {'shape': None,
                   'dtype': 'float',
                   'buffer': None,
                   'offset': 0,
                   'strides': None,
                   'order': None},
                  True,
                  CallType.CLASS)
packages[f'{np_ndarray.library_path}.{np_ndarray.name}'] = np_ndarray

np_array = Call('array',
                'numpy',
                {'object': None,
                 'dtype': None,
                 'copy': True,
                 'order': 'K',
                 'subok': False,
                 'ndmin': 0,
                 'like': None},
                False,
                CallType.FUNCTION,
                [np_ndarray])
packages[f'{np_array.library_path}.{np_array.name}'] = np_array

np_genfromtxt = Call('genfromtxt',
                     'numpy',
                     {'fname': None,
                      'dtype': "<class 'float'>",
                      'comments': '#',
                      'delimiter': None,
                      'skip_header': 0,
                      'skip_footer': 0,
                      'converters': None,
                      'missing_values': None,
                      'filling_values': None,
                      'usecols': None,
                      'names': None,
                      'excludelist': None,
                      'deletechars': " !#$%&'()*+, -./:;<=>?@[\\]^{|}~",
                      'replace_space': '_',
                      'autostrip': False,
                      'case_sensitive': True,
                      'defaultfmt': 'f%i',
                      'unpack': None,
                      'usemask': False,
                      'loose': True,
                      'invalid_raise': True,
                      'max_rows': None,
                      'encoding': 'bytes',
                      'ndmin': 0,
                      'like': None},
                     False,
                     CallType.FUNCTION,
                     [np_ndarray])
packages[f'{np_genfromtxt.library_path}.{np_genfromtxt.name}'] = np_genfromtxt

# numpy # distutils
packages['numpy.distutils'] = Call(name='distutils', library_path='numpy', call_type=CallType.PACKAGE)

distutils_system = Call('system_info',
                        'numpy.distutils',
                        {},
                        True,
                        CallType.CLASS)
packages[f'{distutils_system.library_path}.{distutils_system.name}'] = distutils_system

# numpy # random
packages['numpy.random'] = Call(name='random', library_path='numpy', call_type=CallType.PACKAGE)

# numpy # random # Generator
np_random_generator = Call('Generator',
                           'numpy.random',
                           {'seed': None},
                           True,
                           CallType.CLASS)
packages[f'{np_random_generator.library_path}.{np_random_generator.name}'] = np_random_generator

np_random_default_rng = Call('default_rng',
                             'numpy.random',
                             {'bit_generator': None},
                             False,
                             CallType.FUNCTION,
                             [np_random_generator])
packages[f'{np_random_default_rng.library_path}.{np_random_default_rng.name}'] = np_random_default_rng

np_random_uniform = Call('uniform',
                         'numpy.random',
                         {'low': 0.0,
                          'high': 1.0,
                          'size': None},
                         False,
                         CallType.FUNCTION,
                         [np_ndarray])
packages[f'{np_random_uniform.library_path}.{np_random_uniform.name}'] = np_random_uniform

np_random_randn = Call('randn',
                       'numpy.random',
                       {'*parameters': None},
                       False,
                       CallType.FUNCTION,
                       [np_ndarray])
packages[f'{np_random_randn.library_path}.{np_random_randn.name}'] = np_random_randn

np_random_seed = Call('seed',
                      'numpy.random',
                      {'seed': None},
                      False,
                      CallType.FUNCTION,
                      [])
packages[f'{np_random_seed.library_path}.{np_random_seed.name}'] = np_random_seed

# numpy # random # RandomState
np_random_RandomState = Call('RandomState',
                             'numpy.random',
                             {'seed': None},
                             True,
                             CallType.CLASS)
packages[f'{np_random_RandomState.library_path}.{np_random_RandomState.name}'] = np_random_RandomState

np_random_rand = Call('rand',
                      'numpy.random',
                      {'*parameters': None},
                      False,
                      CallType.FUNCTION,
                      [np_ndarray])
packages[f'{np_random_rand.library_path}.{np_random_rand.name}'] = np_random_rand

np_random_normal = Call('normal',
                        'numpy.random',
                        {'loc': 0.0,
                         'scale': 1.0,
                         'size': None},
                        False,
                        CallType.FUNCTION,
                        [np_ndarray])
packages[f'{np_random_normal.library_path}.{np_random_normal.name}'] = np_random_normal

np_random_random = Call('random',
                        'numpy.random',
                        {'size': None},
                        False,
                        CallType.FUNCTION,
                        [])
packages[f'{np_random_random.library_path}.{np_random_random.name}'] = np_random_random

np_random_choice = Call('choice',
                        'numpy.random',
                        {'a': None,
                         'size': None,
                         'replace': True,
                         'p': None},
                        False,
                        CallType.FUNCTION,
                        [np_ndarray])
packages[f'{np_random_choice.library_path}.{np_random_choice.name}'] = np_random_choice

np_log = Call('log',
              'numpy',
              {'x': None,
               'out': None,
               'where': True,
               'casting': 'same_kind',
               'order': 'K',
               'dtype': None,
               'subok': True},
              False,
              CallType.FUNCTION,
              [np_ndarray])
packages[f'{np_log.library_path}.{np_log.name}'] = np_log

np_set_printoptions = Call('set_printoptions',
                           'numpy',
                           {'precision': None,
                            'threshold': None,
                            'edgeitems': None,
                            'linewidth': None,
                            'suppress': None,
                            'nanstr': None,
                            'infstr': None,
                            'formatter': None,
                            'sign': None,
                            'floatmode': None,
                            'legacy': None},
                           False,
                           CallType.FUNCTION,
                           [])
packages[f'{np_set_printoptions.library_path}.{np_set_printoptions.name}'] = np_set_printoptions

np_mean = Call('mean',
               'numpy',
               {'a': None,
                'axis': None,
                'dtype': None,
                'out': None,
                'keepdims': '<no value>',
                'where': '<no value>'},
               False,
               CallType.FUNCTION,
               [np_ndarray])
packages[f'{np_mean.library_path}.{np_mean.name}'] = np_mean

np_hstack = Call('hstack',
                 'numpy',
                 {'tup': None},
                 False,
                 CallType.FUNCTION,
                 [np_ndarray])
packages[f'{np_hstack.library_path}.{np_hstack.name}'] = np_hstack

np_percentile = Call('percentile',
                     'numpy',
                     {'a': None,
                      'q': None,
                      'axis': None,
                      'out': None,
                      'overwrite_input': False,
                      'method': 'linear',
                      'keepdims': False,
                      'interpolation': None},
                     False,
                     CallType.FUNCTION,
                     [np_ndarray])
packages[f'{np_percentile.library_path}.{np_percentile.name}'] = np_percentile

# numpy # polynomial
packages['numpy.polynomial'] = Call(name='polynomial', library_path='numpy', call_type=CallType.PACKAGE)

# numpy # polynomial # polynomial
packages['numpy.polynomial.polynomial'] = Call(name='polynomial', library_path='numpy.polynomial',
                                               call_type=CallType.PACKAGE)

np_polynomial_polyfit = Call('polyfit',
                             'numpy.polynomial.polynomial',
                             {'x': None,
                              'y': None,
                              'deg': None,
                              'rcond': None,
                              'full': False,
                              'w': None},
                             False,
                             CallType.FUNCTION,
                             [np_ndarray, None])
packages[f'{np_polynomial_polyfit.library_path}.{np_polynomial_polyfit.name}'] = np_polynomial_polyfit

# [{"contain": [{"uri": "http://kglids.org/resource/library/numpy/arange", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/isnan", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/linalg", "contain": [{"uri": "http://kglids.org/resource/library/numpy/linalg/pinv", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/linalg/inv", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/linalg/*", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/linalg/eig", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/linalg/norm", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/linalg/svd", "contain": [], "type": null}], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/NaN", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/where", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/cov", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/nan", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/log10", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/concatenate", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/median", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/zeros", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/asarray", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/std", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/sort", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/argmax", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/loadtxt", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/absolute", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/fft", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/unique", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/sqrt", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/*", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/interp", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/cos", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/sin", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/abs", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/power", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/matlib", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/tanh", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/polyfit", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/vstack", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/expand_dims", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/corrcoef", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/sum", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/bincount", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/linspace", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/squeeze", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/ma", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/newaxis", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/split", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/pi", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/math", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/dot", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/savetxt", "contain": [], "type": null}, {"uri": "http://kglids.org/resource/library/numpy/var", "contain": [], "type": null}], "type": null},
