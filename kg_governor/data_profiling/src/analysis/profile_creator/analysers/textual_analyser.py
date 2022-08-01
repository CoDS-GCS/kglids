import re
from dateutil import parser as date_parser
from dateutil.parser import *
from analysis.profile_creator.analysers.i_analyser import IAnalyser
from datasketch import MinHash
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, sum, collect_list, udf
from allennlp.predictors.predictor import Predictor
from pyspark.sql.types import *

#print("\nloading NER model ({})\n".format("https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz"))
elmo_ner = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz")

class TextualAnalyser(IAnalyser):

    def __init__(self, df: DataFrame):
        self.profiles_info = {}
        self.df = df

    def get_profiles_info(self):
        return self.profiles_info

    def analyse_columns(self):
        self.profiles_info = {}
        columns = self.df.columns
        if not columns:
            return
        minhash_distinct_values_count_dict = self.__get_minhash_distinct_values_counts(columns)
        missing_values_dict = self.__get_missing_values(self.df.columns)
        # print('missing value dict: ', missing_values_dict)
        string_type_dict = self.__get_string_subcategory(self.df.columns)
        # print('string type: ', string_type_dict)
        for column in columns:
            info = minhash_distinct_values_count_dict[column]
            info.update({'missing_values_count': missing_values_dict[column]})
            info.update({'string_subtype': string_type_dict[column]})
            self.profiles_info[column] = info

    def __get_minhash_distinct_values_counts(self, columns: list) -> dict:
        def compute_minhash(l):
            m = MinHash(num_perm=512)
            for v in l:
                if isinstance(v, str):
                    m.update(v.lower().encode('utf8'))
            return m.hashvalues.tolist(), len(l), len(set(l))

        if not columns:
            return columns

        '''profiles_info = self.df.rdd \
            .map(lambda row: row.asDict()) \
            .flatMap(lambda d: [(c, d[c]) for c in columns]) \
            .groupByKey() \
            .map(lambda column: {column[0]:
                                     {'minhash': compute_minhash(column[1]), 'count': len(column[1]),
                                      'distinct_values_count': len(set(column[1]))}}) \
            .reduce(lambda x, y: {**x, **y})
        return profiles_info'''
        schema = StructType([StructField('minhash', ArrayType(IntegerType()), False),
                             StructField('count', IntegerType(), False),
                             StructField('distinct', IntegerType(), False)])
        minhashUDF = udf(lambda z: compute_minhash(z), schema)
        # minhashUDF = udf(lambda z: compute_minhash(z))
        cols = self.df.columns
        cols2 = ['`' + c + '`' for c in cols]
        df2 = self.df.select([collect_list(c) for c in cols2]).toDF(*cols2)
        df2 = df2.toDF(*cols)
        for col in cols:
            df2 = df2.withColumn(col, minhashUDF('`' + col + '`'))
        profiles_info = {}
        d = df2.toPandas().to_dict()
        for c in cols:
            # col_rdd = df.select('`' + c + '`').rdd
            profiles_info[c] = {'minhash': d[c][0]['minhash'], 'count': d[c][0]['count'],
                                'distinct_values_count': d[c][0]['distinct']}
        return profiles_info

    def __get_missing_values(self, columns: list) -> dict:
        return self.df.select(*(sum(col('`' + c + '`').isNull().cast("int")).alias(c) for c in columns)) \
            .rdd \
            .map(lambda x: x.asDict()) \
            .collect()[0]

    def __get_string_subcategory(self, columns: list):

        if not columns:
            return columns
        temp_df = self.df
        original_cols = temp_df.columns
        names = list(range(0, len(original_cols)))
        names = list(map(lambda x: 'c_' + str(x), names))  # rename cols to perform drop
        temp_df = temp_df.toDF(*names)
        temp_df = temp_df.na.drop().limit(10)  # dataframe with 10 samples *******

        def get_string_type(samples):
            sample_size = 10
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            us_code_pattern = '/(^\d{5}(?:[\s]?[-\s][\s]?\d{4})?$)/'
            ca_code_pattern = '/^[ABCEGHJ-NPRSTVXY]\d[ABCEGHJ-NPRSTV-Z][ -]?\d[ABCEGHJ-NPRSTV-Z]\d$/i'
            custom_pattern = '[a-zA-Z]+[0-9]+[a-zA-Z]*[0-9]*'
            

            def check_nlp(person = 0, org = 0, loc = 0):
                
                def check_person(tags):
                    if len(tags) > 3:
                        return False
                    for tag in tags:
                        if 'PERSON' in tag or 'PER' in tag or 'L-PER' in tag or 'B-PER' in tag:
                            return True
                    return False

                def check_location(tags):
                    for tag in tags:
                        if 'GPE' in tag or 'FAC' in tag or 'LOC' in tag or 'U-LOC' in tag or 'U-GPE' in tag or 'U-FAC' in tag:
                            return True
                    return False
                
                def check_org(tags):
                    for tag in tags:
                        if 'ORG' in tag or 'U-ORG' in tag:
                            return True
                    return False
                
                for s in samples:
                    if isinstance(s, str):
                        s = s.replace('\n', '')
                        s = s.replace('\t', '')
                        s = s.replace('\f', '')
                        s = s.replace('\r', '') 
                        try:
                            tags = elmo_ner.predict(sentence=s)['tags'] 
                            nlp_status = False
                            if check_location(tags):
                                loc = loc + 1
                                nlp_status = True
                            if check_person(tags):
                                person = person + 1
                                nlp_status = True
                            if check_org(tags):
                                org = org + 1
                                nlp_status = True
                            if not nlp_status:
                                break
                        except IndexError as error:
                            break

                if person == sample_size:
                    return "T_person"
                if loc == sample_size:
                    return "T_loc"
                if org == sample_size:
                    return "T_org"
                return False         
            
            def check_date(n=0):
                for s in samples:
                    if isinstance(s, str):
                        s = s.replace('\n', '')
                        s = s.replace('\t', '')
                        s = s.replace('\f', '')
                        s = s.replace('\r', '')
                        try:
                            if date_parser.parse(s):
                                n = n + 1
                                continue
                            else:
                                break
                        except (ParserError, UnknownTimezoneWarning, TypeError, OverflowError) as e:
                            break
                if n == sample_size:
                    return True

            def check_code(n=0):
                for s in samples:
                    if isinstance(s, str):
                        s = s.replace('\n', '')
                        s = s.replace('\t', '')
                        s = s.replace('\f', '')
                        s = s.replace('\r', '')
                        if re.search(us_code_pattern, s) or re.search(ca_code_pattern, s) or re.search(custom_pattern,
                                                                                                       s):
                            n = n + 1
                        else:
                            break
                    if n == sample_size:
                        return True

            def check_email(n=0):
                for s in samples:
                    if isinstance(s, str):
                        s = s.replace('\n', '')
                        s = s.replace('\t', '')
                        s = s.replace('\f', '')
                        s = s.replace('\r', '')
                        if re.fullmatch(email_pattern, s):
                            n = n + 1
                        else:
                            break
                if n == sample_size:
                    return True

            if check_code():
                return 'T_code'

            if check_email():
                return 'T_email'

            if check_date():
                return 'T_date'

            nlp = check_nlp()
            #print("nlp: ", nlp)

            if check_nlp():
                return nlp

            # if check_location():
            #     return 'T_loc'
            
            # if check_person():
            #     return 'T_person'

            # if check_organization():
            #     return 'T_org'

            return 'T'

        d = {}
        for i in range(0, len(temp_df.columns)):
            # print("Analyzing column: ", temp_df.columns[i], "original name: ", original_cols[i])
            # print("1. getting values: ")
            col_values = [(row[temp_df.columns[i]]) for row in temp_df.select(temp_df.columns[i]).collect()]
            # print(col_values)
            # print("done.", end ="")
            # print("2. getting type: ")
            d[original_cols[i]] = get_string_type(col_values)
            # print("done.", end = "")

        return d
