import rdflib
import pandas as pd
import csv


def profile_to_csv(graph_name: str='cleaning_graph'):
    graph = rdflib.Graph()
    ttl_file = '../../storage/knowledge_graph/'+ graph_name + '.ttl'
    graph.parse(ttl_file, format="turtle")

    csv_file = 'storage/output' + graph_name + '.csv'

    with open(csv_file, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)

        header = ["Subject", "Predicate", "Object"]
        csv_writer.writerow(header)

        for subject, predicate, obj in graph:
            subject_str = str(subject)
            predicate_str = str(predicate)
            obj_str = str(obj)

            csv_writer.writerow([subject_str, predicate_str, obj_str])

    print(f"Transformation complete. Output saved to {csv_file}")



def get_subject_type(subject):
    parts = subject.split("/")
    if len(parts) == 5:
        return "Source"
    elif len(parts) == 6:
        return "Dataset"
    elif len(parts) == 7:
        return "2Table"
    elif len(parts) == 8:
        return "1Column"
    else:
        return 'NA'

def get_object_type(obj, subject, cleaning_op_list):
    if is_number(obj):
        return "Value"
    elif obj=='http://kglids.org/ontology/Column' or obj=='http://kglids.org/ontology/Source' or obj=='http://kglids.org/ontology/Dataset' or obj=='http://kglids.org/ontology/Table': #http://kglids.org/ontology/Table is considered dataset
        return "ontology"
    elif obj in cleaning_op_list:
        return "CleaningOperation"
    elif len(obj.split("/")) == 6:
        return "Dataset"
    elif '/mnt/Niki' in obj:
        return "Path"
    elif len(obj.split("/")) == 7:
        return "2Table"
    elif len(obj.split("/")) == 8:
        return "1Column"
    elif obj in ['int', 'float', 'natural_language_text', 'named_entity', 'date', 'boolean', 'string']:
        return "datatype"
    elif len(subject.split("/"))==7:
        return 'table-name'
    elif len(subject.split("/"))==8:
        return 'column-name'
    else:
        return 'name'

def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def get_pred(pred,subject):
    if pred=='http://www.w3.org/2000/01/rdf-schema#label':
      if len(subject.split("/"))==7:
        return 'http://www.w3.org/2000/01/rdf-schema#label-table'
      elif len(subject.split("/"))==8:
        return 'http://www.w3.org/2000/01/rdf-schema#label-column'
    if pred=='http://schema.org/name':
      if len(subject.split("/"))==7:
        return 'http://schema.org/name-table'
      elif len(subject.split("/"))==8:
        return 'http://schema.org/name-column'
    if pred=='http://www.w3.org/1999/02/22-rdf-syntax-ns#type':
      if len(subject.split("/"))==7:
        return 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type-table'
      elif len(subject.split("/"))==8:
        return 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type-column'
    if pred=='http://kglids.org/ontology/isPartOf':
      if len(subject.split("/"))==7:
        return 'http://kglids.org/ontology/isPartOf-table'
      elif len(subject.split("/"))==8:
        return 'http://kglids.org/ontology/isPartOf-column'
    else:
      return pred

def create_encoding_file(graph_name: str='cleaning_graph', operation: str='cleaning'):

    df = pd.read_csv('OnDemandDataPrep/Modeling/storage/output_' + graph_name + '.csv', quotechar='"')#"output_cleaning_w_extras_profiles_comp.csv"
    op_list = {'cleaning':['http://kglids.org/resource/library/pandas/DataFrame/fillna','http://kglids.org/resource/library/pandas/DataFrame/dropna','http://kglids.org/resource/library/sklearn/impute/SimpleImputer','http://kglids.org/resource/library/sklearn/impute/KNNImputer','http://kglids.org/resource/library/sklearn/impute/IterativeImputer','http://kglids.org/resource/library/pandas/DataFrame/interpolate'],
               'unary_transformation':['sqrt','log'],
               'scaler_transformation':['MinMaxScaler','RobustScaler','StandardScaler']
               }
    df_type_out = pd.DataFrame()
    df_type_out["stype"] = df["Subject"].apply(get_subject_type)
    df_type_out["ptype"] = df.apply(lambda row: get_pred(row['Predicate'], row['Subject']), axis=1)
    df_type_out['otype'] = df.apply(lambda row: get_object_type(row['Object'], row['Subject'], op_list[operation]), axis=1)

    df_out = pd.DataFrame()
    df_out["Subject"] = df["Subject"].apply(lambda subject:
                    '/'.join(subject.split('/')[:-1])+'/'+'--'.join(subject.split('/')[-2:])
                    if len(subject.split('/')) == 7
                    else '/'.join(subject.split('/')[:-2])+'/'+'--'.join(subject.split('/')[-3:-1])+'/'+'--'.join(subject.split('/')[-3:])
                    if len(subject.split('/')) == 8
                    else subject)
    df_out["Predicate"] = df.apply(lambda row: get_pred(row['Predicate'], row['Subject']), axis=1)
    df_out["Object"] = df["Object"].apply(lambda obj: obj if obj in op_list[operation]
                                          else '/'.join(obj.split('/')[:-1]) + '/' + '--'.join(obj.split('/')[-2:])
                                         if isinstance(obj, str) and len(obj.split('/')) == 7
                                         else '/'.join(obj.split('/')[:-2]) + '/' + '--'.join(obj.split('/')[-3:-1]) + '/' + '--'.join(obj.split('/')[-3:])
                                         if isinstance(obj, str) and len(obj.split('/')) == 8
                                         else obj)

    df_type_out.to_csv("OnDemandDataPrep/Modeling/storage/output_" + graph_name + "_type.csv", index=False)
    df_out.to_csv("OnDemandDataPrep/Modeling/storage/output_" + graph_name + "_file.csv", index=False)
    print('Encoding files are ready!')