import pandas as pd
import gzip
import datetime
import os
import shutil

def compress_gz(f_path):
    f_in = open(f_path, 'rb')
    f_out = gzip.open(f_path + ".gz", 'wb')
    f_out.writelines(f_in)
    f_out.close()
    f_in.close()


def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)

def define_rel_types(g_tsv_df):
    g_tsv_df["p"]

def triplet_encoding(dataset_name_script,label_node):
    dataset_name = dataset_name_script+"_gnn_"+label_node
    dataset_name_csv = "OnDemandDataPrep/storage/output_file_"+dataset_name_script+".csv"
    dataset_types = "OnDemandDataPrep/storage/output_type_"+dataset_name_script+".csv"
    split_by = {"folder_name": "random"}
    target_rel="http://kglids.org/ontology/pipeline/HasCleaningOperation"
    dic_results = {}
    Literals2Nodes = True
    output_root_path = "OnDemandDataPrep/storage/"
    g_tsv_df = pd.read_csv(dataset_name_csv,encoding_errors='ignore',sep=",")
    g_tsv_types_df = pd.read_csv(dataset_types, encoding_errors='ignore')
    g_tsv_df = g_tsv_df.dropna()
    g_tsv_types_df = g_tsv_types_df.dropna()

    try:
        g_tsv_df = g_tsv_df.rename(columns={"Subject": "s", "Predicate": "p", "Object": "o"})
        g_tsv_df = g_tsv_df.rename(columns={0: "s", 1: "p", 2: "o"})
        ######################## Remove Litreal Edges####################
        Literal_edges_lst = []
        g_tsv_df = g_tsv_df[~g_tsv_df["p"].isin(Literal_edges_lst)]
        g_tsv_df = g_tsv_df.drop_duplicates()
        g_tsv_df = g_tsv_df.dropna()
    except:
        print("g_tsv_df columns=", g_tsv_df.columns())
    unique_p_lst = g_tsv_df["p"].unique().tolist()
    ########################delete non target nodes #####################
    relations_lst = g_tsv_df["p"].unique().astype("str").tolist()
    dic_results[dataset_name] = {}
    dic_results[dataset_name]["usecase"] = dataset_name
    dic_results[dataset_name]["TriplesCount"] = len(g_tsv_df)
    
    #################### Remove Split and Target Rel ############
    if target_rel in relations_lst:
        relations_lst.remove(target_rel)
    ################################write relations index ########################
    relations_df = pd.DataFrame(relations_lst, columns=["rel name"])
    relations_df["rel name"] = relations_df["rel name"].apply(lambda x: str(x).split("/")[-1])
    relations_df["rel idx"] = relations_df.index
    relations_df = relations_df[["rel idx", "rel name"]]
    map_folder = output_root_path + dataset_name + "/mapping"
    try:
        os.stat(map_folder)
    except:
        os.makedirs(map_folder)
    relations_df.to_csv(map_folder + "/relidx2relname.csv", index=None)
    compress_gz(map_folder + "/relidx2relname.csv")
    ############################### create target label index ########################
    label_idx_df = pd.DataFrame(g_tsv_df[g_tsv_df["p"] == target_rel]["o"].apply(lambda x: str(x).strip()).unique().tolist(),columns=["label name"])
    dic_results[dataset_name]["ClassesCount"] = len(label_idx_df)
    try:
        label_idx_df["label name"] = label_idx_df["label name"].astype("int64")
        label_idx_df = label_idx_df.sort_values(by=["label name"]).reset_index(drop=True)
    except:
        label_idx_df["label name"] = label_idx_df["label name"].astype("str")
        label_idx_df = label_idx_df.sort_values(by=["label name"]).reset_index(drop=True)

    label_idx_df["label idx"] = label_idx_df.index
    label_idx_df = label_idx_df[["label idx", "label name"]]
    label_idx_df.to_csv(map_folder + "/labelidx2labelname.csv", index=None)
    compress_gz(map_folder + "/labelidx2labelname.csv")
    ###########################################prepare relations mapping#################################
    relations_entites_map = {}
    relations_dic = {}
    entites_dic = {}
    for rel in relations_lst:
        rel_type = rel
        rel_df = g_tsv_df[g_tsv_df["p"] == rel].reset_index(drop=True)
        rel_types = g_tsv_types_df[g_tsv_types_df['ptype'].isin([rel])]
        s_type = rel_types['stype'].values[0]
        o_type = rel_types['otype'].values[0]
        rel_df["s_type"] = s_type
        rel_df["o_type"]=o_type
        #########################################################################################
        rel_entity_types = rel_df[["s_type", "o_type"]].drop_duplicates()
        list_rel_types = []
        for idx, row in rel_entity_types.iterrows():
            list_rel_types.append((row["s_type"], rel, row["o_type"]))

        relations_entites_map[rel] = list_rel_types
        if len(list_rel_types) > 2:
            print(len(list_rel_types))
        relations_dic[rel] = rel_df
        for rel_pair in list_rel_types:
            e1, rel, e2 = rel_pair
            if e1 != "literal" and e1 in entites_dic:
                entites_dic[e1] = entites_dic[e1].union(
                    set(rel_df[rel_df["s_type"] == e1]["s"].apply(
                        lambda x: str(x).split("/")[-1]).unique()))
            elif e1 != "literal":
                entites_dic[e1] = set(rel_df[rel_df["s_type"] == e1]["s"].apply(
                    lambda x: str(x).split("/")[-1]).unique())

            if e2 != "literal" and e2 in entites_dic:
                entites_dic[e2] = entites_dic[e2].union(
                    set(rel_df[rel_df["o_type"] == e2]["o"].apply(
                        lambda x: str(x).split("/")[-1]).unique()))
            elif e2 != "literal":
                entites_dic[e2] = set(rel_df[rel_df["o_type"] == e2]["o"].apply(
                    lambda x: str(x).split("/")[-1]).unique())
    ############################### Make sure all rec papers have target ###########
    target_subjects_lst = g_tsv_df[g_tsv_df["p"] == target_rel]["s"].apply(
        lambda x: str(x).split("/")[-1]).unique().tolist()
    ############################ write entites index #################################
    for key in list(entites_dic.keys()):
        entites_dic[key] = pd.DataFrame(list(entites_dic[key]), columns=['ent name']).astype(
            'str').sort_values(by="ent name").reset_index(drop=True)
        entites_dic[key] = entites_dic[key].drop_duplicates()
        entites_dic[key]["ent idx"] = entites_dic[key].index
        entites_dic[key] = entites_dic[key][["ent idx", "ent name"]]
        entites_dic[str(key) + "_dic"] = pd.Series(entites_dic[key]["ent idx"].values,
                                              index=entites_dic[key]["ent name"]).to_dict()
        map_folder = output_root_path + dataset_name + "/mapping"
        try:
            os.stat(map_folder)
        except:
            os.makedirs(map_folder)
        entites_dic[str(key)].to_csv(map_folder + "/" + key + "_entidx2name.csv", index=None)
        compress_gz(map_folder + "/" + key + "_entidx2name.csv")
    #################### write nodes statistics ######################
    lst_node_has_feat = [
        list(
            filter(lambda entity: str(entity).endswith("_dic") == False, list(entites_dic.keys())))]
    lst_node_has_label = lst_node_has_feat.copy()
    lst_num_node_dict = lst_node_has_feat.copy()
    lst_has_feat = []
    lst_has_label = []
    lst_num_node = []

    for entity in lst_node_has_feat[0]:
        if str(entity) == str(label_node):
            lst_has_label.append("True")
            lst_has_feat.append("True")
        else:
            lst_has_label.append("False")
            lst_has_feat.append("False")

        lst_num_node.append(len(entites_dic[entity + "_dic"]))

    lst_node_has_feat.append(lst_has_feat)
    lst_node_has_label.append(lst_has_label)
    lst_num_node_dict.append(lst_num_node)

    lst_relations = []

    for key in list(relations_entites_map.keys()):
        for elem in relations_entites_map[key]:
            (e1, rel, e2) = elem
            lst_relations.append([e1, str(rel).split("/")[-1], e2])

    map_folder = output_root_path + dataset_name + "/raw"

    try:
        os.stat(map_folder)
    except:
        os.makedirs(map_folder)

    pd.DataFrame(lst_node_has_feat).to_csv(
        output_root_path + dataset_name + "/raw/nodetype-has-feat.csv", header=None,
        index=None)
    compress_gz(output_root_path + dataset_name + "/raw/nodetype-has-feat.csv")

    pd.DataFrame(lst_node_has_label).to_csv(
        output_root_path + dataset_name + "/raw/nodetype-has-label.csv",
        header=None, index=None)
    compress_gz(output_root_path + dataset_name + "/raw/nodetype-has-label.csv")

    pd.DataFrame(lst_num_node_dict).to_csv(
        output_root_path + dataset_name + "/raw/num-node-dict.csv", header=None,
        index=None)
    compress_gz(output_root_path + dataset_name + "/raw/num-node-dict.csv")

    ############################### create label relation index  ######################
    label_idx_df["label idx"] = label_idx_df["label idx"].astype("int64")
    label_idx_df["label name"] = label_idx_df["label name"].apply(lambda x: str(x).split("/")[-1])
    label_idx_dic = pd.Series(label_idx_df["label idx"].values, index=label_idx_df["label name"]).to_dict()
    ############ drop multiple targets per subject keep first#######################
    labels_rel_df = g_tsv_df[g_tsv_df["p"] == target_rel].reset_index(drop=True)
    labels_rel_df = labels_rel_df.sort_values(['s', 'o'], ascending=[True, True])
    labels_rel_df = labels_rel_df.drop_duplicates(subset=["s"], keep='first')
    ###############################################################################
    
    
    #####################DELETING NODE LABELS AS THERE ARE NONE#####################
    
    rel_type = target_rel.split("/")[-1]
    rel_types = g_tsv_types_df[g_tsv_types_df['ptype'].isin([target_rel])]
    try:
        s_type = rel_types['stype'].values[0]
        o_type = rel_types['otype'].values[0]

    except IndexError:
        s_type = None
        o_type = None
        print("Empty 'rel_types' dataframe")
    
    s_label_type = label_node
    o_label_type = o_type
    label_type = label_node
    
    if s_type is not None:
        labels_rel_df["s_idx"] = labels_rel_df["s"].apply(lambda x: str(x).split("/")[-1])
        labels_rel_df["s_idx"] = labels_rel_df["s_idx"].astype("str")
        labels_rel_df["s_idx"] = labels_rel_df["s_idx"].apply(lambda x: entites_dic[s_label_type + "_dic"][x] if x in entites_dic[s_label_type + "_dic"].keys() else -1)

        labels_rel_df_notfound = labels_rel_df[labels_rel_df["s_idx"] == -1]
        labels_rel_df = labels_rel_df[labels_rel_df["s_idx"] != -1]
        labels_rel_df = labels_rel_df.sort_values(by=["s_idx"]).reset_index(drop=True)
        labels_rel_df["o_idx"] = labels_rel_df["o"].apply(lambda x: str(x).split("/")[-1])
        labels_rel_df["o_idx"] = labels_rel_df["o_idx"].apply(lambda x: label_idx_dic[str(x)] if str(x) in label_idx_dic.keys() else -1)
        out_labels_df = labels_rel_df[["o_idx"]]
    else:
        out_labels_df = pd.DataFrame(columns=["o_idx"])
        out_labels_df.loc[0] = 0
    
    map_folder = output_root_path + dataset_name + "/raw/node-label/" + s_label_type
    try:
        os.stat(map_folder)
    except:
        os.makedirs(map_folder)
    
    out_labels_df.to_csv(map_folder + "/node-label.csv", header=None, index=None)
    compress_gz(map_folder + "/node-label.csv")

    ###################### write entites relations for nodes only (non literals) #########################
    idx = 0
    for rel in relations_dic:
        for rel_list in relations_entites_map[rel]:
            e1, rel, e2 = rel_list
            relations_dic[rel]["s_idx"] = relations_dic[rel]["s"].apply(
                lambda x: str(x).split("/")[-1])
            relations_dic[rel]["s_idx"] = relations_dic[rel]["s_idx"].apply(
                lambda x: entites_dic[e1 + "_dic"][x] if x in entites_dic[
                    e1 + "_dic"].keys() else -1)
            relations_dic[rel] = relations_dic[rel][relations_dic[rel]["s_idx"] != -1]
            relations_dic[rel]["o_idx"] = relations_dic[rel]["o"].apply(
                lambda x: str(x).split("/")[-1])
            relations_dic[rel]["o_idx"] = relations_dic[rel]["o_idx"].apply(
                lambda x: entites_dic[e2 + "_dic"][x] if x in entites_dic[
                    e2 + "_dic"].keys() else -1)
            relations_dic[rel] = relations_dic[rel][relations_dic[rel]["o_idx"] != -1]
            relations_dic[rel] = relations_dic[rel].sort_values(by="s_idx").reset_index(drop=True)
            rel_out = relations_dic[rel][["s_idx", "o_idx"]]
            if len(rel_out) > 0:
                map_folder = output_root_path + dataset_name + "/raw/relations/" + e1 + "___" + \
                             rel.split("/")[-1] + "___" + e2
                try:
                    os.stat(map_folder)
                except:
                    os.makedirs(map_folder)
                rel_out.to_csv(map_folder + "/edge.csv", index=None, header=None)
                compress_gz(map_folder + "/edge.csv")
                ########## write relations num #################
                f = open(map_folder + "/num-edge-list.csv", "w")
                f.write(str(len(relations_dic[rel])))
                f.close()
                compress_gz(map_folder + "/num-edge-list.csv")
                ##################### write relations idx #######################
                rel_idx = \
                    relations_df[relations_df["rel name"] == rel.split("/")[-1]]["rel idx"].values[0]
                rel_out["rel_idx"] = rel_idx
                rel_idx_df = rel_out["rel_idx"]
                rel_idx_df.to_csv(map_folder + "/edge_reltype.csv", header=None, index=None)
                compress_gz(map_folder + "/edge_reltype.csv")
            else:
                lst_relations.remove([e1, str(rel).split("/")[-1], e2])

            pd.DataFrame(lst_relations).to_csv(output_root_path + dataset_name + "/raw/triplet-type-list.csv", header=None, index=None)
            compress_gz(output_root_path + dataset_name + "/raw/triplet-type-list.csv")
            #####################Zip Folder ###############
        shutil.make_archive(output_root_path + dataset_name, 'zip',
                            root_dir=output_root_path, base_dir=dataset_name)
    end_t = datetime.datetime.now()
    pd.DataFrame(dic_results).transpose().to_csv(
        output_root_path + dataset_name.split(".")[0]+"_ToHetroG_times.csv", index=False)
    print('Encoding Completed!')
        
