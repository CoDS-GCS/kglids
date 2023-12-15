from gnn_applications.OnDemandDataPrep.calculate_embeddings_kglids import EmbeddingCreator
from gnn_applications.OnDemandDataPrep.dataset_pyg_hsh import PygNodePropPredDataset_hsh
from torch.nn import ModuleList, Linear, ParameterDict, Parameter
from torch_sparse import SparseTensor
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.nn import MessagePassing
import torch
import torch.nn.functional as F

import pandas as pd
import numpy as np


class RGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_node_types,
                 num_edge_types):
        super(RGCNConv, self).__init__(aggr='mean')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        self.rel_lins = ModuleList([
            Linear(in_channels, out_channels, bias=False)
            for _ in range(num_edge_types)
        ])

        self.root_lins = ModuleList([
            Linear(in_channels, out_channels, bias=True)
            for _ in range(num_node_types)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.rel_lins:
            lin.reset_parameters()
        for lin in self.root_lins:
            lin.reset_parameters()

def get_column_emb(dataset_name_script,df_train):
    recommender = EmbeddingCreator()
    entity_df = df_train
    df = recommender.new_emb(entity_df, 'scaling-column')
    file1 = pd.read_csv('OnDemandDataPrep/storage/'+dataset_name_script+'_gnn_Table/mapping/Column_entidx2name.csv')
    file1['ent name'] = file1['ent name'].astype(str)

    # Perform a left join on 'Table' column while preserving the order of file1
    df_emb = pd.merge(file1, df, left_on='ent name', right_index=True, how='left')
    df_emb = df_emb.drop('ent name', axis=1)
    df_emb = df_emb.drop('ent idx', axis=1)
    df_emb = df_emb.reset_index(drop=True)
    df_emb = df_emb.fillna(0)
    array_emb = df_emb.values
    column_emb = torch.from_numpy(array_emb.astype(np.float32))
    return column_emb
    
    
class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_nodes_dict, x_types, num_edge_types):
        super(RGCN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        node_types = list(num_nodes_dict.keys())
        num_node_types = len(node_types)

        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        self.emb_dict = ParameterDict({
            f'{key}': Parameter(torch.Tensor(num_nodes_dict[key], in_channels))
            for key in set(node_types).difference(set(x_types))
        })
        I, H, O = in_channels, hidden_channels, out_channels  # noqa

        # Create `num_layers` many message passing layers.
        self.convs = ModuleList()
        self.convs.append(RGCNConv(I, O, self.num_node_types, num_edge_types))

        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.emb_dict.values():
            torch.nn.init.xavier_uniform_(emb)
        for conv in self.convs:
            conv.reset_parameters()


    def inference(self, dataset_name_script, df_train, x_dict, edge_index_dict, key2int):
        device = list(x_dict.values())[0].device
        for key, emb in self.emb_dict.items():#ADDED
            #overwrite column embeddings (0)
            if int(key)==0:
              x_dict[int(key)] = get_column_emb(dataset_name_script,df_train)
            if int(key)==1:
              x_dict[int(key)] = self.emb_dict[str(1)] #switch the encoding
            else:
              x_dict[int(key)] = emb
        del self.emb_dict

        adj_t_dict = {}
        for key, (row, col) in edge_index_dict.items():
            adj_t_dict[key] = SparseTensor(row=col, col=row).to(device)


        for i, conv in enumerate(self.convs):
            out_dict = {}
            for j, x in x_dict.items():
                out_dict[j] = conv.root_lins[j](x)
            for keys, adj_t in adj_t_dict.items():
                src_key, target_key = keys[0], keys[-1]
                out = out_dict[key2int[target_key]]
                tmp = adj_t.matmul(x_dict[key2int[src_key]], reduce='mean')
                ################## fill missed rows hsh############################
                tmp = conv.rel_lins[key2int[keys]](tmp).resize_([out.size()[0], out.size()[1]])
                out.add_(tmp)

            del adj_t_dict
            if i != self.num_layers - 1:
                for j in range(self.num_node_types):
                    F.relu_(out_dict[j])

        return out_dict


def graphSaint(dataset_name_script,df_train):
    to_remove_pedicates = []
    to_remove_subject_object = []
    GA_dataset_name = dataset_name_script + "_gnn_Table"
    gsaint_Final_Test = 0
    dataset = PygNodePropPredDataset_hsh(name=GA_dataset_name,
                                         root='OnDemandDataPrep/storage/',
                                         numofClasses=str(3))

    data = dataset[0]
    del dataset
    for key, tensor in data.edge_reltype.items():
        data.edge_reltype[key] = np.where(tensor == 0,9, tensor) #Change based on encoding
    global subject_node
    subject_node = list(data.y_dict.keys())[0]
    data.node_year_dict = None
    data.edge_reltype_dict = None

    to_remove_rels = []
    for keys, (row, col) in data.edge_index_dict.items():
        if (keys[2] in to_remove_subject_object) or (keys[0] in to_remove_subject_object):
            to_remove_rels.append(keys)

    for keys, (row, col) in data.edge_index_dict.items():
        if (keys[1] in to_remove_pedicates):
            to_remove_rels.append(keys)
            to_remove_rels.append((keys[2], '_inv_' + keys[1], keys[0]))

    for elem in to_remove_rels:
        data.edge_index_dict.pop(elem, None)
        data.edge_reltype.pop(elem, None)

    for key in to_remove_subject_object:
        data.num_nodes_dict.pop(key, None)

    ##############add inverse edges ###################
    edge_index_dict = data.edge_index_dict
    key_lst = list(edge_index_dict.keys())
    for key in key_lst:
        r, c = edge_index_dict[(key[0], key[1], key[2])]
        edge_index_dict[(key[2], 'inv_' + key[1], key[0])] = torch.stack([c, r])

    out = group_hetero_graph(edge_index_dict, data.num_nodes_dict)
    edge_index, _, _, _, _, key2int = out

    recommender = EmbeddingCreator()
    entity_df = df_train
    df_emb = np.array(recommender.new_emb(entity_df, 'scaling-table'))

    if df_emb.size == 0:
        feat = torch.Tensor(data.num_nodes_dict[subject_node], 300)
        torch.nn.init.xavier_uniform_(feat)
    else:
        feat = torch.from_numpy(df_emb.astype(np.float32)).unsqueeze(0)

    feat_dic = {subject_node: feat}
    del df_emb
    del feat
    x_dict = {}
    for key, x in feat_dic.items():
        x_dict[key2int[key]] = x

    model = RGCN(1800, 32, 3, 1, 0.5, {0: 16062, 1: 179, 2: 619, 3: 15820, 4: 7, 5: 16202, 6: 4}, [1], 44)

    loadTrainedModel = 1
    if loadTrainedModel == 1:
        with torch.no_grad():
            model.eval()
            model.load_state_dict(torch.load(
                "OnDemandDataPrep/Models/kgfarm_gnn_GA_0_DBLP_conf_GSAINT_QM_e30_r3_s30_l1_hc128_w4_lr10_drop5_new_emb_1800.model"))
            out = model.inference(dataset_name_script,df_train, x_dict, edge_index_dict, key2int)
            out = out[key2int[subject_node]]
            y_pred = out.argmax(dim=-1, keepdim=True).cpu()
            pred_lst = torch.flatten(y_pred).tolist()
            gsaint_Final_Test = pred_lst[0]

    return gsaint_Final_Test






