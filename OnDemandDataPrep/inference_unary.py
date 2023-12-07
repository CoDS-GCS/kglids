from copy import copy
import datetime
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ParameterDict, Parameter
from torch_sparse import SparseTensor
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.nn import MessagePassing
import pandas as pd
from OnDemandDataPrep.dataset_pyg_hsh import PygNodePropPredDataset_hsh
import numpy as np
from memory_profiler import memory_usage
from OnDemandDataPrep.calculate_embeddings_kglids import EmbeddingCreator

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

    def reset_parameters(self):
        for lin in self.rel_lins:
            lin.reset_parameters()
        for lin in self.root_lins:
            lin.reset_parameters()

    def forward(self, x, edge_index, edge_type, node_type):
        out = x.new_zeros(x.size(0), self.out_channels)

        for i in range(self.num_edge_types):
            mask = edge_type == i
            out.add_(self.propagate(edge_index[:, mask], x=x, edge_type=i))

        for i in range(self.num_node_types):
            mask = node_type == i
            out[mask] += self.root_lins[i](x[mask])

        return out

    def message(self, x_j, edge_type: int):
        return self.rel_lins[edge_type](x_j)


def get_column_emb(dataset_name_script,df_train):
    recommender = EmbeddingCreator()
    entity_df = df_train
    df = recommender.new_emb(entity_df,'unary-column')
    file1 = pd.read_csv('OnDemandDataPrep/storage/'+dataset_name_script+'_gnn_Column/mapping/Column_entidx2name.csv')
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

        # Create embeddings for all node types that do not come with features.
        self.emb_dict = ParameterDict({
            f'{key}': Parameter(torch.Tensor(num_nodes_dict[key], in_channels))
            for key in set(node_types).difference(set(x_types))
        })

        I, H, O = in_channels, hidden_channels, out_channels  # noqa

        # Create `num_layers` many message passing layers.
        self.convs = ModuleList()
        self.convs.append(RGCNConv(I, O, self.num_node_types, num_edge_types))



    def reset_parameters(self):
        for emb in self.emb_dict.values():
            torch.nn.init.xavier_uniform_(emb)
        for conv in self.convs:
            conv.reset_parameters()

    def group_input(self, x_dict, node_type, local_node_idx):
        # Create global node feature matrix.
        h = torch.zeros((node_type.size(0), self.in_channels),
                        device=node_type.device)

        for key, x in x_dict.items():
            mask = node_type == key
            h[mask] = x[local_node_idx[mask]]

        for key, emb in self.emb_dict.items():
            mask = node_type == int(key)
            h[mask] = emb[local_node_idx[mask]]

        return h

    def forward(self, x_dict, edge_index, edge_type, node_type,
                local_node_idx):

        x = self.group_input(x_dict, node_type, local_node_idx)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type, node_type)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        return x.log_softmax(dim=-1)

    def inference(self, x_dict, edge_index_dict, key2int, new_emb_dict=None):
        device = list(x_dict.values())[0].device

        x_dict = copy(x_dict)
        for key, emb in self.emb_dict.items():
            if torch.isnan(emb[0][0]):
                x_dict[int(key)] = new_emb_dict[key]
            else:
                x_dict[int(key)] = emb

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
                tmp = conv.rel_lins[key2int[keys]](tmp).detach().resize_([out.size()[0], out.size()[1]]) #NIKI
                out.add_(tmp)

            if i != self.num_layers - 1:
                for j in range(self.num_node_types):
                    F.relu_(out_dict[j])

            x_dict = out_dict
        return x_dict


dic_results = {}

def graphSaint(dataset_name_script,df_train):
    to_remove_pedicates = []
    to_remove_subject_object =[]
    to_keep_edge_idx_map = []
    GA_Index = 0
    MAG_datasets=[dataset_name_script+"_gnn_Column",]
    gsaint_Final_Test=[]
    for GA_dataset_name in MAG_datasets:
        dataset = PygNodePropPredDataset_hsh(name=GA_dataset_name,
                                             root='OnDemandDataPrep/storage/',
                                             numofClasses=str(3))
        dataset_name=GA_dataset_name+"_GA_"+str(GA_Index)
        dic_results[dataset_name] = {}
        dic_results[dataset_name]["GNN_Model"] = "GSaint"
        dic_results[dataset_name]["GA_Index"] = GA_Index
        dic_results[dataset_name]["to_keep_edge_idx_map"] = to_keep_edge_idx_map
        dic_results[dataset_name]["usecase"] = dataset_name

        start_t = datetime.datetime.now()
        data = dataset[0]
        global subject_node
        subject_node=list(data.y_dict.keys())[0]
        for key, tensor in data.edge_reltype.items():
            data.edge_reltype[key] = np.where(tensor == 0,3, tensor) #Change according to encoding

        end_t = datetime.datetime.now()
        dic_results[dataset_name]["GSaint_data_init_time"] = (end_t - start_t).total_seconds()
        # We do not consider those attributes for now.
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
            data.edge_reltype.pop(elem,None)

        for key in to_remove_subject_object:
            data.num_nodes_dict.pop(key, None)


        dic_results[dataset_name]["data"] = str(data)
        ##############add inverse edges ###################
        edge_index_dict = data.edge_index_dict
        key_lst = list(edge_index_dict.keys())
        for key in key_lst:
            r, c = edge_index_dict[(key[0], key[1], key[2])]
            edge_index_dict[(key[2], 'inv_' + key[1], key[0])] = torch.stack([c, r])

        out = group_hetero_graph(edge_index_dict, data.num_nodes_dict)
        edge_index, _, _, _, _, key2int = out


        start_t = datetime.datetime.now()


        end_t = datetime.datetime.now()
        dic_results[dataset_name]["GSaint_Sampling_time"] = (end_t - start_t).total_seconds()
        start_t = datetime.datetime.now()
        res = memory_usage((get_column_emb, (dataset_name_script,df_train,)), retval=True)
        max_memory3 = max(res[0])
        feat = res[1]

        if feat.size == 0:
            feat = torch.Tensor(data.num_nodes_dict[subject_node], 300)
            torch.nn.init.xavier_uniform_(feat)

        feat_dic = {subject_node: feat}
        x_dict = {}
        for key, x in feat_dic.items():
            x_dict[key2int[key]] = x

        num_nodes_dict = {}
        for key, N in data.num_nodes_dict.items():
            num_nodes_dict[key2int[key]] = N

        end_t = datetime.datetime.now()
        dic_results[dataset_name]["model init Time"] = (end_t - start_t).total_seconds()
        model = RGCN(300, 32, 3, 1, 0.5, {0: 201, 1: 52, 2: 541, 3: 3, 4: 237, 5: 1}, [0], 28).to('cpu')
        x_dict = {k: v.to('cpu') for k, v in x_dict.items()}

        loadTrainedModel = 1
        if loadTrainedModel == 1:
            with torch.no_grad():
                emb_dict = {}
                for key in model.emb_dict.keys():
                    emb_dict[key] = model.emb_dict[key].detach().clone()
                model.load_state_dict(torch.load("OnDemandDataPrep/Models/kgfarm_gnn_GA_0_DBLP_conf_GSAINT_QM_e10_r3_s10_lr005_layer1_w4_hc32_new_emb_v2_82split_3target.model"))
                out = model.inference(x_dict, edge_index_dict, key2int, new_emb_dict=emb_dict)
                out = out[key2int[subject_node]]
                y_pred = out.argmax(dim=-1, keepdim=True).cpu()
                pred_lst = torch.flatten(y_pred).tolist()
                gsaint_Final_Test = pred_lst

    return gsaint_Final_Test






