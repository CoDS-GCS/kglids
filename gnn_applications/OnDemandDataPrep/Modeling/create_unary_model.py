from copy import copy
import argparse
import shutil
from tqdm import tqdm
import datetime
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ParameterDict, Parameter
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.nn import MessagePassing
import pandas as pd
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from gnn_applications.OnDemandDataPrep.dataset_pyg_hsh import PygNodePropPredDataset_hsh
from gnn_applications.OnDemandDataPrep.logger import Logger
import faulthandler
faulthandler.enable()
import numpy as np

subject_node = '1Column'



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


def get_column_emb(graph_name):
    df = pd.read_csv(
        'OnDemandDataPrep/Modeling/storage/Embeddings_columns_' + graph_name + '.csv')
    file1 = pd.read_csv(
        'OnDemandDataPrep/Modeling/storage/' + graph_name + '/mapping/1Column_entidx2name.csv')

    file1['ent name'] = file1['ent name'].astype(str)
    df_emb = pd.merge(file1, df, left_on='ent name', right_on='Key', how='left')
    df_emb = df_emb.drop('ent name', axis=1)
    df_emb = df_emb.drop('ent idx', axis=1)
    df_emb.to_csv('training_data.csv')
    df_emb = df_emb.drop('Key', axis=1)
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

        self.reset_parameters()

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

    def inference(self, x_dict, edge_index_dict, key2int):
        device = list(x_dict.values())[0].device

        x_dict = copy(x_dict)
        for key, emb in self.emb_dict.items():
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
                tmp = conv.rel_lins[key2int[keys]](tmp).resize_([out.size()[0], out.size()[1]])
                out.add_(tmp)

            if i != self.num_layers - 1:
                for j in range(self.num_node_types):
                    F.relu_(out_dict[j])

            x_dict = out_dict

        return x_dict


dic_results = {}


def graphSaint(graph_name):
    def train(epoch):
        model.train()
        pbar = tqdm(total=args.num_steps * args.batch_size)
        pbar.set_description(f'Epoch {epoch:02d}')

        total_loss = total_examples = 0
        for data in train_loader:
            start_t = datetime.datetime.now()
            data = data.to(device)
            optimizer.zero_grad()
            out = model(x_dict, data.edge_index, data.edge_attr, data.node_type,
                        data.local_node_idx)
            out = out[data.train_mask]
            y = data.y[data.train_mask].squeeze()
            loss = F.nll_loss(out, y)
            loss.backward()
            optimizer.step()
            num_examples = data.train_mask.sum().item()
            total_loss += loss.item() * num_examples
            total_examples += num_examples
            pbar.update(args.batch_size)
            end_t = datetime.datetime.now()


        pbar.close()
        return total_loss / total_examples

    @torch.no_grad()
    def test():
        model.eval()

        out = model.inference(x_dict, edge_index_dict, key2int)
        out = out[key2int[subject_node]]

        y_pred = out.argmax(dim=-1, keepdim=True).cpu()
        y_true = data.y_dict[subject_node]

        train_acc = evaluator.eval({
            'y_true': y_true[split_idx['train'][subject_node]],
            'y_pred': y_pred[split_idx['train'][subject_node]],
        })['acc']
        valid_acc = evaluator.eval({
            'y_true': y_true[split_idx['valid'][subject_node]],
            'y_pred': y_pred[split_idx['valid'][subject_node]],
        })['acc']
        test_acc = evaluator.eval({
            'y_true': y_true[split_idx['test'][subject_node]],
            'y_pred': y_pred[split_idx['test'][subject_node]],
        })['acc']
        print('val labels:', y_true[split_idx['valid'][subject_node]], y_pred[split_idx['valid'][subject_node]])
        print('test labels:', y_true[split_idx['test'][subject_node]], y_pred[split_idx['test'][subject_node]])
        return train_acc, valid_acc, test_acc

    parser = argparse.ArgumentParser(description='OGBN-MAG (GraphSAINT)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--walk_length', type=int, default=4)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--loadTrainedModel', type=int, default=0)
    parser.add_argument('--graphsaint_dic_path', type=str, default='none')

    args = parser.parse_args()

    to_remove_pedicates = []
    to_remove_subject_object = []
    to_keep_edge_idx_map = []
    GA_Index = 0

    GA_dataset_name = graph_name

    gsaint_Final_Test = 0
    gsaint_start_t = datetime.datetime.now()
    ###################################Delete Folder if exist #############################
    dir_path = "OnDemandDataPrep/Modeling/storage/" + GA_dataset_name
    try:
        shutil.rmtree(dir_path)
        print("Folder Deleted")
    except OSError as e:
        print("Error Deleting : %s : %s" % (dir_path, e.strerror))
    dataset = PygNodePropPredDataset_hsh(name=GA_dataset_name,
                                         root='OnDemandDataPrep/Modeling/storage/',
                                         numofClasses=str(3))
    dataset_name = GA_dataset_name + "_GA_" + str(GA_Index)

    dic_results[dataset_name] = {}
    dic_results[dataset_name]["GNN_Model"] = "GSaint"
    dic_results[dataset_name]["GA_Index"] = GA_Index
    dic_results[dataset_name]["to_keep_edge_idx_map"] = to_keep_edge_idx_map
    dic_results[dataset_name]["usecase"] = dataset_name
    dic_results[dataset_name]["gnn_hyper_params"] = str(args)

    start_t = datetime.datetime.now()
    data = dataset[0]
    global subject_node
    subject_node = list(data.y_dict.keys())[0]
    split_idx = dataset.get_idx_split('random')  # Can Change
    end_t = datetime.datetime.now()
    dic_results[dataset_name]["GSaint_data_init_time"] = (end_t - start_t).total_seconds()
    evaluator = Evaluator(name='ogbn-mag')
    logger = Logger(args.runs, args)

    start_t = datetime.datetime.now()
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
        data.edge_reltype.pop(elem, None)

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
    edge_index, edge_type, node_type, local_node_idx, local2global, key2int = out

    homo_data = Data(edge_index=edge_index, edge_attr=edge_type,
                     node_type=node_type, local_node_idx=local_node_idx,
                     num_nodes=node_type.size(0))

    homo_data.y = node_type.new_full((node_type.size(0), 1), -1)
    homo_data.y[local2global[subject_node]] = data.y_dict[subject_node]

    homo_data.train_mask = torch.zeros((node_type.size(0)), dtype=torch.bool)
    homo_data.train_mask[local2global[subject_node][split_idx['train'][subject_node]]] = True

    start_t = datetime.datetime.now()
    train_loader = GraphSAINTRandomWalkSampler(
        homo_data,
        batch_size=args.batch_size,
        walk_length=args.num_layers,
        num_steps=args.num_steps,
        sample_coverage=0,
        save_dir=dataset.processed_dir)

    end_t = datetime.datetime.now()
    dic_results[dataset_name]["GSaint_Sampling_time"] = (end_t - start_t).total_seconds()
    start_t = datetime.datetime.now()
    feat2 = get_column_emb(graph_name)
    feat_dic = {subject_node: feat2}
    ################################################################
    x_dict = {}
    for key, x in feat_dic.items():
        x_dict[key2int[key]] = x

    num_nodes_dict = {}
    for key, N in data.num_nodes_dict.items():
        num_nodes_dict[key2int[key]] = N

    end_t = datetime.datetime.now()
    dic_results[dataset_name]["model init Time"] = (end_t - start_t).total_seconds()
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    model = RGCN(300, args.hidden_channels, dataset.num_classes, args.num_layers,
                 args.dropout, num_nodes_dict, list(x_dict.keys()),
                 len(edge_index_dict.keys())).to(device)

    x_dict = {k: v.to(device) for k, v in x_dict.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.loadTrainedModel == 1:
        model.load_state_dict(torch.load("ogbn-DBLP-FM-GSaint.model"))
        model.eval()
        out = model.inference(x_dict, edge_index_dict, key2int)
        out = out[key2int[subject_node]]
        y_pred = out.argmax(dim=-1, keepdim=True).cpu()
        y_true = data.y_dict[subject_node]

        out_lst = torch.flatten(y_true).tolist()
        pred_lst = torch.flatten(y_pred).tolist()
        out_df = pd.DataFrame({"y_pred": pred_lst, "y_true": out_lst})
        out_df.to_csv("GSaint_DBLP_conf_output.csv", index=None)
    else:
        test()  # Test if inference on GPU succeeds.
        total_run_t = 0
        for run in range(args.runs):
            start_t = datetime.datetime.now()
            model.reset_parameters()
            for epoch in range(1, 1 + args.epochs):
                start_t = datetime.datetime.now()
                loss = train(epoch)
                ##############
                if loss == -1:
                    return 0.001
                ##############

                torch.cuda.empty_cache()
                result = test()
                logger.add_result(run, result)
                train_acc, valid_acc, test_acc = result
                end_t = datetime.datetime.now()
                print("Epoch time=", end_t - start_t, " sec.")
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}%, '
                      f'Test: {100 * test_acc:.2f}%')
            logger.print_statistics(run)
            end_t = datetime.datetime.now()
            total_run_t = total_run_t + (end_t - start_t).total_seconds()

        total_run_t = (total_run_t + 0.00001) / args.runs
        gsaint_end_t = datetime.datetime.now()
        Highest_Train, Highest_Valid, Final_Train, Final_Test = logger.print_statistics()
        dic_results[dataset_name]["Highest_Train"] = Highest_Train.item()
        dic_results[dataset_name]["Highest_Valid"] = Highest_Valid.item()
        dic_results[dataset_name]["Final_Train"] = Final_Train.item()
        gsaint_Final_Test = Final_Test.item()
        dic_results[dataset_name]["Final_Test"] = Final_Test.item()
        dic_results[dataset_name]["runs_count"] = args.runs
        dic_results[dataset_name]["avg_train_time"] = total_run_t
        dic_results[dataset_name]["rgcn_total_time"] = (gsaint_end_t - gsaint_start_t).total_seconds()
        pd.DataFrame(dic_results).transpose().to_csv(
            "OnDemandDataPrep/Modeling/storage/GSAINT_" + GA_dataset_name + "_Times_unary.csv",
            index=False)
        print('model.state_dict()', model.state_dict())
        torch.save(model.state_dict(),
                   "OnDemandDataPrep/Models/" + dataset_name + "_unary.model")
        print('Modeling Completed!')
