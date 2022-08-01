import argparse
import os
from datetime import datetime
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
import numpy as np
from tqdm import tqdm

import networkx as nx
import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score

from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy

import wandb


def load_graphs_to_stellargraph(graph_uris, graphs_dir):
    uri_to_sg_graph = {}
    for uri in tqdm(graph_uris):
        file_path = os.path.join(graphs_dir, uri.lstrip('http://kglids.org/resource/kaggle/') + '.tsv')
        df_spo = pd.read_csv(file_path, delimiter='\t').astype(str)
        node_embeddings = pd.read_pickle(file_path.replace('.tsv', '.pickle'))
        g = nx.DiGraph()
        df_spo.apply(lambda x: g.add_edge(x['s'], x['o'], type=x['p']), axis=1)

        for node in g.nodes():
            g.nodes[node]['features'] = node_embeddings[node]['transE']  # ['complEx']  # TODO: complEx or transE?

        g = sg.StellarDiGraph.from_networkx(g, edge_type_attr='type', node_features='features')
        uri_to_sg_graph[uri] = g

    return uri_to_sg_graph


def train_and_evaluate_classification_model(train_graphs, val_graphs, test_graphs, train_labels, val_labels, test_labels,
                                            num_classes, epochs=100, batch_size=50, sysname='KGLiDS'):
    gen = PaddedGraphGenerator(graphs=train_graphs + val_graphs + test_graphs)
    k = wandb.config.k  # the number of rows for the output tensor
    layer_sizes = [wandb.config.gcn_layer_size] * (wandb.config.gcn_layers - 1) + [1]  # last layer is of size 1.
    activations = ["tanh"] * wandb.config.gcn_layers
    dgcnn_model = DeepGraphCNN(
        layer_sizes=layer_sizes,
        activations=activations,
        k=k,
        bias=False,
        generator=gen,
    )
    x_inp, x_out = dgcnn_model.in_out_tensors()

    x_out = Conv1D(filters=16, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)
    x_out = MaxPool1D(pool_size=2)(x_out)

    x_out = Conv1D(filters=32, kernel_size=5, strides=1)(x_out)

    x_out = Flatten()(x_out)

    x_out = Dense(units=wandb.config.fc_size, activation="relu")(x_out)  # 128
    x_out = Dropout(rate=wandb.config.dropout)(x_out)

    predictions = Dense(units=num_classes, activation="softmax")(x_out)

    model = Model(inputs=x_inp, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=wandb.config.lr), loss=categorical_crossentropy, metrics=["acc"])

    train_gen = gen.flow(
        graphs=train_graphs,
        targets=train_labels,
        batch_size=batch_size,
        symmetric_normalization=False,
    )
    val_gen = gen.flow(
        graphs=val_graphs,
        targets=val_labels,
        batch_size=1,
        symmetric_normalization=False,
    )
    test_gen = gen.flow(
        graphs=test_graphs,
        targets=test_labels,
        batch_size=1,
        symmetric_normalization=False,
    )

    checkpoint = ModelCheckpoint(f'tmp/{task}_{sysname}.h5', verbose=False, monitor='val_acc', save_weights_only=True,
                                 save_best_only=True, mode='max')
    # fit
    history = model.fit(train_gen, epochs=epochs, verbose=1, validation_data=val_gen,
                        shuffle=True, callbacks=[checkpoint])  # , callbacks=[Metrics(model, test_graphs, test_labels)])

    for i in range(len(history.history['loss'])):
        wandb.log({"Epoch": i + 1, f"{sysname} Train Loss": history.history['loss'][i],
                   f"{sysname} Train Acc": history.history['acc'][i],
                   f"{sysname} Valid Loss": history.history['val_loss'][i],
                   f"{sysname} Valid Acc": history.history['val_acc'][i]})
    model.load_weights(f'tmp/{task}_{sysname}.h5')  # best model
    test_predictions = model.predict(test_gen)
    test_accuracy = accuracy_score(np.argmax(test_labels, axis=1), np.argmax(test_predictions, axis=1))
    print(f"{sysname} Test Accuracy:", round(test_accuracy, 4))

    best_epoch = np.argmax(history.history['val_acc'])
    wandb.log({f"{sysname} Best Train Acc": history.history['acc'][best_epoch],
               f"{sysname} Best Valid Acc": history.history['val_acc'][best_epoch],
               f"{sysname} Test Acc": round(test_accuracy, 4)})

    sg.utils.plot_history(history)



parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--dropout', type=float, default=0.25)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--gcn_layers', type=int, default=3)
parser.add_argument('--gcn_layer_size', type=int, default=32)
parser.add_argument('--k', type=int, default=35)
parser.add_argument('--fc_size', type=int, default=32)

args = parser.parse_args()

wandb.init(project="task3-ml-tasks",
           config={'epochs': args.epochs, 'dropout': args.dropout, 'lr': args.lr, 'gcn_layers': args.gcn_layers,
                   'gcn_layer_size': args.gcn_layer_size, 'k': args.k, 'fc_size': args.fc_size})



task = 'task3'

# get graph names and classes
uris_labels = pd.read_csv(f'{task}_uris_labels.csv')
uris_labels = uris_labels[uris_labels['uri'].apply(
    lambda x: os.path.exists(f'{task}_kglids_graphs/' + x.lstrip('http://kglids.org/resource/kaggle/') + '.tsv'))]
uris_labels = uris_labels[uris_labels['uri'].apply(
    lambda x: os.path.exists(f'{task}_kglids_graphs/' + x.lstrip('http://kglids.org/resource/kaggle/') + '.pickle'))]
uris_labels = uris_labels[uris_labels['uri'].apply(
    lambda x: os.path.exists(f'{task}_graph4code_graphs/' + x.lstrip('http://kglids.org/resource/kaggle/') + '.tsv'))]
uris_labels = uris_labels[uris_labels['uri'].apply(lambda x: os.path.exists(
    f'{task}_graph4code_graphs/' + x.lstrip('http://kglids.org/resource/kaggle/') + '.pickle'))]
uris_labels['label'] = uris_labels['label'].astype('category').cat.codes
pips = uris_labels['uri'].tolist()
labels = uris_labels['label'].tolist()
num_pipeline_classes = len(uris_labels['label'].unique())
print(len(pips), 'Pipelines')
print(uris_labels.label.value_counts())

train_names, test_names, train_labels, test_labels = train_test_split(pips, labels, train_size=0.8, stratify=labels,
                                                                      random_state=3)
val_names, test_names, val_labels, test_labels = train_test_split(test_names, test_labels, train_size=0.5,
                                                                  stratify=test_labels, random_state=3)
encoder = LabelBinarizer()
train_labels, val_labels, test_labels = encoder.fit_transform(train_labels), encoder.fit_transform(val_labels), encoder.fit_transform(test_labels)

kglids_stellargraph = load_graphs_to_stellargraph(pips, f'{task}_kglids_graphs')
d0 = datetime.now()
train_and_evaluate_classification_model(train_graphs=[kglids_stellargraph[i] for i in train_names],
                                        val_graphs = [kglids_stellargraph[i] for i in val_names],
                                        test_graphs=[kglids_stellargraph[i] for i in test_names],
                                        train_labels=train_labels, val_labels=val_labels, test_labels=test_labels,
                                        num_classes=num_pipeline_classes,
                                        epochs=wandb.config.epochs,
                                        batch_size=50, sysname='KGLiDS')
d1 = datetime.now()

kglids_stellargraph = None
graph4code_stellargraph = load_graphs_to_stellargraph(pips, f'{task}_graph4code_graphs')

d2 = datetime.now()
train_and_evaluate_classification_model(train_graphs=[graph4code_stellargraph[i] for i in train_names],
                                        val_graphs=[graph4code_stellargraph[i] for i in val_names],
                                        test_graphs=[graph4code_stellargraph[i] for i in test_names],
                                        train_labels=train_labels, val_labels=val_labels, test_labels=test_labels, 
                                        num_classes=num_pipeline_classes,
                                        epochs=wandb.config.epochs,
                                        batch_size=5, sysname='GraphGen4Code')
d3 = datetime.now()

print('KGLiDS Time:', d1 - d0)
print('GraphGen4Code Time:', d3 - d2)
