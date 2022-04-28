#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:24:16 2020

@author: zichan
"""

import pandas as pd
import numpy as np
import dgl
import tensorflow as tf
import networkx as nx
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder

#G = nx.read_edgelist('PMADS_dis_sta/edgecon/edge0928.edge', nodetype=int, create_using=nx.DiGraph()) ## nodetype = str
G = nx.read_edgelist('../all_data/alledge.edge', nodetype=str, create_using=nx.DiGraph())

for edge in G.edges(): 
    G[edge[0]][edge[1]]['weight'] = 1
    
G = G.to_undirected()

# ### homophily
# node2vec = Node2Vec(G, dimensions=128, p=1, q=0.5)
### structually equivalence
node2vec = Node2Vec(G, dimensions=8, p=1, q=2)

#, walk_length=30, num_walks=200, workers=4)


model = node2vec.fit(window=10, min_count=1, batch_words=4)
# model.wv.most_similar('AAAA')


model.wv.save_word2vec_format('../all_data/allnode8stru.nodeemb')

edges_embs = HadamardEmbedder(keyed_vectors=model.wv)

# Get all edges in a separate KeyedVectors instance - use with caution could be huge for big networks
edges_kv = edges_embs.as_keyed_vectors()

# Look for most similar edges - this time tuples must be sorted and as str
edges_kv.most_similar(str(('AAAA', 'AAAB')))

# Save embeddings for later use
edges_kv.save_word2vec_format('n2vemb/alledge.edgeemb')



out_label = pd.read_csv('../data/pmads_data.csv')    
out_label = out_label[['Time','KEY','label']]
all_edge = np.load("../all_data/static_all_edge.edge.npy", allow_pickle=True)
nodekey_file = open("../all_data/nodes.ndkey", 'r')
nodekey = nodekey_file.readlines()
nodekey = np.array([nodekey[i].split('\t')[0].strip() for i in range(len(nodekey))])
source_node = [all_edge[i,1] for i in range(len(all_edge))]
sink_node = [all_edge[i,2] for i in range(len(all_edge))]
source_node_id = [int(np.where(nodekey == source_node[i])[0]) for i in range(len(source_node))]
sink_node_id = [int(np.where(nodekey == sink_node[i])[0]) for i in range(len(sink_node))]
origG = dgl.graph((source_node_id, sink_node_id),num_nodes=len(nodekey))
origG.edata['id'] = tf.constant(range(2142))
## turn into line graph with edge id transfered along to node id.
lineG = origG.line_graph(shared=True)
g = dgl.to_networkx(lineG, node_attrs=['id'])
edge_node2vec = Node2Vec(g, dimensions=8)#, walk_length=30, num_walks=200, workers=4)
edge_model = edge_node2vec.fit(window=10, min_count=1, batch_words=4)
edge_model.wv.save_word2vec_format('../all_data/alledge8.edgeemb')