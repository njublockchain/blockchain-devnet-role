# %%
import torch
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data
import networkx as nx
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 指定设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 设置随机种子以确保结果的一致性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(42)

class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super(RGCN, self).__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations=num_relations)
        self.conv2 = RGCNConv(hidden_channels, out_channels, num_relations=num_relations)

    def forward(self, x, edge_index, edge_type):
        x = F.relu(self.conv1(x.to(device), edge_index.to(device), edge_type=edge_type))
        x = self.conv2(x, edge_index.to(device), edge_type=edge_type)
        return x

class LSTMEmbedder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMEmbedder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Return the last hidden state
        return out[:, -1, :]

def encode_events(events, event_types_dict):
    """将事件类型编码为整数，根据边的方向调整正负"""
    return [event_types_dict[event_type] * direction for _, event_type, direction in events]

def prepare_sequences(node_events, node_mapping, event_types_dict):
    """准备LSTM的输入序列，包括编码事件类型和排序"""
    sequences = {node: [] for node in node_mapping}
    for node, events in node_events.items():
        encoded_events = encode_events(events, event_types_dict)
        sequences[node] = encoded_events
    return sequences

def sequence_to_embedding(sequences, lstm_embedder, device, node_mapping):
    """将事件序列转换为LSTM嵌入向量"""
    embeddings = torch.zeros((len(sequences), lstm_embedder.hidden_size), device=device)
    for node, sequence in sequences.items():
        if not sequence:  # 如果序列为空，则跳过
            continue
        sequence_tensor = torch.tensor(sequence, dtype=torch.float).view(1, -1, 1).to(device)
        embedding = lstm_embedder(sequence_tensor)
        embeddings[node_mapping[node]] = embedding.squeeze(0)  # 使用传入的 node_mapping 进行索引
    return embeddings


def load_data(gexf_path, lstm_embedder, device):
    G = nx.read_gexf(gexf_path)
    edge_index = []
    edge_type = []
    node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
    edge_types = {etype: i for i, etype in enumerate(set(nx.get_edge_attributes(G, 'event_type').values()))}

    # 初始化入度和出度计数
    in_counts = {node: {etype: 0 for etype in edge_types} for node in G.nodes()}
    out_counts = {node: {etype: 0 for etype in edge_types} for node in G.nodes()}
    
    # 初始化存储价格信息的字典
    in_prices = {node: [] for node in G.nodes()}
    out_prices = {node: [] for node in G.nodes()}

    # 准备节点事件数据结构
    node_events = {node: [] for node in G.nodes()}
    
    for u, v, data in G.edges(data=True):
        edge_index.append([node_mapping[u], node_mapping[v]])
        edge_type.append(edge_types[data['event_type']])
        out_counts[u][data['event_type']] += 1
        in_counts[v][data['event_type']] += 1

        # 假设时间戳信息存在于边属性中
        timestamp = data['created_at']
        node_events[u].append((timestamp, data['event_type'], 1))  # 出度事件，正数
        node_events[v].append((timestamp, data['event_type'], -1))  # 入度事件，负数
        
        # 根据边的方向分别存储价格信息
        if 'price' in data:  # 确保数据中有价格信息
            out_prices[u].append(data['price'])
            in_prices[v].append(data['price'])
    
    # 准备和编码事件序列
    sequences = prepare_sequences(node_events, node_mapping, edge_types)

    # 在 load_data 函数内部调用 sequence_to_embedding 时传入 node_mapping
    embeddings = sequence_to_embedding(sequences, lstm_embedder, device, node_mapping)

    # 计算价格的平均值和方差
    in_price_means = {node: np.mean(prices) if prices else 0 for node, prices in in_prices.items()}
    in_price_vars = {node: np.var(prices) if prices else 0 for node, prices in in_prices.items()}
    out_price_means = {node: np.mean(prices) if prices else 0 for node, prices in out_prices.items()}
    out_price_vars = {node: np.var(prices) if prices else 0 for node, prices in out_prices.items()}

    # 构建特征矩阵，加入每个节点的入度和出度
    num_nodes = len(G.nodes())
    num_edge_types = len(edge_types)
    lstm_embedding_size = lstm_embedder.hidden_size

    # 特征矩阵增加(6+X)列：4列为价格的平均值和方差，2列为入度和出度，另外是lstm序列
    x = torch.zeros((num_nodes, num_edge_types * 2 + 6 + lstm_embedding_size), dtype=torch.float)

    for node, idx in node_mapping.items():
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        for etype, count in in_counts[node].items():
            x[idx, edge_types[etype]] = count
        for etype, count in out_counts[node].items():
            x[idx, edge_types[etype] + num_edge_types] = count
        # 添加价格特征
        x[idx, -6 - lstm_embedding_size] = in_price_means[node]
        x[idx, -5 - lstm_embedding_size] = in_price_vars[node]
        x[idx, -4 - lstm_embedding_size] = out_price_means[node]
        x[idx, -3 - lstm_embedding_size] = out_price_vars[node]
        # 添加入度和出度特征
        x[idx, -2 - lstm_embedding_size] = in_degree
        x[idx, -1 - lstm_embedding_size] = out_degree
        # 合并 LSTM 嵌入
        x[idx, -lstm_embedding_size:] = embeddings[idx]
    
    # 指定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    edge_type = torch.tensor(edge_type, dtype=torch.long).to(device)

    return Data(x=x.to(device), edge_index=edge_index, edge_type=edge_type), len(edge_types), G, node_mapping

def adjacency_matrix(edge_index, num_nodes):
    adj_matrix = torch.zeros((num_nodes, num_nodes)).to(device)
    for source, target in edge_index.t():
        adj_matrix[source, target] = 1
    return adj_matrix

def train(model, data, optimizer, epochs=100):
    adj_matrix = adjacency_matrix(data.edge_index, data.num_nodes).to(data.x.device)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        embeddings = model(data.x, data.edge_index, data.edge_type)
        
        pred_adj_matrix = torch.sigmoid(torch.mm(embeddings, embeddings.t()))
        loss = F.binary_cross_entropy(pred_adj_matrix, adj_matrix)
        
        loss.backward(retain_graph=True)
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')

def cluster_and_evaluate(embeddings, cluster_method):
    scores = []
    for n_clusters in range(2, 11):
        if cluster_method == 'kmeans':
            clustering_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        else:  # AHC
            clustering_model = AgglomerativeClustering(n_clusters=n_clusters)

        labels = clustering_model.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        scores.append(score)
        print(f'Clusters: {n_clusters}, Silhouette Score: {score}')

    optimal_clusters = np.argmax(scores) + 2
    print(f'Optimal number of clusters: {optimal_clusters}')
    
    # Re-run clustering with the optimal number of clusters to get the final labels
    if cluster_method == 'kmeans':
        optimal_clustering_model = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
    else:
        optimal_clustering_model = AgglomerativeClustering(n_clusters=optimal_clusters)
    optimal_labels = optimal_clustering_model.fit_predict(embeddings)
    
    return optimal_labels

# 定义 LSTM 嵌入器的参数
input_size = 1  # LSTM 每个时间步的输入特征维度
hidden_size = 128  # LSTM 隐藏层的大小
num_layers = 1  # LSTM 网络的层数

# 创建 LSTM 嵌入器实例
lstm_embedder = LSTMEmbedder(input_size, hidden_size, num_layers)

# 确定使用的设备，CUDA 如果可用的话
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lstm_embedder.to(device)

# Adjust these paths according to your environment
gexf_path = '/home/ta/devnet/data_source/bitcoin_2023_monthly_time_price_gexf/2022-12.gexf'
data, num_relations, G, node_mapping = load_data(gexf_path, lstm_embedder, device)
print(num_relations)

#---------------------------------
# 获取所有唯一的事件类型
unique_event_types = sorted(set(nx.get_edge_attributes(G, 'event_type').values()))

# 为每种事件类型生成入度和出度的特征名
feature_columns = ['In_' + etype for etype in unique_event_types] + ['Out_' + etype for etype in unique_event_types]

# 加入新特征的列名：价格统计信息
new_feature_columns = ['InPriceMean', 'InPriceVar', 'OutPriceMean', 'OutPriceVar']
feature_columns += new_feature_columns

# 加入入度和出度作为新特征的列名
feature_columns += ['InDegree', 'OutDegree']

# 为 LSTM 嵌入的每个维度生成列名
lstm_feature_columns = ['LSTM_Feature_' + str(i) for i in range(hidden_size)]

# 将 LSTM 特征列名加入到 feature_columns 列表中
feature_columns += lstm_feature_columns

# 将特征矩阵转换为Pandas DataFrame
features_np = data.x.detach().cpu().numpy()
nodes = [node for node in G.nodes()]
features_df = pd.DataFrame(features_np, columns=feature_columns)
features_df.insert(0, 'Node', nodes)

# 导出DataFrame到CSV文件
csv_file_path = '/home/ta/devnet/evolved-RGCN/node_feature_matrix/rgcn_tpe_bitcoin_202212_node_features_matrix.csv'
features_df.to_csv(csv_file_path, index=False)  # 不将自动生成的索引导出到CSV文件中

print(f"特征矩阵已导出到 {csv_file_path}")


#---------------------------------
# 归一化特征矩阵
scaler = MinMaxScaler()
features_np = data.x.detach().cpu().numpy()  # 假设data.x是从你的图G中提取的特征矩阵
features_normalized = scaler.fit_transform(features_np)

# 确保feature_columns包含所有特征列名，包括新添加的价格统计信息列名
# 如果之前已经更新了feature_columns，则此处无需再次更新

# 转换归一化后的特征为Pandas DataFrame，并为节点列指定列名称"Node"
nodes = list(G.nodes())  # 获取节点列表
features_df = pd.DataFrame(features_normalized, columns=feature_columns, index=nodes)
features_df.reset_index(inplace=True, drop=False)  # 将节点从索引转换为列，不保留原索引
features_df.rename(columns={'index': 'Node'}, inplace=True)  # 重命名该列为"Node"

# 导出DataFrame到CSV文件，不包含索引列
csv_file_path = '/home/ta/devnet/evolved-RGCN/node_feature_normalized_matrix/rgcn_tpe_bitcoin_202212_normalized_features_matrix.csv'
features_df.to_csv(csv_file_path, index=False)  # 确保不将自动生成的索引导出到CSV

print(f"归一化后的特征矩阵已导出到 {csv_file_path}")

# %%
model = RGCN(data.num_features, 64, 16, num_relations).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train(model, data, optimizer, epochs=20)

model.eval()
with torch.no_grad():
    embeddings = model(data.x, data.edge_index, data.edge_type).detach().cpu().numpy()

optimal_labels = cluster_and_evaluate(embeddings, cluster_method='ahc')

# %%
# 计算网络指标
# in_degree_dict = dict(G.in_degree())
# out_degree_dict = dict(G.out_degree())
# degree_dict = dict(G.degree())
# try:
#     eccentricity_dict = nx.eccentricity(G)
# except:
#     eccentricity_dict = {node: 0 for node in G.nodes()}  # 对于非连通图
# closeness_centrality_dict = nx.closeness_centrality(G)
# betweenness_centrality_dict = nx.betweenness_centrality(G)
# page_rank_dict = nx.pagerank(G)
# clustering_dict = nx.clustering(G)
# eigenvector_centrality_dict = nx.eigenvector_centrality(G)
# # HITS 算法可能不在所有网络类型上运行，这里用try-except块
# try:
#     hits_hub_dict, hits_authority_dict = nx.hits(G)
# except:
#     hits_hub_dict = {node: 0 for node in G.nodes()}
#     hits_authority_dict = {node: 0 for node in G.nodes()}

# Update the original NetworkX graph with cluster labels
for node, idx in node_mapping.items():
    G.nodes[node]['cluster'] = int(optimal_labels[idx])

    # # 添加网络指标
    # G.nodes[node].update({
    #     'in_degree': in_degree_dict[node],
    #     'out_degree': out_degree_dict[node],
    #     'degree': degree_dict[node],
    #     'eccentricity': eccentricity_dict[node],
    #     'closeness_centrality': closeness_centrality_dict[node],
    #     'betweenness_centrality': betweenness_centrality_dict[node],
    #     'page_rank': page_rank_dict[node],
    #     'clustering_coefficient': clustering_dict[node],
    #     'eigenvector_centrality': eigenvector_centrality_dict[node],
    #     'authority': hits_authority_dict[node],
    #     'hub': hits_hub_dict[node]
    # })

    # 提取该节点的特征
    node_features = features_np[idx]
    
    # 创建一个属性字典，每个特征对应一个属性（这里用列名作为属性名）
    feature_attributes = {feature_columns[i]: float(node_features[i]) for i in range(len(feature_columns))}
    
    # 将属性字典添加到对应的节点上
    G.nodes[node].update(feature_attributes)

# Write the updated graph to a new GEXF file
new_gexf_path = '/home/ta/devnet/evolved-RGCN/cluster_result/bips_tpe_202212_time_price_clustered.gexf'
nx.write_gexf(G, new_gexf_path)
# %%
