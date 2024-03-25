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

def load_data(gexf_path):
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

    for u, v, data in G.edges(data=True):
        edge_index.append([node_mapping[u], node_mapping[v]])
        edge_type.append(edge_types[data['event_type']])
        out_counts[u][data['event_type']] += 1
        in_counts[v][data['event_type']] += 1
        
        # 根据边的方向分别存储价格信息
        if 'price' in data:  # 确保数据中有价格信息
            out_prices[u].append(data['price'])
            in_prices[v].append(data['price'])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long)

    # 计算价格的平均值和方差
    in_price_means = {node: np.mean(prices) if prices else 0 for node, prices in in_prices.items()}
    in_price_vars = {node: np.var(prices) if prices else 0 for node, prices in in_prices.items()}
    out_price_means = {node: np.mean(prices) if prices else 0 for node, prices in out_prices.items()}
    out_price_vars = {node: np.var(prices) if prices else 0 for node, prices in out_prices.items()}

    # 构建特征矩阵，加入每个节点的入度和出度
    num_nodes = len(G.nodes())
    num_edge_types = len(edge_types)
    # 特征矩阵增加6列：4列为价格的平均值和方差，2列为入度和出度
    x = torch.zeros((num_nodes, num_edge_types * 2 + 6), dtype=torch.float)
    for node, idx in node_mapping.items():
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        for etype, count in in_counts[node].items():
            x[idx, edge_types[etype]] = count
        for etype, count in out_counts[node].items():
            x[idx, edge_types[etype] + num_edge_types] = count
        # 新增的价格特征
        x[idx, -6] = in_price_means[node]
        x[idx, -5] = in_price_vars[node]
        x[idx, -4] = out_price_means[node]
        x[idx, -3] = out_price_vars[node]
        # 新增的入度和出度特征
        x[idx, -2] = in_degree
        x[idx, -1] = out_degree

    # 指定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return Data(x=x.to(device), edge_index=edge_index.to(device), edge_type=edge_type.to(device)), len(edge_types), G, node_mapping

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
        
        loss.backward()
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


# Adjust these paths according to your environment
gexf_path = '/home/ta/devnet/bips_2023_time_price/2023-01.gexf'
data, num_relations, G, node_mapping = load_data(gexf_path)
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

# 将特征矩阵转换为Pandas DataFrame
features_np = data.x.cpu().numpy()
nodes = [node for node in G.nodes()]
features_df = pd.DataFrame(features_np, columns=feature_columns)
features_df.insert(0, 'Node', nodes)

# 导出DataFrame到CSV文件
csv_file_path = '/home/ta/devnet/rgcn_pe_202301_node_features_matrix.csv'
features_df.to_csv(csv_file_path, index=False)  # 不将自动生成的索引导出到CSV文件中

print(f"特征矩阵已导出到 {csv_file_path}")


#---------------------------------
# 归一化特征矩阵
scaler = MinMaxScaler()
features_np = data.x.cpu().numpy()  # 假设data.x是从你的图G中提取的特征矩阵
features_normalized = scaler.fit_transform(features_np)

# 确保feature_columns包含所有特征列名，包括新添加的价格统计信息列名
# 如果之前已经更新了feature_columns，则此处无需再次更新

# 转换归一化后的特征为Pandas DataFrame，并为节点列指定列名称"Node"
nodes = list(G.nodes())  # 获取节点列表
features_df = pd.DataFrame(features_normalized, columns=feature_columns, index=nodes)
features_df.reset_index(inplace=True, drop=False)  # 将节点从索引转换为列，不保留原索引
features_df.rename(columns={'index': 'Node'}, inplace=True)  # 重命名该列为"Node"

# 导出DataFrame到CSV文件，不包含索引列
csv_file_path = '/home/ta/devnet/rgcn_pe_202301_normalized_features_matrix.csv'
features_df.to_csv(csv_file_path, index=False)  # 确保不将自动生成的索引导出到CSV

print(f"归一化后的特征矩阵已导出到 {csv_file_path}")


# %%
model = RGCN(data.num_features, 64, 16, num_relations).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train(model, data, optimizer, epochs=100)

model.eval()
with torch.no_grad():
    embeddings = model(data.x, data.edge_index, data.edge_type).detach().cpu().numpy()

optimal_labels = cluster_and_evaluate(embeddings, cluster_method='ahc')

# %%
# Update the original NetworkX graph with cluster labels
for node, idx in node_mapping.items():
    G.nodes[node]['cluster'] = int(optimal_labels[idx])

# Write the updated graph to a new GEXF file
new_gexf_path = '/home/ta/devnet/bips_pe_202301_time_price_clustered.gexf'
nx.write_gexf(G, new_gexf_path)
# %%
