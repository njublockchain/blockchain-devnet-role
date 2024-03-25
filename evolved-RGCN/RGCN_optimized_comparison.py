import torch
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
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

set_seed()

class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super(RGCN, self).__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations=num_relations)
        self.conv2 = RGCNConv(hidden_channels, out_channels, num_relations=num_relations)

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        edge_type = batch.edge_attr  # 注意这里使用 batch.edge_attr 来获取 edge_type

        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
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

    for u, v, data in G.edges(data=True):
        edge_index.append([node_mapping[u], node_mapping[v]])
        edge_type.append(edge_types[data['event_type']])
        out_counts[u][data['event_type']] += 1
        in_counts[v][data['event_type']] += 1

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long)

    # 构建特征矩阵 节点特征矩阵需要改成每个节点入度出度中不同事件类型(event_type)的数量（InEvent1，InEvent2, ... OutEvent1, OutEvent2,...）
    num_nodes = len(G.nodes())
    num_edge_types = len(edge_types)
    x = torch.zeros((num_nodes, num_edge_types * 2), dtype=torch.float)
    for node, idx in node_mapping.items():
        for etype, count in in_counts[node].items():
            x[idx, edge_types[etype]] = count
        for etype, count in out_counts[node].items():
            x[idx, edge_types[etype] + num_edge_types] = count 

    return Data(x=x.to(device), edge_index=edge_index.to(device), edge_attr=edge_type.to(device)), len(edge_types), G, node_mapping

def adjacency_matrix(edge_index, num_nodes):
    # edge_index: [2, E] 形状的张量，其中E是边的数量
    # num_nodes: 图中的节点数量
    # 创建一个稀疏矩阵，其中边存在的位置为1，其余为0
    
    # indices是edge_index，但需要转换为LongTensor（如果还没转换的话）
    indices = edge_index.long()
    # values是一个大小与边数相等的全1向量
    values = torch.ones(indices.shape[1], device=device)
    # 构造稀疏张量的大小
    size = torch.Size([num_nodes, num_nodes])
    # 创建稀疏张量
    adj_matrix = torch.sparse.FloatTensor(indices, values, size).to(device)
    return adj_matrix

def train(model, loader, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:  # 迭代小批量数据
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)  # 修改模型前向传播调用
            # 假设我们想要使用图重构损失，需要对每个batch计算邻接矩阵
            adj_matrix = adjacency_matrix(batch.edge_index, batch.num_nodes).to(device)
            pred_adj_matrix = torch.sigmoid(torch.mm(out, out.t()))
            loss = F.binary_cross_entropy(pred_adj_matrix, adj_matrix)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {total_loss / len(loader)}')

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

# 将Data对象转换为列表，因为DataLoader期望输入为列表形式
data_list = [data]

# 使用DataLoader
loader = DataLoader(data_list, batch_size=32, shuffle=True)

#---------------------------------
# 将特征矩阵转换为Pandas DataFrame
# 获取所有唯一的事件类型
unique_event_types = sorted(set(nx.get_edge_attributes(G, 'event_type').values()))

# 为每种事件类型生成入度和出度的特征名
feature_columns = ['In_' + etype for etype in unique_event_types] + ['Out_' + etype for etype in unique_event_types]

# 将特征矩阵转换为Pandas DataFrame
# 注意：这里直接使用节点名称作为一列数据，而不是索引
features_np = data.x.cpu().numpy()
nodes = [node for node in G.nodes()]
features_df = pd.DataFrame(features_np, columns=feature_columns)
features_df.insert(0, 'Node', nodes)  # 在第一列插入节点名称列，并命名为"Node"

# 导出DataFrame到CSV文件
csv_file_path = '/home/ta/devnet/rgcn_node_features_matrix.csv'
features_df.to_csv(csv_file_path, index=False)  # 不将自动生成的索引导出到CSV文件中

print(f"特征矩阵已导出到 {csv_file_path}")

#---------------------------------
# 归一化特征矩阵
scaler = MinMaxScaler()
features_np = data.x.cpu().numpy()  # 假设data.x是从你的图G中提取的特征矩阵
features_normalized = scaler.fit_transform(features_np)

# 转换归一化后的特征为Pandas DataFrame，并为节点列指定列名称"Node"
nodes = list(G.nodes())  # 获取节点列表
features_df = pd.DataFrame(features_normalized, columns=feature_columns, index=nodes)
features_df.reset_index(inplace=True, drop=False)  # 将节点从索引转换为列，不保留原索引
features_df.rename(columns={'index': 'Node'}, inplace=True)  # 重命名该列为"Node"

# 导出DataFrame到CSV文件，不包含索引列
csv_file_path = '/home/ta/devnet/rgcn_normalized_features_matrix.csv'
features_df.to_csv(csv_file_path, index=False)  # 确保不将自动生成的索引导出到CSV


print(f"归一化后的特征矩阵已导出到 {csv_file_path}")

#---------------------------------
model = RGCN(data.num_features, 64, 16, num_relations).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train(model, loader, optimizer, epochs=30)

model.eval()
with torch.no_grad():
    # Create a data object that contains all necessary attributes
    batch_data = Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_type).to(device)
    # Now, you should modify the model call to pass the entire batch_data
    embeddings = model(batch_data).detach().cpu().numpy()

optimal_labels = cluster_and_evaluate(embeddings, cluster_method='ahc')


# Update the original NetworkX graph with cluster labels
for node, idx in node_mapping.items():
    G.nodes[node]['cluster'] = int(optimal_labels[idx])

# Write the updated graph to a new GEXF file
new_gexf_path = '/home/ta/devnet/bips_202301_time_price_optimized_clustered.gexf'
nx.write_gexf(G, new_gexf_path)
