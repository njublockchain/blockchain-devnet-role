import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import HeteroData
import networkx as nx
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 步骤 1: 加载GEXF图文件

def load_heterogeneous_graph(gexf_path, device):
    G = nx.read_gexf(gexf_path)
    data = HeteroData()

    # 步骤 1: 为每个节点分配一个唯一的数字索引
    node_mapping = {node_id: idx for idx, node_id in enumerate(G.nodes())}

    # 初始化用于收集不同类型边的边索引的字典
    edge_indices = {}

    for u, v, edge_attr in G.edges(data=True):
        edge_type = edge_attr['event_type']
        if edge_type not in edge_indices:
            edge_indices[edge_type] = [[], []]  # 初始化源和目标列表
        u_idx, v_idx = node_mapping[u], node_mapping[v]
        edge_indices[edge_type][0].append(u_idx)
        edge_indices[edge_type][1].append(v_idx)

    # 步骤 2: 使用数字索引创建边索引张量，并将其添加到data对象
    for edge_type, (src, dst) in edge_indices.items():
        data['node', edge_type, 'node'].edge_index = torch.tensor([src, dst], dtype=torch.long).to(device)

    # 假设所有节点都有相同的特征维度，这里用单位矩阵作为示例
    num_nodes = len(G.nodes())
    data['node'].x = torch.eye(num_nodes, dtype=torch.float).to(device)

    return data


# 步骤 2: 定义异构图神经网络模型
class HeteroGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, metadata):
        super().__init__()
        self.convs = torch.nn.ModuleDict({
            "_".join(edge_type): RGCNConv(in_channels, out_channels, num_relations=len(metadata[1]))
            for edge_type in metadata[1]
        })

    def forward(self, x, edge_index_dict):
        for edge_type, conv in self.convs.items():
            edge_index = edge_index_dict[edge_type].edge_index
            x = conv(x, edge_index)
        return x

def train(model, data, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data['node'].x, data.edge_index_dict)  # 假设模型输出是每个节点的嵌入
        # 示例：使用均方误差作为损失函数
        loss = F.mse_loss(out, torch.randn_like(out))  # 假设存在随机目标嵌入
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")


def extract_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        embeddings = model(data['node'].x, data.edge_index_dict)
    return embeddings.cpu().numpy()

def cluster_and_evaluate(embeddings):
    best_score = -1
    best_k = 0
    for k in range(2, 11):  # 尝试不同的聚类数量
        clustering = AgglomerativeClustering(n_clusters=k)
        labels = clustering.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
    print(f"Best number of clusters: {best_k}, Silhouette Score: {best_score}")
    return best_k, best_labels

# 加载数据
data = load_heterogeneous_graph('/home/lab0/devnet/rebuilt/bips_2023_issue_tracking_time.gexf', device)

# 初始化模型和优化器
model = HeteroGNN(in_channels=16, out_channels=16, metadata=data.metadata()).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型
train(model, data, optimizer, epochs=100)

# 提取节点嵌入
embeddings = extract_embeddings(model, data)

# 执行聚类并选择最优聚类数量
best_k, best_labels = cluster_and_evaluate(embeddings)