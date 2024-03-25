# 功能和流程
# 加载多重有向图 (load_multidigraph_from_gexf): 从GEXF文件中加载图，允许多重边（即两个节点之间可以有多条边，边可以有不同的类型）。
# 寻找三节点motif (find_three_node_motifs_optimized): 这个函数的目标是找出图中所有三节点构成的motif及其出现的次数。它通过迭代每个节点和其邻居节点的组合来实现，计算这些组合中所有可能的边类型组合。
# 获取边的类型 (get_all_edge_types): 给定两个节点，此函数返回它们之间所有边的类型。
# 统计和输出：统计最常见的100种motif，并将结果输出到CSV文件中。
# 对于三节点motif的定义：该代码统计的是基于边类型的组合，而不是基于节点连接方式的传统图论motif。也就是说，它更关注边的类型组合，而不是节点间的连接结构。
import networkx as nx
from itertools import combinations, product
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from itertools import permutations
import os

def load_multidigraph_from_gexf(gexf_path):
    """
    从GEXF文件中加载多重有向图。
    """
    return nx.read_gexf(gexf_path, node_type=str)

def find_three_node_motifs_optimized(G):
    """
    优化的方法找到所有三节点motif的出现次数
    """
    motifs = Counter()
    checked_combinations = set()  # 记录已检查的节点组合，避免重复计算

    for node in G.nodes():
        neighbors = set(G.neighbors(node))  # 获取当前节点的所有邻居
        for u, v in combinations(neighbors, 2):  # 对邻居节点进行组合
            if frozenset([node, u, v]) not in checked_combinations:  # 检查是否已经计算过这个组合
                # 获取每对节点间所有可能的边类型组合
                edge_types_combinations = []
                for a, b in product([node, u, v], repeat=2):
                    if a != b and G.has_edge(a, b):
                        edge_types = get_all_edge_types(G, a, b)
                        for edge_type in edge_types:
                            edge_types_combinations.append((a, b, edge_type))

                if len(edge_types_combinations) >= 2:  # 至少存在两种不同的边
                    # 统计这个组合的motif
                    motif = tuple(sorted((edge[2] for edge in edge_types_combinations)))  # 按类型排序
                    motifs[motif] += 1
                checked_combinations.add(frozenset([node, u, v]))  # 标记为已检查

    return motifs

def get_all_edge_types(G, u, v):
    """
    获取两个节点之间所有边的类型。
    """
    data = G.get_edge_data(u, v)
    if data:
        return [edge_data['event_type'] for edge_data in data.values()]
    return []

def draw_motif_instance(G, nodes, save_path):
    """
    绘制并保存图中特定节点构成的子图。
    Args:
        G: 网络图
        nodes: 构成motif的节点列表
        save_path: 保存图像的路径
    """
    subG = G.subgraph(nodes)
    pos = nx.spring_layout(subG)
    plt.figure(figsize=(30, 8))

    # 绘制节点和边
    nx.draw(subG, pos, with_labels=False, edge_color="black", node_size=700, node_color="lightblue")

    # 准备自定义节点标签
    custom_node_labels = {node: node.split('/')[-1] for node in subG.nodes()}  # 从URL中提取用户名
    nx.draw_networkx_labels(subG, pos, labels=custom_node_labels, font_size=8)

    # 准备自定义边标签，确保所有键都存在
    custom_edge_labels = {}
    for u, v, data in subG.edges(data=True):
        edge_type = data.get('event_type', 'N/A')  # 使用 'N/A' 如果属性不存在
        created_at = data.get('created_at', 'N/A')  # 使用 'N/A' 如果属性不存在
        custom_edge_labels[(u, v)] = f"{edge_type}\n{created_at}"

    # 绘制边的标签
    nx.draw_networkx_edge_labels(subG, pos, edge_labels=custom_edge_labels, font_size=8)

    plt.savefig(save_path, format='PNG', bbox_inches='tight')
    plt.close()

def find_motif_instances(G, motif, limit, output_dir='/home/ta/devnet/motif_instances'):
    """
    找到并绘制符合特定motif的实例。
    Args:
        G: 网络图
        motif: 要寻找的motif，一个包含边类型的元组
        limit: 最大绘制的实例数量
    """
    # 转换motif为边的集合，因为每个边类型可能对应多个不同的节点对
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    count = 0
    # 遍历所有可能的节点三元组
    for nodes in permutations(G.nodes(), r=3):
        # 存储这三个节点间的所有边类型
        edge_types = []
        for u, v in permutations(nodes, r=2):
            # 对于每对节点，获取它们之间所有边的类型
            if G.has_edge(u, v):
                for key in G[u][v]:
                    edge_types.append(G[u][v][key]['event_type'])
        
        # 如果边类型集合中包含motif中定义的所有类型，则视为找到一个motif实例
        if all(motif.count(event) <= edge_types.count(event) for event in motif):
            count += 1
            save_path = f'{output_dir}/motif_instance_{count}.png'
            draw_motif_instance(G, nodes, save_path)
            if count >= limit:
                break
    
    print(f'Found and saved {count} instances of the motif.')


# 示例使用
gexf_path = '/home/ta/devnet/bips_2023_non_professional_time.gexf'
G = load_multidigraph_from_gexf(gexf_path)
motifs = find_three_node_motifs_optimized(G)
most_common_motifs = motifs.most_common(50)  # 获取出现次数最多的5种motif
# print("Most common three-node motifs with multiple edge types:", most_common_motifs)

# 转换most_common_motifs为DataFrame
df_motifs = pd.DataFrame(most_common_motifs, columns=['Motif', 'Count'])

# 导出DataFrame到CSV文件
csv_file_path = '/home/ta/devnet/bips_2023_non_professional_time_motifs.csv'
df_motifs.to_csv(csv_file_path, index=False)

print(f"Motifs已经成功导出到 {csv_file_path}")

# -------------------------------

# 为每个motif分配一个编号
motif_to_id = {motif: f"M{i+1}" for i, (motif, _) in enumerate(most_common_motifs)}
# 反转映射，用于图例
id_to_motif = {v: k for k, v in motif_to_id.items()}

# 更新most_common_motifs，将motif替换为编号
most_common_motifs_id = [(motif_to_id[motif], count) for motif, count in most_common_motifs]


# 提取编号和它们的计数
ids, counts = zip(*most_common_motifs_id)

# 调整图的大小和条形的高度
plt.figure(figsize=(20, 20))  # 增大图的尺寸
bar_height = 0.4  # 调整条形的高度，减小这个值可以增加条形之间的间距

# 绘制条形图，使用调整后的条形高度
plt.barh(range(len(ids)), counts, color='skyblue', height=bar_height)
plt.yticks(range(len(ids)), ids)

# 调整y轴标签的字体大小，如果需要
plt.tick_params(axis='y', labelsize=10)

plt.xlabel('Count')
plt.title('Most Common Three-Node Motifs with Multiple Edge Types')

# 保存图像到文件
image_file_path = '/home/ta/devnet/motifs_distribution_with_id.png'
plt.savefig(image_file_path, bbox_inches='tight')  # 使用bbox_inches='tight'确保所有标签都包含在保存的图像中

print(f"Motif分布图（已调整间距）已经成功保存到 {image_file_path}")

# # 使用示例
# motif_to_find = ('IssueCommentEvent', 'IssuesEvent', 'IssuesEvent')  # 定义要寻找的motif
# find_motif_instances(G, motif_to_find, 10)