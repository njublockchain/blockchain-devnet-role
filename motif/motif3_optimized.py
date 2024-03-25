import networkx as nx
from itertools import combinations, product
from collections import Counter
import pandas as pd
import os
from tqdm.contrib.concurrent import process_map  # 使用tqdm的process_map代替Pool

def load_multidigraph_from_gexf(gexf_path):
    return nx.read_gexf(gexf_path, node_type=str)

def find_three_node_motifs_for_subset(args):
    G, subset_nodes = args  # 修改这里以适应process_map的参数形式
    motifs = Counter()
    checked_combinations = set()
    for node in subset_nodes:
        neighbors = set(G.neighbors(node))
        for u, v in combinations(neighbors, 2):
            if frozenset([node, u, v]) not in checked_combinations:
                edge_types_combinations = []
                for a, b in product([node, u, v], repeat=2):
                    if a != b and G.has_edge(a, b):
                        edge_types = get_all_edge_types(G, a, b)
                        for edge_type in edge_types:
                            edge_types_combinations.append((a, b, edge_type))

                if len(edge_types_combinations) >= 2:
                    motif = tuple(sorted((edge[2] for edge in edge_types_combinations)))
                    motifs[motif] += 1
                checked_combinations.add(frozenset([node, u, v]))
    return motifs

def get_all_edge_types(G, u, v):
    data = G.get_edge_data(u, v)
    if data:
        return [edge_data['event_type'] for edge_data in data.values()]
    return []

def find_three_node_motifs_optimized(G, num_processes=64):
    nodes = list(G.nodes())
    subset_size = len(nodes) // num_processes
    subsets = [nodes[i:i + subset_size] for i in range(0, len(nodes), subset_size)]
    args = [(G, subset) for subset in subsets]

    # 使用process_map处理每个子集，并添加进度条
    results = process_map(find_three_node_motifs_for_subset, args, max_workers=num_processes, chunksize=1)

    final_motifs = Counter()
    for result in results:
        final_motifs.update(result)
    
    return final_motifs

# 使用示例，调用改造后的find_three_node_motifs_optimized函数
gexf_path = '/home/ta/devnet/bips_2023_time.gexf'
G = load_multidigraph_from_gexf(gexf_path)
motifs = find_three_node_motifs_optimized(G, num_processes=64)

# 将motifs保存到CSV文件
df_motifs = pd.DataFrame(motifs.items(), columns=['Motif', 'Count'])
df_motifs = df_motifs.sort_values(by='Count', ascending=False)  # 按Count降序排序
csv_file_path = '/home/ta/devnet/motifs_optimized.csv'
df_motifs.to_csv(csv_file_path, index=False)

print(f"Motifs已经成功导出到 {csv_file_path}")