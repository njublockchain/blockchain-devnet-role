from itertools import combinations, product
from collections import Counter
import networkx as nx

def load_multidigraph_from_gexf(gexf_path):
    return nx.read_gexf(gexf_path, node_type=str)

def find_four_node_motifs(G):
    motifs = Counter()
    checked_combinations = set()

    for node_combination in combinations(G.nodes(), 4):  # 改为遍历四个节点的所有组合
        if frozenset(node_combination) in checked_combinations:
            continue  # 如果这个组合已经检查过，跳过

        # 对这四个节点中的每一对节点，检查它们之间可能存在的边
        edge_types_combinations = []
        for u, v in combinations(node_combination, 2):
            if G.has_edge(u, v):
                edge_types = get_all_edge_types(G, u, v)
                for edge_type in edge_types:
                    edge_types_combinations.append((u, v, edge_type))

        # 这里不做>=3的检查，因为我们希望找到所有可能的边组合，包括可能不形成完整四节点结构的情况
        if len(edge_types_combinations) > 0:
            motif = tuple(sorted((edge[2] for edge in edge_types_combinations)))
            motifs[motif] += 1

        checked_combinations.add(frozenset(node_combination))

    return motifs

def get_all_edge_types(G, u, v):
    data = G.get_edge_data(u, v)
    return [edge_data['event_type'] for edge_data in data.values()] if data else []

# 示例使用
gexf_path = '/home/ta/devnet/bips_2023_code_review_time.gexf'
G = load_multidigraph_from_gexf(gexf_path)
motifs = find_four_node_motifs(G)
print(motifs.most_common(10))
