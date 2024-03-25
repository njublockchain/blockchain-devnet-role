import networkx as nx
from concurrent.futures import ProcessPoolExecutor

def read_gexf(path):
    """
    读取单个GEXF文件并返回图对象。
    """
    return nx.read_gexf(path)

def merge_identical_gexf_graphs(gexf_paths):
    """
    使用多进程并行读取GEXF文件，并合并成一个图。
    """
    merged_graph = nx.MultiDiGraph()  # 创建一个新的MultiDiGraph来存储合并后的结果
    
    with ProcessPoolExecutor() as executor:
        # 并行读取GEXF文件
        graphs = list(executor.map(read_gexf, gexf_paths))
        
    # 合并图
    for graph in graphs:
        merged_graph = nx.compose(merged_graph, graph)
    
    return merged_graph

# 你的GEXF文件路径列表
gexf_paths = [
    '/home/lab0/devnet/rebuilt/bitcoin_2023_code_contrib_time.gexf',
    '/home/lab0/devnet/rebuilt/bitcoin_2023_code_review_time.gexf',
    '/home/lab0/devnet/rebuilt/bitcoin_2023_issue_tracking_time.gexf',
    '/home/lab0/devnet/rebuilt/bitcoin_2023_non_professional_time.gexf'
]

# 合并图
merged_graph = merge_identical_gexf_graphs(gexf_paths)

# 保存合并后的图到新的GEXF文件中
nx.write_gexf(merged_graph, '/home/lab0/devnet/rebuilt/bitcoin_2023_time.gexf')
