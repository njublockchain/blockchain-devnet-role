import networkx as nx
from datetime import datetime
import os

def split_graph_by_month(gexf_path, output_dir):
    # 读取 GEXF 文件创建图
    G = nx.read_gexf(gexf_path)

    # 存储每个月的子图
    monthly_subgraphs = {}

    # 遍历图中的每条边
    for u, v, data in G.edges(data=True):
        create_at = data['created_at']  # 假设 create_at 格式为 'YYYY-MM-DD'
        month_year = datetime.strptime(create_at, '%Y-%m-%d').strftime('%Y-%m')  # 转换为 'YYYY-MM'

        if month_year not in monthly_subgraphs:
            monthly_subgraphs[month_year] = nx.MultiDiGraph()  # 对于每个月份创建一个新的子图
        
        monthly_subgraphs[month_year].add_node(u, **G.nodes[u])
        monthly_subgraphs[month_year].add_node(v, **G.nodes[v])
        monthly_subgraphs[month_year].add_edge(u, v, **data)

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 为每个月份的子图写入 GEXF 文件
    for month_year, subgraph in monthly_subgraphs.items():
        nx.write_gexf(subgraph, os.path.join(output_dir, f"{month_year}.gexf"))

    print(f"Finished splitting the graph into monthly subgraphs in {output_dir}.")

# 使用示例
gexf_path = "/home/ta/devnet/bips_2023_time_price.gexf"  # 更改为你的 GEXF 文件路径
output_dir = "/home/ta/devnet/bips_2023_time_price"  # 更改为你的输出目录路径
split_graph_by_month(gexf_path, output_dir)
