import networkx as nx
from datetime import datetime, timezone

def convert_timestamps(gexf_path, output_path):
    # 读取GEXF文件来创建有向图
    G = nx.read_gexf(gexf_path)

    # 检查图是否为MultiDiGraph
    if isinstance(G, nx.MultiDiGraph):
        # 遍历图中的所有边，包括边的键
        for u, v, key, data in G.edges(keys=True, data=True):
            # 获取'created_at'属性，如果不存在则跳过
            if 'created_at' in data:
                timestamp = data['created_at']
                # 将浮点数时间戳转换为真实的年月日时间格式
                real_date = datetime.fromtimestamp(float(timestamp), timezone.utc).strftime('%Y-%m-%d')
                # 使用正确的方法更新边属性
                G[u][v][key]['created_at'] = real_date
    else:
        # 对于非MultiDiGraph，使用原始方法
        for u, v, data in G.edges(data=True):
            if 'created_at' in data:
                timestamp = data['created_at']
                real_date = datetime.fromtimestamp(float(timestamp), timezone.utc).strftime('%Y-%m-%d')
                G[u][v]['created_at'] = real_date

    # 将修改后的图写回到新的GEXF文件
    nx.write_gexf(G, output_path)

# 调用函数
gexf_path = '/home/lab0/devnet/rebuilt/bitcoin_2023_non_professional.gexf'  # 你的GEXF文件路径
output_path = '/home/lab0/devnet/rebuilt/bitcoin_2023_non_professional_time.gexf'  # 输出文件的路径
convert_timestamps(gexf_path, output_path)