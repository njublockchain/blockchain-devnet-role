# 给定一个图文件gexf，统计不同的事件类型（边属性event_type）各自的数量
import networkx as nx

from collections import Counter

# 加载GEXF文件
gexf_path = '/home/ta/devnet/bitcoin_2023_time.gexf'  # 替换为你的GEXF文件路径
G = nx.read_gexf(gexf_path)

# 初始化计数器
event_type_counter = Counter()

# 遍历所有边，统计事件类型
for _, _, edge_data in G.edges(data=True):
    event_type = edge_data.get('event_type', None)
    if event_type:
        event_type_counter[event_type] += 1

# 打印结果
for event_type, count in event_type_counter.items():
    print(f"Event Type: {event_type}, Count: {count}")
