# 统计每个节点每个月（边的属性为“年月日”的created_at）的入度和出度中不同类型（event_type）边的总数

import networkx as nx
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from collections import defaultdict

# 替代 lambda 的可序列化函数
def defaultdict_int():
    return defaultdict(int)

def defaultdict_defaultdict_int():
    return defaultdict(defaultdict_int)

def process_edges(edges):
    in_counts = defaultdict(defaultdict_defaultdict_int)
    out_counts = defaultdict(defaultdict_defaultdict_int)
    for u, v, data in edges:
        event_type = data.get('event_type')
        created_at = data.get('created_at')
        if event_type and created_at:
            year_month = datetime.strptime(created_at, '%Y-%m-%d').strftime('%Y-%m')
            out_counts[u][year_month][event_type] += 1
            in_counts[v][year_month][event_type] += 1
    return in_counts, out_counts


def merge_counts(counts_list):
    def merge_two_dicts(a, b):
        for k, v in b.items():
            if k in a:
                for sub_k, sub_v in v.items():
                    if sub_k in a[k]:
                        for event_type, count in sub_v.items():
                            a[k][sub_k][event_type] += count
                    else:
                        a[k][sub_k] = sub_v
            else:
                a[k] = v
        return a

    total_in_counts = {}
    total_out_counts = {}
    for in_counts, out_counts in counts_list:
        total_in_counts = merge_two_dicts(total_in_counts, in_counts)
        total_out_counts = merge_two_dicts(total_out_counts, out_counts)
    
    return total_in_counts, total_out_counts

def prepare_final_dataframe(total_in_counts, total_out_counts, all_nodes, all_event_types, all_months):
    # 初始化数据列表
    data = []
    # 遍历所有节点和月份
    for node in all_nodes:
        for month in all_months:
            # 初始化每行数据的字典
            row = {"Node": node, "Month": month}
            # 分别添加入度和出度计数
            for event_type in all_event_types:
                row[f"In_{event_type}"] = total_in_counts.get(node, {}).get(month, {}).get(event_type, 0)
                row[f"Out_{event_type}"] = total_out_counts.get(node, {}).get(month, {}).get(event_type, 0)
            data.append(row)
    
    # 将数据转换为DataFrame
    df = pd.DataFrame(data)
    
    # 指定列的顺序
    columns_order = ["Node", "Month"]
    for event_type in sorted(all_event_types):
        columns_order.append(f"In_{event_type}")
    for event_type in sorted(all_event_types):
        columns_order.append(f"Out_{event_type}")
    
    # 调整DataFrame的列顺序
    df = df[columns_order]

    return df

def count_event_types_in_out_parallel(gexf_path, workers=4):
    G = nx.read_gexf(gexf_path)
    if not G.is_directed():
        raise ValueError("Graph must be directed to calculate in-degree and out-degree")

    edges = list(G.edges(data=True))
    chunk_size = len(edges) // workers
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_edges, edges[i:i + chunk_size]) for i in range(0, len(edges), chunk_size)]
        results = [f.result() for f in futures]
        
    total_in_counts, total_out_counts = merge_counts(results)

    # 获取所有唯一的节点、事件类型和月份
    all_nodes = set(total_in_counts.keys()) | set(total_out_counts.keys())
    all_event_types = set()
    all_months = set()
    for counts in (total_in_counts, total_out_counts):
        for node, months in counts.items():
            all_months.update(months.keys())
            for month, events in months.items():
                all_event_types.update(events.keys())

    df = prepare_final_dataframe(total_in_counts, total_out_counts, all_nodes, all_event_types, sorted(all_months))

    return df

# Replace the file path with your actual GEXF file path
gexf_path = '/home/lab0/devnet/rebuilt/bitcoin_2023_non_professional_time.gexf'

# Execute the function
df = count_event_types_in_out_parallel(gexf_path)

# Replace the CSV path with your desired output file path
csv_path = '/home/lab0/devnet/rebuilt/output_bitcoin_non_professional_month.csv'
df.to_csv(csv_path, index=False)

print(f"CSV saved to {csv_path}")
