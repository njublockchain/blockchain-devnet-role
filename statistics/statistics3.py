# 统计每个节点每天（边的属性为“年月日”的created_at）的入度和出度中不同类型（event_type）边的总数
import networkx as nx
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from collections import defaultdict

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
            # 修改这里以按天统计
            date = datetime.strptime(created_at, '%Y-%m-%d').strftime('%Y-%m-%d')
            out_counts[u][date][event_type] += 1
            in_counts[v][date][event_type] += 1
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

def prepare_final_dataframe(total_in_counts, total_out_counts, all_nodes, all_event_types, all_dates):
    data = []
    for node in all_nodes:
        for date in all_dates:
            row = {"Node": node, "Date": date}
            for event_type in all_event_types:
                in_count = total_in_counts.get(node, {}).get(date, {}).get(event_type, 0)
                out_count = total_out_counts.get(node, {}).get(date, {}).get(event_type, 0)
                row[f"In_{event_type}"] = in_count
                row[f"Out_{event_type}"] = out_count
            data.append(row)
    df = pd.DataFrame(data)
    columns_order = ["Node", "Date"] + [f"In_{et}" for et in sorted(all_event_types)] + [f"Out_{et}" for et in sorted(all_event_types)]
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

    # Collect all unique nodes, event types, and dates
    all_nodes = set(total_in_counts.keys()) | set(total_out_counts.keys())
    all_event_types = set()
    all_dates = set()
    for counts in (total_in_counts, total_out_counts):
        for node, dates in counts.items():
            all_dates.update(dates.keys())
            for date, events in dates.items():
                all_event_types.update(events.keys())

    df = prepare_final_dataframe(total_in_counts, total_out_counts, all_nodes, all_event_types, sorted(all_dates))

    return df


# 使用你的文件路径替换这里的路径
gexf_path = '/home/lab0/devnet/rebuilt/bitcoin_2023_non_professional_time.gexf'

# 执行函数
df = count_event_types_in_out_parallel(gexf_path)

# 替换成你想要的输出文件路径
csv_path = '/home/lab0/devnet/rebuilt/output_bitcoin_non_professional_day.csv'
df.to_csv(csv_path, index=False)

print(f"CSV saved to {csv_path}")
