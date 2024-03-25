# 给定一个gexf文件，分别统计每个节点入度和出度中不同类型（event_type）边的总数
import networkx as nx
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

def process_edges(edges):
    in_counts = {}
    out_counts = {}
    for u, v, data in edges:
        event_type = data.get('event_type')
        if event_type:
            if u not in out_counts:
                out_counts[u] = {}
            if event_type not in out_counts[u]:
                out_counts[u][event_type] = 0
            out_counts[u][event_type] += 1

            if v not in in_counts:
                in_counts[v] = {}
            if event_type not in in_counts[v]:
                in_counts[v][event_type] = 0
            in_counts[v][event_type] += 1
    return in_counts, out_counts

def merge_counts(counts_list):
    total_in_counts = {}
    total_out_counts = {}
    for in_counts, out_counts in counts_list:
        for node, counts in in_counts.items():
            if node not in total_in_counts:
                total_in_counts[node] = counts
            else:
                for event_type, count in counts.items():
                    if event_type not in total_in_counts[node]:
                        total_in_counts[node][event_type] = 0
                    total_in_counts[node][event_type] += count
                    
        for node, counts in out_counts.items():
            if node not in total_out_counts:
                total_out_counts[node] = counts
            else:
                for event_type, count in counts.items():
                    if event_type not in total_out_counts[node]:
                        total_out_counts[node][event_type] = 0
                    total_out_counts[node][event_type] += count
                    
    return total_in_counts, total_out_counts

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

    # Now add the in-degree and out-degree for each node
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    # Convert counts to DataFrame
    in_df = pd.DataFrame.from_dict(total_in_counts, orient='index').fillna(0).astype(int)
    in_df.columns = [f'In_{col}' for col in in_df.columns]
    
    out_df = pd.DataFrame.from_dict(total_out_counts, orient='index').fillna(0).astype(int)
    out_df.columns = [f'Out_{col}' for col in out_df.columns]
    
    # Adding total in-degree and out-degree
    in_df['Total_In_Degree'] = in_df.index.map(in_degrees)
    out_df['Total_Out_Degree'] = out_df.index.map(out_degrees)
    
    df = in_df.join(out_df, how='outer').fillna(0).astype(int).reset_index()
    df.rename(columns={'index': 'Node'}, inplace=True)

    return df

# Adjust the file path to your GEXF file
gexf_path = '/home/lab0/devnet/rebuilt/bitcoin_2023_non_professional.gexf'

# Execute the updated function
df = count_event_types_in_out_parallel(gexf_path)

# Saving the DataFrame to a CSV file
csv_path = '/home/lab0/devnet/rebuilt/output_bitcoin_non_professional.csv'
df.to_csv(csv_path, index=False)

print(f"CSV saved to {csv_path}")