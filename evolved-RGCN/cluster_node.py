# 以给定一个gexf文件，生成节点及其所有属性的csv

import networkx as nx
import pandas as pd

# Load the graph from the GEXF file
gexf_file_path = '/home/ta/devnet/bips_tpe_202312_time_price_clustered.gexf'  # Replace with your actual file path
G = nx.read_gexf(gexf_file_path)

# Create a DataFrame to hold the nodes and their attributes
nodes_data = []
for node, data in G.nodes(data=True):
    node_data = {'node': node}
    node_data.update(data)
    nodes_data.append(node_data)

df = pd.DataFrame(nodes_data)

# Save the DataFrame to a CSV file
csv_file_path = gexf_file_path.replace('.gexf', '_nodes.csv')
df.to_csv(csv_file_path, index=False)

print(f"The CSV file has been saved to {csv_file_path}")
