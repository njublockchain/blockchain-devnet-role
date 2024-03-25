# 统计不同图的密度、直径、最短路径等网络属性指标
import pandas as pd
import networkx as nx
import os

# Replace with the path to your GEXF files directory
directory = '/home/ta/devnet/bips_2023_time_price'
output_csv_path = '/home/ta/devnet/network_statistics.csv'

# List to hold all graph statistics
all_graph_stats = []

# Loop through each GEXF file in the directory
for gexf_file in sorted(os.listdir(directory)):
    if gexf_file.endswith('.gexf'):
        # Load the graph
        G = nx.read_gexf(os.path.join(directory, gexf_file))
        
        # Get basic stats
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        density = nx.density(G)
        
        # Convert to undirected for certain calculations
        if nx.is_directed(G):
            G = G.to_undirected()
        
        # Largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        
        diameter = nx.diameter(subgraph)
        avg_path_length = nx.average_shortest_path_length(subgraph)
        
        
        # Degree Assortativity
        assortativity = nx.degree_assortativity_coefficient(G)
        
        # Collect statistics in a dictionary
        graph_stats = {
            'Month_Year': gexf_file.rstrip('.gexf'),
            'Nodes': num_nodes,
            'Edges': num_edges,
            'Density': density,
            'Diameter': diameter,
            'Average Path Length': avg_path_length,
            'Assortativity': assortativity
        }
        
        # Append dictionary to the list
        all_graph_stats.append(graph_stats)

# Create DataFrame from list of dictionaries
df_stats = pd.DataFrame(all_graph_stats)

# Save the DataFrame to CSV
df_stats.to_csv(output_csv_path, index=False)

print(f"Saved network statistics to {output_csv_path}")
