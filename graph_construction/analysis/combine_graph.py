import networkx
import pickle
G0 = pickle.load(open("bq_graphs/bips_2023.pkl", "rb"))
G1 = pickle.load(open("ch_graphs/bitcoin_2023.pkl", "rb"))

print(type(G0))
print(type(G1))

# To combine graphs that have common nodes, consider compose(G, H) or the method, Graph.update().
G = networkx.compose(G0, G1)

with open("rebuilt/combined_2023.pkl", "wb") as f:
    pickle.dump(G, f)
    print(f"Done writing to combined_2023.pkl")


