# %%
from datetime import datetime
from pymongo import MongoClient
from tqdm import tqdm
import networkx
import pickle

from bigquery_base import fetch_repo


repo_urls = ["https://github.com/bitcoin/bitcoin", "https://github.com/bitcoin/bips"]

repo = "bitcoin/bips"

G = networkx.MultiDiGraph()
year = 2023

with open(f"repos/{repo}.pkl", "rb") as f:
    all_rows = pickle.load(f)
    actors = set()
    for edge in tqdm(all_rows):
        if edge["created_at"].year != year:
            continue
        G.add_edge(edge["actor"]["url"], edge["repo"]["url"], event_type=edge["type"], key=edge["id"], created_at=edge["created_at"].timestamp(), org=edge["org"])

        url = edge["actor"]["url"]
        login = edge["actor"]["login"]
        actors.add((login, url))

    for login, url in tqdm(list(actors)):
        with open(f"actors/{login}.pkl", "rb") as f:
            all_rows = pickle.load(f)
            for sub_edge in all_rows:
                if sub_edge["created_at"].year != year:
                    continue
                G.add_edge(url, sub_edge["repo"]["url"], event_type=sub_edge["type"], key=sub_edge["id"], created_at=sub_edge["created_at"].timestamp(), org=edge["org"])

for node in G.nodes:
    if str(node).startswith("https://api.github.com/repos/"):
        G.nodes[node]["type"] = "repo"
    elif str(node).startswith("https://api.github.com/users/"):
        G.nodes[node]["type"] = "user"
    else:
        print(f"Unknown type for {node}")

# write with pickle
with open(f"{repo.split('/')[-1]}_{year}.pkl", "wb") as f:
    pickle.dump(G, f)

networkx.write_gexf(G, f"{repo.split("/")[-1]}_{year}.gexf")
