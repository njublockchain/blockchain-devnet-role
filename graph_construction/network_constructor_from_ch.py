# %%
import os
from datetime import datetime
from pymongo import MongoClient
from tqdm import tqdm
import networkx
import pickle

from bigquery_base import fetch_repo
import clickhouse_connect

repo_urls = [
    "https://api.github.com/repos/bitcoin/bitcoin",
    "https://api.github.com/repos/bitcoin/bips",
]

repo = repo_urls[0]

os.makedirs("ch_graphs", exist_ok=True)

G = networkx.MultiDiGraph()
year = 2023
import orjson as json

# %%

os.makedirs(f"prelude/clickhouse/repos", exist_ok=True)
os.makedirs(f"prelude/clickhouse/actors", exist_ok=True)

with clickhouse_connect.get_client() as client:
    if os.path.exists(f"prelude/clickhouse/repos/{repo.split('/')[-1]}.pkl"):
        with open(f"prelude/clickhouse/repos/{repo.split('/')[-1]}.pkl", "rb") as f:
            results = pickle.load(f)
    else:
        q = client.query(
            f"SELECT * FROM github.events0310 WHERE target='{repo_urls[0]}' and toYear(created_at) = 2023"
        )
        results = []
        for row in q.result_rows:
            result = {col: row[i] for i, col in enumerate(q.column_names)}
            results.append(result)
        # print(results)

        with open(f"prelude/clickhouse/repos/{repo.split('/')[-1]}.pkl", "wb") as f:
            pickle.dump(results, f)

    actor_urls = set()
    for edge in tqdm(results):
        if edge["created_at"].year != year:
            continue
        G.add_edge(
            edge["actor"],
            edge["target"],
            event_type=edge["event_type"],
            key=json.loads(edge["data"])["id"],
            created_at=edge["created_at"].timestamp(),
        )
        url = edge["actor"]
        actor_urls.add((url))
    
    import multiprocessing.dummy as mp

    # for actor_url in tqdm(list(actor_urls)):
    def work(actor_url):
        with clickhouse_connect.get_client() as client:
            if os.path.exists(f"prelude/clickhouse/actors/{actor_url.split('/')[-1]}.pkl"):
                print(f"Loading {actor_url}")
                with open(f"prelude/clickhouse/actors/{actor_url.split('/')[-1]}.pkl", "rb") as f:
                    results = pickle.load(f)
            else:
                print(f"Querying {actor_url}")
                q = client.query(
                    f"SELECT * FROM github.events0310 WHERE actor='{actor_url}' and toYear(created_at) = 2023"
                )
                results = []
                for row in q.result_rows:
                    result = {col: row[i] for i, col in enumerate(q.column_names)}
                    results.append(result)

                with open(f"prelude/clickhouse/actors/{actor_url.split('/')[-1]}.pkl", "wb") as f:
                    pickle.dump(results, f)

            for sub_edge in results:
                if sub_edge["created_at"].year != year:
                    continue
                G.add_edge(
                    url,
                    sub_edge["target"],
                    event_type=sub_edge["event_type"],
                    key=json.loads(sub_edge["data"])["id"],
                    created_at=sub_edge["created_at"].timestamp(),
                )
    with mp.Pool(32) as p:
        list(tqdm(p.imap(work, list(actor_urls)), total=len(actor_urls)))
# %%

for node in G.nodes:
    if str(node).startswith("https://api.github.com/repos/"):
        G.nodes[node]["type"] = "repo"
    elif str(node).startswith("https://api.github.com/users/"):
        G.nodes[node]["type"] = "user"
    else:
        print(f"Unknown type for {node}")

os.makedirs(f"ch_graphs", exist_ok=True)

# write with pickle
with open(f"ch_graphs/{repo.split('/')[-1]}_{year}.pkl", "wb") as f:
    pickle.dump(G, f)

networkx.write_gexf(G, f"ch_graphs/{repo.split("/")[-1]}_{year}.gexf")
