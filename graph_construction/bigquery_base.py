import glob
from google.cloud import bigquery
import os
import requests
from tqdm import tqdm
import json
import pickle
import time
import google

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/lab0/.config/gcloud/application_default_credentials.json"

# Construct a BigQuery client object.
PROJECT = "nomadic-line-416907"
client = bigquery.Client(project=PROJECT)
adapter = requests.adapters.HTTPAdapter(
    pool_connections=128, pool_maxsize=128, max_retries=3
)
client._http.mount("https://", adapter)
client._http._auth_request.session.mount("https://", adapter)


def fetch_repo(repo: str):
    if os.path.exists(f"repos/{repo}.pkl"):
        print(f"Skipping {repo}")
        with open(f"repos/{repo}.pkl", "rb") as f:
            all_rows = pickle.load(f)
        return all_rows

    os.makedirs(f"repos/{repo.split('/')[0]}", exist_ok=True)
    all_rows = []

    queries = []
    while True:
        for year in range(2011, 2024):
            query = f"""
                        SELECT type, public, repo, actor, org, created_at, id
                        FROM githubarchive.year.{year}
                        WHERE actor.url='https://api.github.com/repos/{repo}'
                    """
            q = client.query(query)
            queries.append(q)
        try:
            results = [q.result() for q in queries]
            for rows in results:
                all_rows.extend(list(rows))
            break
        except google.api_core.exceptions.Forbidden as e:
            print(f"Failed to fetch {repo} @ {year}: {e}")
            continue

    for f in glob(f"repos/{repo}.20*.pkl"):
        os.remove(f)

    with open(f"repos/{repo}.pkl", "wb") as f:
        pickle.dump([dict(row.items()) for row in all_rows], f)
    print(f"Done {repo}")
    return all_rows


def fetch_actor(login: str, actor_url: str):
    if os.path.exists(f"actors/{login}.pkl"):
        print(f"Skipping {login}")
        with open(f"actors/{login}.pkl", "rb") as f:
            all_rows = pickle.load(f)
        return all_rows

    all_rows = []

    queries = []
    while True:
        for year in range(2011, 2024):
            query = f"""
                        SELECT type, public, repo, actor, org, created_at, id
                        FROM githubarchive.year.{year}
                        WHERE actor.url='{actor_url}'
                    """
            q = client.query(query)
            queries.append(q)
        try:
            results = [q.result() for q in queries]
            for rows in results:
                all_rows.extend(list(rows))
            break
        except google.api_core.exceptions.Forbidden as e:
            print(f"Failed to fetch {actor_url} @ {year}: {e}")
            continue

    os.makedirs(f"actors", exist_ok=True)
    with open(f"actors/{login}.pkl", "wb") as f:
        pickle.dump([dict(row.items()) for row in all_rows], f)

    print(f"Done {login}")
    return all_rows


def main():
    import multiprocessing.dummy as mt
    def unpack(args):
        return fetch_actor(*args)

    entrances = ["bitcoin/bitcoin"]

    with mt.Pool(64) as p:
        for repo in entrances:
            all_rows = fetch_repo(repo)

            actors = set()
            for row in all_rows:
                if row["actor"]["login"] == "":
                    print(row["actor"])
                actors.add((row["actor"]["login"], row["actor"]["url"]))
            print(f"Found {len(actors)} actors on {repo}")
            actors = sorted(list(actors))
            list(tqdm(p.imap(unpack, actors), total=len(actors)))


if __name__ == "__main__":
    main()
