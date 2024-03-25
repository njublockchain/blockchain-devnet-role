from glob import glob
from google.cloud import bigquery
import os
from tqdm import tqdm
import json
import pickle
import time
import google
import requests


from bigquery_base import fetch_actor, fetch_repo

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/lab0/.config/gcloud/application_default_credentials.json"

# Construct a BigQuery client object.
PROJECT = "nomadic-line-416907"

def main():
    import multiprocessing.dummy as mt

    def unpack(args):
        return fetch_actor(*args)
    
    entrances = ["bitcoin/bips"]
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
