# %%
import os
import requests
import time
import json

os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

from pycoingecko import CoinGeckoAPI
cg = CoinGeckoAPI()
from datetime import datetime, timedelta
# dt = datetime.strptime('31-12-2012', '%d-%m-%Y')
dt = datetime.now()
delta = timedelta(days=1)

os.makedirs("btc_price", exist_ok=True)
while True:
    dt -= delta
    if os.path.exists(f"btc_price/{dt.strftime('%d-%m-%Y')}.json"):
        print(f"Skipping {dt.strftime('%d-%m-%Y')}")
        continue

    data = cg.get_coin_history_by_id(id='bitcoin', date=dt.strftime('%d-%m-%Y') , localization=False)
    # print(data)
    if data.get("market_data") is None:
        print(f"Failed to fetch {dt.strftime('%d-%m-%Y')}")
        break

    with open(f"btc_price/{dt.strftime('%d-%m-%Y')}.json", "w") as f:
        json.dump(data, f)
    
    print(f"Done {dt.strftime('%d-%m-%Y')}")
# %%