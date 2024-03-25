import os
import requests
import time

os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
now = int(time.time())
url = f"https://query1.finance.yahoo.com/v7/finance/download/BTC-USD?period1=0&period2={now}&interval=1d&events=history&includeAdjustedClose=true"
headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Cookie": "gam_id=y-hXkBIaBE2uIzU5OIOYo6pWKZx1TYtC2R~A; GUCS=AVzIrbZP; GUC=AQEBCAFl7-hmGUIcWQRk&s=AQAAAIpnAOFj&g=Ze6jqw; A1=d=AQABBH0-JGUCEJOGn9DCH7s1ldHqzz_BkkYFEgEBCAHo72UZZtwu0iMA_eMBAAcIfT4kZT_BkkY&S=AQAAAp9TmkAEvfEjc63qAwObMaI; A3=d=AQABBH0-JGUCEJOGn9DCH7s1ldHqzz_BkkYFEgEBCAHo72UZZtwu0iMA_eMBAAcIfT4kZT_BkkY&S=AQAAAp9TmkAEvfEjc63qAwObMaI; A1S=d=AQABBH0-JGUCEJOGn9DCH7s1ldHqzz_BkkYFEgEBCAHo72UZZtwu0iMA_eMBAAcIfT4kZT_BkkY&S=AQAAAp9TmkAEvfEjc63qAwObMaI; cmp=t=1710138261&j=0&u=1YNN; gpp=DBAA; gpp_sid=-1; PRF=t%3DBTC-USD; axids=gam=y-hXkBIaBE2uIzU5OIOYo6pWKZx1TYtC2R~A&dv360=eS1yYzZIU21sRTJ1R3BHTklYSE45TEc5WU8xZ3VTUndaWH5B&ydsp=y-rgDKKLtE2uJpT5aIdzIPNIosGW635gGH~A&tbla=y-9wK.YTVE2uIm.vhhUfkUqnEaYrDLUvsq~A; tbla_id=3c08079c-7963-4916-b5e6-3e889b079bb7-tuctc1dc3fb; __eoi=ID=da8679c0aeaf175a:T=1710138279:RT=1710138840:S=AA-Afja92bRaU9x5EJw69_XJmbD1",
    "Referer": "https://finance.yahoo.com/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}
raw = requests.get(url, headers=headers).text
with open("btc_price.csv", "w") as f:
    f.write(raw)
