from datetime import datetime, timedelta
import os
import requests
from threading import Thread
import re
from requests.adapters import HTTPAdapter, Retry, PoolManager
import time
import threading
import os
from tqdm import tqdm
from urllib.request import urlopen

MAX_THREADS = 30  # 限制最大线程数量


def download_file(url, save_folder, failed_links_file, session):
    try:

        # Use regex to extract year, month, day, and hour from the URL
        match = re.search(r"/(\d{4})-(\d{2})-(\d{2})-(\d{1,2})\.json\.gz", url)

        if match:
            year, month, day, hour = match.groups()

            year_folder = os.path.join(save_folder, year)
            os.makedirs(year_folder, exist_ok=True)

            save_path = os.path.join(
                year_folder, f"{year}-{month}-{day}-{hour}.json.gz"
            )

            # 访问url进行下载
            response = session.get(url, stream=True)
            response.raise_for_status()

            # 获取文件长度
            file_size_str = response.headers["Content-Length"]  # 提取出来的是个数字str
            file_size = int(file_size_str) / 1024 / 1024

            # 判断本地文件存在时
            if os.path.exists(save_path):
                # 获取本地文件大小
                first_byte = os.path.getsize(save_path)
            else:
                # 初始大小为0
                first_byte = 0

            if first_byte >= file_size:
                print(f"文件已经存在{save_path},无需下载")
                return file_size

            # header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
            pbar = tqdm(
                total=file_size,
                initial=first_byte,
                unit="B",
                unit_scale=True,
                desc=url.split("/")[-1],
            )

            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(1024)
            print(f"Downloaded: {url}")
            pbar.close()
            return True
            """
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
            """

        else:
            print(f"Error parsing path elements for {url}: Unexpected path structure.")
            with open(failed_links_file, "a") as failed_file:
                failed_file.write(url + "\n")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {response.status_code}")
        if response.status_code != 404:
            with open(failed_links_file, "a") as failed_file:
                failed_file.write(url + f"{e}\n")


def generate_download_links(base_url):
    download_links = []
    date = datetime(2011, 2, 12, 0)
    end = datetime(2024, 1, 1, 0)
    delta = timedelta(
        hours=1,
    )
    while date < end:
        link = base_url.format(date.year, date.month, date.day, date.hour)
        download_links.append(link)
        date += delta
    return download_links


def main():
    base_url = "https://data.gharchive.org/{}-{:02d}-{:02d}-{:0>1}.json.gz"
    save_folder = "/storage/gharchive"
    os.makedirs(save_folder, exist_ok=True)
    failed_links_file = "./failed_links_exclude_404.txt"
    failed_links_file_404 = "./failed_links_404.txt"
    failed_links_file_incomplete = "./failed_links_incomplete.txt"

    # Create a session with connection pool
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    download_links = generate_download_links(base_url)

    threads = []
    for link in download_links:
        thread = Thread(
            target=download_file, args=(link, save_folder, failed_links_file, session)
        )
        threads.append(thread)
        thread.start()
        time.sleep(0.1)  # 在请求之间引入小的延迟
        if threading.active_count() >= MAX_THREADS:
            thread.join()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
