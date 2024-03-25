import gzip
import json
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

save_path = "format.json"


def generate_path(base_path):
    path_all = []
    date = datetime(2011, 2, 12, 0)
    end = datetime(2017, 1, 1, 0)
    delta = timedelta(hours=1)
    while date < end:
        link = base_path.format(date.year, date.year, date.month, date.day, date.hour)
        path_all.append(link)
        date += delta
    return path_all


types = []
# formats {type : [format1, format2, ...] }
formats = {
    "None": [],
    "CommitCommentEvent": [],
    "CreateEvent": [],
    "DeleteEvent": [],
    "ForkEvent": [],
    "GollumEvent": [],
    "IssueCommentEvent": [],
    "IssuesEvent": [],
    "MemberEvent": [],
    "PublicEvent": [],
    "PullRequestEvent": [],
    "PullRequestReviewCommentEvent": [],
    "PushEvent": [],
    "ReleaseEvent": [],
    "SponsorshipEvent": [],
    "WatchEvent": [],
    "GistEvent": [],
    "FollowEvent": [],
    "DownloadEvent": [],
    "PullRequestReviewEvent": [],
    "ForkApplyEvent": [],
    "Event": [],
    "TeamAddEvent": [],
}


def get_format(file_path):
    global types
    global formats
    try:
        with gzip.open(file_path, "rb") as f_in:
            print(file_path)
            for line in f_in:
                data = json.loads(line.decode("utf-8"))
                # print(f'{data.keys()}\n')
                if data is not None:
                    event_type = data.get("type")
                    flag = 0
                    for format in formats.get(event_type):
                        if data.keys() == format.keys():
                            flag = 1
                    if flag == 0:
                        print(f"{event_type} {data.keys()}\n")
                        formats.get(event_type).append(data)

    except Exception as e:
        print(e)


base_path = "/storage/gharchive/{}/{}-{:02d}-{:02d}-{:0>1}.json.gz"
path_all = generate_path(base_path)

# Process files using ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=10) as executor:
    executor.map(get_format, path_all)

with open(save_path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(formats, f, indent=2, sort_keys=True, ensure_ascii=False)
