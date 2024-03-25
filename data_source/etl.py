import os
import gzip
import json
from datetime import datetime
import clickhouse_connect

DB_NAME = 'github'
Table_Name='events'



client = clickhouse_connect.get_client(database=DB_NAME)

# client.command(f'CREATE DATABASE IF NOT EXISTS {DB_NAME}')
sql_CreateTable = f'''CREATE TABLE {DB_NAME}.{Table_Name}
(
    event_type Enum('None' = 0, 'CommitCommentEvent' = 1, 'CreateEvent' = 2, 'DeleteEvent' = 3, 'ForkEvent' = 4,
                    'GollumEvent' = 5, 'IssueCommentEvent' = 6, 'IssuesEvent' = 7, 'MemberEvent' = 8,
                    'PublicEvent' = 9, 'PullRequestEvent' = 10, 'PullRequestReviewCommentEvent' = 11,
                    'PushEvent' = 12, 'ReleaseEvent' = 13, 'SponsorshipEvent' = 14, 'WatchEvent' = 15,
                    'GistEvent' = 16, 'FollowEvent' = 17, 'DownloadEvent' = 18, 'PullRequestReviewEvent' = 19,
                    'ForkApplyEvent' = 20, 'Event' = 21, 'TeamAddEvent' = 22),
    actor_login LowCardinality(String),
    repo_name LowCardinality(String),
    created_at DateTime,
    payload String,
) ENGINE = ReplacingMergeTree ORDER BY (event_type, repo_name, created_at);'''

try:
    client.command(sql_CreateTable)
    print('create table succeed')
except:
    print('creat_table_failed')


# 枚举类型映射
event_type_map = {
    'None':0,
    'CommitCommentEvent': 1,
    'CreateEvent': 2,
    'DeleteEvent': 3,
    'ForkEvent': 4,
    'GollumEvent': 5,
    'IssueCommentEvent': 6,
    'IssuesEvent': 7,
    'MemberEvent': 8,
    'PublicEvent': 9,
    'PullRequestEvent': 10,
    'PullRequestReviewCommentEvent': 11,
    'PushEvent': 12,
    'ReleaseEvent': 13,
    'SponsorshipEvent': 14,
    'WatchEvent': 15,
    'GistEvent': 16,
    'FollowEvent': 17,
    'DownloadEvent': 18,
    'PullRequestReviewEvent': 19,
    'ForkApplyEvent': 20,
    'Event': 21,
    'TeamAddEvent': 22
}

#默认值，数据为空时替换
default_date = datetime.strptime("2000/01/01 00:00:01", "%Y/%m/%d %H:%M:%S")
default_number = 0
default_map_number = 0
default_string = 'default'

def generate_path(base_path):#生成.gz文件访问路径
    path_all = []
    for year in reversed(range(2012, 2024)):
        for month in reversed(range(1, 13)):
            for day in range(1, 32):
                for hour in range(24):
                    link = base_path.format(year, year, month, day, hour)
                    path_all.append(link)
    return path_all

base_path ='/storage/gharchive/{}/{}-{:02d}-{:02d}-{:0>1}.json.gz'
log = f"./etl_log.txt"
failed_path = f"./etl_failed.txt"
path_all = generate_path(base_path)

for file_path in path_all:
    # 解压缩文件
    if os.path.exists(file_path) and not os.path.exists(file_path + '.lock'):

        try:
            with gzip.open(file_path, 'rb') as f_in:
                # 读取并解析JSON数据
                for line in f_in:
                    data = json.loads(line.decode('utf-8'))
                    if data is not None:
                        # 进行数据清洗和格式转换
                        event_type = event_type_map.get(data.get('type'), default_number)

                        if 'actor_attributes' in data.keys():
                            actor_login = data.get('actor_attributes').get('login')
                        elif 'actor' in data.keys():
                            actor_login = data.get('actor').get('login')
                        else:
                            actor_login = ''

                        if 'repo' in data.keys():
                            repo_name = data.get('repo').get('name')
                        elif 'repository' in data.keys():
                            repo_name = data.get('repository', ).get('name', )
                        else:
                            repo_name = ''

                        created_at = datetime.strptime(data.get('created_at', default_date), '%Y-%m-%dT%H:%M:%S%z')

                        payload = str(data.get('payload'))

                        # 数据列表
                        row = [event_type, actor_login, repo_name, created_at, payload]

                        # 列名列表
                        column_names = ['event_type', 'actor_login', 'repo_name', 'created_at', 'payload']

                        # 插入数据
                        client.insert(Table_Name, [row], column_names=column_names)
                        print(event_type, actor_login, repo_name, created_at, '\n')
            # write empty lock file after success
            with open(file_path + '.lock', 'w') as lock_file:
                lock_file.write('')
        except Exception as e:
            print(e)
            with open(log, 'a') as log_info:
                log_info.write(f'{file_path}:{e}\n')
            with open(failed_path, 'a') as failed_file:
                failed_file.write(f'{file_path}\n')

    else:
        with open(log, 'a') as log_info:
            log_info.write(f'{file_path}: not exists\n')
        with open(file_path, 'a') as failed_file:
            failed_file.write(f'{file_path}\n')

