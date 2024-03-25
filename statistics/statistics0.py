# 统计每个月（边的属性为“年月日”的created_at）有多少不同类型的事件（边的属性为event_type）的总数，并输出csv文件。

import networkx as nx
import pandas as pd
from datetime import datetime
from collections import defaultdict

def read_gexf_and_count_events(gexf_path):
    # 读取GEXF文件
    G = nx.read_gexf(gexf_path)

    # 初始化计数器
    event_counts = defaultdict(lambda: defaultdict(int))

    # 处理每条边
    for _, _, data in G.edges(data=True):
        event_type = data.get('event_type')
        created_at = data.get('created_at')
        if event_type and created_at:
            # 将日期转换为年-月格式
            month_year = datetime.strptime(created_at, '%Y-%m-%d').strftime('%Y-%m')
            # 计数
            event_counts[month_year][event_type] += 1

    return event_counts

def save_event_counts_to_csv(event_counts, csv_path):
    # 准备数据转换为DataFrame
    data = []
    for month_year, events in event_counts.items():
        row = {'Month_Year': month_year}
        row.update(events)
        data.append(row)

    # 创建DataFrame
    df = pd.DataFrame(data)
    # 用0填充NaN值
    df.fillna(0, inplace=True)
    # 按日期排序
    df.sort_values('Month_Year', inplace=True)
    # 将结果保存到CSV
    df.to_csv(csv_path, index=False)

# 指定GEXF文件路径和CSV输出路径
gexf_path = '/home/lab0/devnet/rebuilt/bitcoin_2023_non_professional_time.gexf'
csv_path = '/home/lab0/devnet/rebuilt/output_bitcoin_non_professional_overview.csv'

# 处理GEXF文件并计数
event_counts = read_gexf_and_count_events(gexf_path)
# 保存结果到CSV
save_event_counts_to_csv(event_counts, csv_path)

print(f"CSV file has been saved to {csv_path}")
