import json
import os
import networkx as nx

# 假设 gexf_path 是你的 GEXF 文件路径
gexf_path = '/home/ta/devnet/bitcoin_2023_time.gexf'
# 假设 price_dir 是存储 JSON 文件的目录
price_dir = '/home/ta/devnet/btc_price'

# 读取 GEXF 文件创建图
G = nx.read_gexf(gexf_path)

# 遍历图中的每条边
for u, v, key, data in G.edges(keys=True, data=True):
    created_at = data.get('created_at')
    if created_at:
        # 调整日期格式以匹配 JSON 文件的命名格式
        date_format = f"{created_at[8:10]}-{created_at[5:7]}-{created_at[0:4]}"
        json_file = os.path.join(price_dir, f"{date_format}.json")
        if os.path.exists(json_file):
            with open(json_file, 'r') as file:
                price_data = json.load(file)
                # 假设 "current_price"-"usd" 下的值是我们需要的价格
                price = price_data['market_data']['current_price']['usd']
                # 为边添加价格属性
                G[u][v][key]['price'] = price
        else:
            print(f"Warning: No price file for date {created_at} (looking for {json_file})")

# 保存图（如果需要）
nx.write_gexf(G, '/home/ta/devnet/bitcoin_2023_time_price.gexf')

