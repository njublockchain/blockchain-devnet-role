# %%
import networkx
import pickle

name = "bips_2023"
with open("bips_2023.pkl", "rb") as f:
    G: networkx.MultiDiGraph = pickle.load(f)
print(G.number_of_nodes(), G.number_of_edges(), len(list(networkx.isolates(G))))
# %%
event_cat = "issue_tracking"
code_contrib_events = ["PushEvent", "CreateEvent", "DeleteEvent", "CommitCommentEvent"]
code_review_events = ["PullRequestReviewEvent", "PullRequestReviewCommentEvent", "PullRequestReviewThreadEvent"]
issue_tracking_events = ["IssuesEvent", "IssueCommentEvent"]
non_professional_events = [
    "WatchEvent",
    "ForkEvent",
    "GollumEvent",
    "PublicEvent",
    "SponsorshipEvent",
]


event_cats = {
    "code_contrib": code_contrib_events,
    "non_professional": non_professional_events,
    "code_review": code_review_events,
    "issue_tracking": issue_tracking_events,
}

# %%
wip_G = networkx.MultiDiGraph()
for u, v, k, data in G.edges(keys=True, data=True):
    if data["event_type"] in event_cats[event_cat]:
        wip_G.add_edge(u, v, key=k, **data)

G = wip_G
print(G.number_of_nodes(), G.number_of_edges(), len(list(networkx.isolates(G))))

# %%
for node in G.nodes:
    if str(node).startswith("https://api.github.com/repos/"):
        G.nodes[node]["type"] = "repo"
    elif str(node).startswith("https://api.github.com/users/"):
        G.nodes[node]["type"] = "user"
    else:
        print(f"Unknown type for {node}")

# write with pickle
import os

os.makedirs("rebuilt", exist_ok=True)
with open(f"rebuilt/{name}_{event_cat}_wip.pkl", "wb") as f:
    pickle.dump(G, f)

networkx.write_gexf(G, f"rebuilt/{name}_{event_cat}_wip.gexf")
# %%


# 找到所有类型为B的节点
# repo_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "repo"]
# from datetime import datetime


# def has_edge_in_one_day(repo, u, v):
#     return any(
#         datetime.fromtimestamp(x["created_at"]).date()
#         == datetime.fromtimestamp(y["created_at"]).date()
#         for _, _, x in G.edges((repo, u), data=True)
#         for _, _, y in G.edges((repo, v), data=True)
#     )


# # 对每个B节点
# for repo in repo_nodes:
#     # 获取与B相连的A节点
#     neighbor_user_nodes = [n for n in G.neighbors(repo)]

#     # 在这些A节点之间添加新边
#     for u, v in [
#         (x, y)
#         for x in neighbor_user_nodes
#         for y in neighbor_user_nodes
#         if x != y and has_edge_in_one_day(repo, x, y)
#     ]:
#         G.add_edge(u, v, repo=repo)


# 删除所有B节点
# G.remove_nodes_from(repo_nodes)

print(G.number_of_nodes(), G.number_of_edges(), len(list(networkx.isolates(G))))

# %%

# convert all repo into user or org
repo_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "repo"]

new_G = networkx.MultiDiGraph()

# for node, node_data in G.nodes(data=True):
#     if node_data["type"] == "user":
#         # new_G.add_node(node) # 添加用户节点

#         for u, v, k, edge_data in G.edges(node, keys=True, data=True):
#             new_G.add_edge(u, v, key=k, **edge_data)

#     else: # node is repo
#         repo_name = node.split('/')[-1]
#         repo_owner = "https://api.github.com/users/" + node.split('/')[-2]

#         # if repo_owner not in G.nodes: # orgname/repo
#         #     new_G.add_node(repo_owner)
        
#         for u, v, k, edge_data in G.edges(node, keys=True, data=True):
#             if u.startswith("https://api.github.com/users/"):
#                 new_G.add_edge(u, repo_owner, key=k, **edge_data)
#             else:
#                 new_G.add_edge(v, repo_owner, key=k, **edge_data)

for u, v, k, edge_data in G.edges(keys=True, data=True):
    if u.startswith("https://api.github.com/repos/"):
        print("err: repo as a from")
        exit
    
    if v.startswith("https://api.github.com/repos/"):
        new_G.add_edge(u, "https://api.github.com/users/" + v.split("/")[-2], key=k, **edge_data)
    else:
        new_G.add_edge(u, v, key=k, **edge_data)

G = new_G

for node in G.nodes:
    if str(node).startswith("https://api.github.com/repos/"):
        G.nodes[node]["type"] = "repo"
    elif str(node).startswith("https://api.github.com/users/"):
        G.nodes[node]["type"] = "user"
    else:
        print(f"Unknown type for {node}")

print(G.number_of_nodes(), G.number_of_edges(), len(list(networkx.isolates(G))))

# %%
# G = networkx.Graph(G)

# print(G.number_of_nodes(), G.number_of_edges())

# %%
os.makedirs("rebuilt", exist_ok=True)
with open(f"rebuilt/{name}_{event_cat}.pkl", "wb") as f:
    pickle.dump(G, f)

networkx.write_gexf(G, f"rebuilt/{name}_{event_cat}.gexf")
# %%
