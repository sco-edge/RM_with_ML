import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter
from tqdm import tqdm
from heapq import nlargest

tqdm.pandas()

df = pd.read_parquet("./sample_data/valid_traces.parquet")

df.columns = df.columns.str.lower()

grouped = list(df.groupby("traceid"))

def graph_signature(G):
    
    sorted_nodes = sorted(G.nodes())
    index_map = {node: i for i, node in enumerate(sorted_nodes)}

    adj = np.zeros((len(sorted_nodes), len(sorted_nodes)), dtype=int)
    for u, v in G.edges():
        i, j = index_map[u], index_map[v]
        adj[i][j] = 1  

    return (tuple(sorted_nodes), tuple(adj.flatten()))

graph_class_mapping = {}
graph_class_counter = 0
signature_counter = Counter()
trace_graph_map = {}

# main loop
for trace_id, group in tqdm(grouped, desc="Processing traces"):
    

    G = nx.DiGraph()
    for _, row in group.iterrows():
        G.add_edge(row['um'], row['dm'])
    

    signature = graph_signature(G)

    if signature in graph_class_mapping:
        graph_class_id = graph_class_mapping[signature]
    else:
        graph_class_mapping[signature] = graph_class_counter
        graph_class_id = graph_class_counter
        graph_class_counter += 1
    signature_counter[signature] += 1
    trace_graph_map[trace_id] = signature
print(f"[Step 1] Total unique graph signatures: {len(signature_counter)}", flush=True)

top_signatures = set(sig for sig, _ in nlargest(36, signature_counter.items(), key=lambda x: x[1]))

print("[Step 2] Filtering traceIds...", flush=True)
filtered_trace_ids = set(tid for tid, sig in trace_graph_map.items() if sig in top_signatures)
ood_trace_ids = set(trace_graph_map.keys()) - filtered_trace_ids

df["traceid"] = df["traceid"].astype(str)
filtered_df = df[df["traceid"].progress_apply(lambda x: x in filtered_trace_ids)]
ood_df = df[df["traceid"].progress_apply(lambda x: x in ood_trace_ids)]



filtered_df.to_parquet("./sample_data/graph_filtered_trace_data.parquet", engine="fastparquet")
ood_df.to_parquet("./sample_data/ood_trace_data.parquet", engine="fastparquet")
print(f"Saved filtered trace data: {filtered_df.shape}")
print(f"Saved OOD trace data: {ood_df.shape}")
