import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict, deque
import os
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
# CSV 로딩
df = pd.read_parquet("./pre-processing/sample_data/filtered_trace_data.parquet")

# 그래프 및 argument 분류용 구조체
graph_class_mapping = {}
graph_class_counter = 0

arg_class_mapping = {}
arg_class_counter = 0

trace_info_list = []
trace_time_list = []

# traceTime 정리
df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')  # 문자열도 안전하게 처리
df['traceTime'] = df.groupby('traceid')['timestamp'].transform('min') 
grouped = df.groupby("traceid")







def graph_signature(G):
    
    sorted_nodes = sorted(G.nodes())
    index_map = {node: i for i, node in enumerate(sorted_nodes)}

    adj = np.zeros((len(sorted_nodes), len(sorted_nodes)), dtype=int)
    for u, v in G.edges():
        i, j = index_map[u], index_map[v]
        adj[i][j] = 1  # 존재 여부만 기록 (횟수 무시)

    return (tuple(sorted_nodes), tuple(adj.flatten()))

# main loop
for trace_id, group in tqdm(grouped, desc="Processing traces", total=len(grouped)):

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
        print(f"[NEW] Trace {trace_id} registered as new graph class {graph_class_counter}",flush=True)
        print(f" -> Edges: {list(G.edges())}",flush=True)
        print(f" -> Nodes: {list(G.nodes())}",flush=True)
        
    root_rows = group[~group['uminstanceid'].isin(group['dminstanceid'])]


    if root_rows.empty:
        arg_value = group.iloc[0]["service"]  # fallback
    else:
        arg_value = root_rows.iloc[0]["service"]

    if arg_value not in arg_class_mapping:
        arg_class_mapping[arg_value] = arg_class_counter
        arg_class_id = arg_class_counter
        arg_class_counter += 1
    else:
        arg_class_id = arg_class_mapping[arg_value]

    trace_time = group['traceTime'].iloc[0]
    trace_time_list.append(trace_time)
    trace_info_list.append((graph_class_id, arg_class_id))

# 시간 윈도우 설정
window_s = 60
zipped = sorted(zip(trace_info_list, trace_time_list), key=lambda x: x[1])
trace_info_list, trace_time_list = zip(*zipped)

start_time = min(trace_time_list)
end_time = max(trace_time_list)
total_windows = int((end_time - start_time) / window_s) + 1

num_graph_classes = graph_class_counter
num_arg_classes = arg_class_counter

arg_counts = [[0] * num_arg_classes for _ in range(total_windows)]
graph_counts = [[0] * num_graph_classes for _ in range(total_windows)]
load_counts = [0] * total_windows

for (graph_class_id, arg_class_id), t in zip(trace_info_list, trace_time_list):
    win_index = int((t - start_time) / window_s)
    if 0 <= win_index < total_windows:
        load_counts[win_index] += 1
        arg_counts[win_index][arg_class_id] += 1
        graph_counts[win_index][graph_class_id] += 1

# 분포 계산
arg_dist_list = []
graph_dist_list = []
for i in tqdm(range(total_windows), desc="Building window distribution"):
    count = load_counts[i]
    if count > 0:
        arg_dist = [c / count for c in arg_counts[i]]
        graph_dist = [c / count for c in graph_counts[i]]
    else:
        arg_dist = [0] * num_arg_classes
        graph_dist = [0] * num_graph_classes
    arg_dist_list.append(arg_dist)
    graph_dist_list.append(graph_dist)

# 부하 정규화
max_load = max(load_counts) if max(load_counts) > 0 else 1
load_counts = [x / max_load for x in load_counts]

# 최종 벡터 구성
arg_array = np.array(arg_dist_list)           
graph_array = np.array(graph_dist_list)       
load_array = np.array(load_counts).reshape(-1, 1)  

full_vector = np.concatenate([arg_array, graph_array, load_array], axis=1)  # shape: (T, A+G+1)

# 시퀀스 생성
l = 5
input_seqs = []
target_seqs = []
arg_dim = arg_array.shape[1]
graph_dim = graph_array.shape[1]

for t in tqdm(range(0, total_windows - 2 * l + 1, l), desc="Generating sequences"):
    input_seq = full_vector[t : t + l]
    target = full_vector[t + l][arg_dim:arg_dim + graph_dim].reshape(1, -1)
    input_seqs.append(input_seq)
    target_seqs.append(target)

    #if t % 1000 == 0:
        #print(f"[INFO] Processed window {t}/{total_windows}")

# 저장
input_array = np.stack(input_seqs)         # shape: (N, l, D)
target_array = np.stack(target_seqs)       # shape: (N, D)

np.save("data/graph_predict/all_input_Alibaba_filtered.npy", input_array)
np.save("data/graph_predict/all_target_Alibaba_filtered.npy", target_array)

print("all_input.npy shape:", input_array.shape,flush=True)
print("all_target.npy shape:", target_array.shape,flush=True)
print(arg_counts[0],graph_counts[0],flush=True)
print("arg_array shape:", arg_array.shape,flush=True)
print("graph_array shape:", graph_array.shape,flush=True)
print("load_array shape:", load_array.shape,flush=True)
print("full_vector shape:", full_vector.shape,flush=True)
print("num_arg_classes:", num_arg_classes,flush=True)
print("num_graph_classes:", num_graph_classes,flush=True)

print("=== Argument Class Mapping (parentOperation 기준) ===")
for arg_str, class_id in arg_class_mapping.items():
    print(f"Arg Class {class_id}: {arg_str}",flush=True)

total_arg_counts = np.sum(arg_counts, axis=0)
total_graph_counts = np.sum(graph_counts, axis=0)

# Argument Class 분포 출력
print("=== Argument Class Distribution ===")
for arg_id, count in enumerate(total_arg_counts):
    percent = (count / sum(total_arg_counts)) * 100 if sum(total_arg_counts) > 0 else 0
    print(f"  Arg Class {arg_id}: {count} traces ({percent:.2f}%)",flush=True)

# Graph Class 분포 출력
print("\n=== Graph Class Distribution ===")
for graph_id, count in enumerate(total_graph_counts):
    percent = (count / sum(total_graph_counts)) * 100 if sum(total_graph_counts) > 0 else 0
    print(f"  Graph Class {graph_id}: {count} traces ({percent:.2f}%)",flush=True)