import pandas as pd
import numpy as np

df = pd.read_csv("data/trace.csv")

df['traceTime'] = df['traceTime'] // 1000

graph_structures = {}
graph_class_mapping = {}
graph_class_counter = 0

arg_class_mapping = {}
arg_class_counter = 0

trace_info_list = []
trace_time_list = []

grouped = df.groupby("traceId")

for trace_id, group in grouped:
    edges = set()
    for _, row in group.iterrows():
        parentMS = 'parentMS'  
        childMS = 'childMS'    
        edges.add((row[parentMS], row[childMS]))
    edge_key = tuple(sorted(edges))
    
    if edge_key not in graph_class_mapping:
        graph_class_mapping[edge_key] = graph_class_counter
        graph_class_id = graph_class_counter
        graph_class_counter += 1
    else:
        graph_class_id = graph_class_mapping[edge_key]
    
    root_rows = group[group['parentId'] == group['traceId']]
    arg_value = None
    if not root_rows.empty:
        arg_value = root_rows.iloc[0]["parentOperation"]
    else:
        arg_value = group.iloc[0]["parentOperation"]  # 대체값
    
    if arg_value not in arg_class_mapping:
        arg_class_mapping[arg_value] = arg_class_counter
        arg_class_id = arg_class_counter
        arg_class_counter += 1
    else:
        arg_class_id = arg_class_mapping[arg_value]

    trace_time = group['traceTime'].iloc[0]
    trace_time_list.append(trace_time)
    trace_info_list.append((graph_class_id, arg_class_id))

window_ms = 10
start_time = min(trace_time_list)
end_time = max(trace_time_list)
total_windows = int((end_time - start_time) / window_ms) + 1

num_graph_classes = graph_class_counter
num_arg_classes = arg_class_counter

arg_counts = [[0]*num_arg_classes for _ in range(total_windows)]
graph_counts = [[0]*num_graph_classes for _ in range(total_windows)]
load_counts = [0]*total_windows


for (graph_class_id, arg_class_id), t in zip(trace_info_list, trace_time_list):
    win_index = int((t - start_time) / window_ms)
    if 0 <= win_index < total_windows:
        load_counts[win_index] += 1
        arg_counts[win_index][arg_class_id] += 1
        graph_counts[win_index][graph_class_id] += 1

# calculate distribution
arg_dist_list = []
graph_dist_list = []
for i in range(total_windows):
    count = load_counts[i]
    if count > 0:
        arg_dist = [c / count for c in arg_counts[i]]
        graph_dist = [c / count for c in graph_counts[i]]
    else:
        arg_dist = [0] * num_arg_classes
        graph_dist = [0] * num_graph_classes
    arg_dist_list.append(arg_dist)
    graph_dist_list.append(graph_dist)
max_load = max(load_counts) if max(load_counts) > 0 else 1
load_counts = [x / max_load for x in load_counts]


l = 5
inputs = []
targets = []

for t in range(0, total_windows -2*l+1, l):
    input_seq = []
    target_seq = []

    for j in range(t, t + l):
        vec = arg_dist_list[j] + graph_dist_list[j] + [load_counts[j]]
        input_seq.append(vec)
        
    if t <= total_windows - 2 :
        for j in range(t + l, t + l+2):
            vec = arg_dist_list[j] + graph_dist_list[j] + [load_counts[j]]
            target_seq.append(vec)
    else:
        #padding
        last_vec = input_seq[-1]
        target_seq = [last_vec] * l
        
    

    inputs.append(input_seq)
    targets.append(target_seq)
all_input_array = np.array(inputs)  
all_target_array = np.array(targets)  

# save
np.save("data/graph_predict/all_input.npy", all_input_array)
np.save("data/graph_predict/all_target.npy", all_target_array)

print("all_input.npy shape:", all_input_array.shape)
print("all_target.npy shape:", all_target_array.shape)
