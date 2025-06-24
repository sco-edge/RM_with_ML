import pandas as pd
from collections import defaultdict
import hashlib

# CSV 읽기
df = pd.read_csv("data/CallGraph/merged_callgraph.csv", on_bad_lines='skip')

# trace별로 그룹화
grouped = df.groupby("traceid")

graph_class_dict = {}
arg_class_dict = {}
data_by_class_pair = defaultdict(list)
graph_class_counter = 0
arg_class_counter = 0

for trace_id, group in grouped:
    # argument 구분: root 요청의 service 기준
    root_rows = group[group['um'] == 'USER']
    if root_rows.empty:
        continue
    arg_class_key = root_rows['service'].iloc[0]

    if arg_class_key not in arg_class_dict:
        if len(arg_class_dict) >= 12:
            continue  # 12개 초과 X
        arg_class_dict[arg_class_key] = arg_class_counter
        arg_class_counter += 1
    arg_class = arg_class_dict[arg_class_key]

    # 그래프 구조 정의 (parent-child microservice 간 edge 집합)
    edges = set()
    for _, row in group.iterrows():
        edges.add((row['um'], row['dm']))
    edge_key = tuple(sorted(edges))
    graph_key_hash = hashlib.md5(str(edge_key).encode()).hexdigest()

    if graph_key_hash not in graph_class_dict:
        if len(graph_class_dict) >= 36:
            continue  # 36개 초과 X
        graph_class_dict[graph_key_hash] = graph_class_counter
        graph_class_counter += 1
    graph_class = graph_class_dict[graph_key_hash]

    # (argument class, graph class) pair에 해당하는 trace 저장
    class_pair = (arg_class, graph_class)
    data_by_class_pair[class_pair].append(group)

# 균형 있게 N개씩만 추출
balanced_traces = []
N = 10  # 클래스당 최대 N개
for traces in data_by_class_pair.values():
    balanced_traces.extend(traces[:N])

# 결과 합치기 및 저장
final_df = pd.concat(balanced_traces)
final_df.to_csv("balanced_subset.csv", index=False)
print(f"Saved {len(final_df)} rows to balanced_subset.csv")
