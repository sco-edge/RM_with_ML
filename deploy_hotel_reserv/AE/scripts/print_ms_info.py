import os
import pandas as pd
from collections import defaultdict

data_root = "./AE/data/data_hotel-reserv/offlineTestResult"
epochs = [0]
graph_types = ["simple", "complex"]

all_dfs = []
for epoch in epochs:
    for gtype in graph_types:
        file_path = f"{data_root}/60/epoch_{epoch}/{gtype}/raw_data.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df["epoch"] = epoch
            df["graph_type"] = gtype
            all_dfs.append(df)

if not all_dfs:
    print(" No data found.")
    exit()

merged_df = pd.concat(all_dfs, ignore_index=True)

# Microservices
all_services = sorted(set(merged_df["childMS"]).union(set(merged_df["parentMS"])))
print("Microservices:")
print(all_services)
print()

# parent / child 
parent_services = sorted(set(merged_df["parentMS"]))
child_services = sorted(set(merged_df["childMS"]))
print("🔹 Parent Services:")
print(parent_services)
print()
print("🔸 Child Services:")
print(child_services)
print()

# simple / complex count
svc_stats = defaultdict(lambda: {"simple": 0, "complex": 0})
for trace_id, trace_df in merged_df.groupby("traceId"):
    ms_set = set(trace_df["childMS"]) | set(trace_df["parentMS"])
    gtype = trace_df["graph_type"].iloc[0]
    for ms in ms_set:
        svc_stats[ms][gtype] += 1

print("비스별 simple/complex 등장 횟수:")
for ms in all_services:
    print(f"{ms}: simple={svc_stats[ms]['simple']}, complex={svc_stats[ms]['complex']}")
print()

# complex / simple only
complex_non_shared = [ms for ms in all_services if svc_stats[ms]["simple"] == 0 and svc_stats[ms]["complex"] > 0]
simple_non_shared = [ms for ms in all_services if svc_stats[ms]["complex"] == 0 and svc_stats[ms]["simple"] > 0]

print("Complex non-shared:")
print(complex_non_shared)
print()
print("Simple non-shared:")
print(simple_non_shared)
print()

# container (실제 값 넣어야 정확)
container_numbers = {
    "frontend": 8,
    "geo": 1,
    "profile": 5,
    "rate": 1,
    "recommendation": 2,
    "reservation": 3,
    "search": 2
}

# ms_dict calculation
trace_counts = {
    "simple": defaultdict(int),
    "complex": defaultdict(int),
}


base_path = "./AE/data/data_hotel-reserv/offlineTestResult/60/epoch_0"
simple_path = os.path.join(base_path, "simple", "raw_data.csv")
complex_path = os.path.join(base_path, "complex", "raw_data.csv")

# raw_data load
simple_df = pd.read_csv(simple_path)
complex_df = pd.read_csv(complex_path)

# trace 
simple_groups = list(simple_df.groupby("traceId"))
complex_groups = list(complex_df.groupby("traceId"))
simple_rows = [df for _, df in simple_groups]
complex_rows = [df for _, df in complex_groups]
for trace_df in simple_rows:
    for ms in set(trace_df["childMS"]).union(set(trace_df["parentMS"])):
        trace_counts["simple"][ms] += 1

for trace_df in complex_rows:
    for ms in set(trace_df["childMS"]).union(set(trace_df["parentMS"])):
        trace_counts["complex"][ms] += 1
ms_dict = {}
for ms in all_services:
    simple_invokes = svc_stats[ms]["simple"]
    complex_invokes = svc_stats[ms]["complex"]
    num_simple = trace_counts["simple"].get(ms, 1)  # 0 방지용 default 1
    num_complex = trace_counts["complex"].get(ms, 1)
    
    avg_simple = simple_invokes / num_simple
    avg_complex = complex_invokes / num_complex
    n = container_numbers.get(ms, 1)
    
    ms_dict[ms] = round((9 * avg_simple + 1 * avg_complex) / n, 2)
    print(ms,simple_invokes, complex_invokes, num_simple, num_complex, avg_simple,avg_complex)
    

print("ms_dict:")
for k, v in ms_dict.items():
    print(f'"{k}": {v},')
