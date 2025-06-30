import pandas as pd
from collections import Counter
from tqdm import tqdm
from heapq import nlargest

tqdm.pandas()

# 데이터 로드
df = pd.read_parquet("./sample_data/graph_filtered_trace_data.parquet")
df.columns = df.columns.str.lower()

df['traceid'] = df['traceid'].astype(str)
df['rpc_id'] = df['rpc_id'].astype(str)

# trace별로 group
grouped = list(df.groupby("traceid"))

# Step 1: trace별 root interface 수집
interface_counter = Counter()
trace_arg_map = {}

for trace_id, group in tqdm(grouped, desc="Analyzing root interfaces"):
    root_rows = group[group['rpc_id'] == "0"]
    if not root_rows.empty:
        interface = root_rows.iloc[0]['interface']
        trace_arg_map[trace_id] = interface
        interface_counter[interface] += 1
    else:
        continue  # root 없는 trace는 제외

# Step 2: 상위 12개 interface 선택
top_interfaces = set(iface for iface, _ in nlargest(12, interface_counter.items(), key=lambda x: x[1]))

print(f"[INFO] Top 12 interfaces selected: {len(top_interfaces)}")

# Step 3: filtering traceIds
filtered_trace_ids = set(tid for tid, iface in trace_arg_map.items() if iface in top_interfaces)
ood_trace_ids = set(trace_arg_map.keys()) - filtered_trace_ids

print(f"[INFO] Filtered traces: {len(filtered_trace_ids)}, OOD traces: {len(ood_trace_ids)}")

# Step 4: trace별 필터링
filtered_df = df[df["traceid"].progress_apply(lambda x: x in filtered_trace_ids)]
ood_df = df[df["traceid"].progress_apply(lambda x: x in ood_trace_ids)]

# Step 5: 저장
filtered_df.to_parquet("./sample_data/argument_filtered_trace_data.parquet", engine="fastparquet")
ood_df.to_parquet("./sample_data/ood_trace_data_argument.parquet", engine="fastparquet")

print(f"Saved filtered trace data: {filtered_df.shape}")
print(f"Saved OOD trace data: {ood_df.shape}")
