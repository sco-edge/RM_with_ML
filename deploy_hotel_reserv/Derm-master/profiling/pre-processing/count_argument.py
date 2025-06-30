import pandas as pd
from collections import Counter
from tqdm import tqdm


tqdm.pandas()

# 파일 로딩
df = pd.read_parquet("./sample_data/graph_filtered_trace_data.parquet")

# 안전한 타입 변환
df['rpc_id'] = df['rpc_id'].astype(str)
df['traceid'] = df['traceid'].astype(str)

# 각 trace별로 root 호출 찾기 (rpcid == "0")
grouped = df.groupby("traceid")
arg_class_counter = Counter()

for trace_id, group in tqdm(grouped, desc="Processing traces"):
    root_rows = group[group['rpc_id'] == "0"]
    if not root_rows.empty:
        rpcid_val = root_rows.iloc[0]['rpc_id']
        arg_class_counter[rpcid_val] += 1
    else:
        # fallback: 가장 첫 row의 rpcid 사용
        fallback_val = group.iloc[0]['rpc_id']
        arg_class_counter[fallback_val] += 1

# 결과 출력
print("=== Argument Class Count (based on rpcid) ===")
for arg, count in arg_class_counter.items():
    print(f"  Argument Class '{arg}': {count} traces")

print(f"\n총 고유 argument class 수: {len(arg_class_counter)}")
