import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import time

# Convert CSV to Parquet
def csv_to_parquet(csv_path, parquet_path):
    chunksize = 1_000_000  
    writer = None
    
    print(f"Converting {csv_path} to {parquet_path} ...")
    for i, chunk in enumerate(pd.read_csv(
        csv_path,
        dtype=str,
        chunksize=chunksize,
        on_bad_lines='skip',
        engine='python'  
    )):
        table = pa.Table.from_pandas(chunk)
        if writer is None:
            writer = pq.ParquetWriter(parquet_path, table.schema)
        writer.write_table(table)
        print(f"[Chunk {i}] 저장 완료")

    if writer:
        writer.close()
    print("Conversion complete.\n")

# Sanity Check Function
def sanity_check(group):   
    if not (group['rpc_id'] == '0').any():
        return False

    if (
        group['um'].isin(['UNKNOWN', 'UNAVAILABLE']).any() or 
        group['dm'].isin(['UNKNOWN', 'UNAVAILABLE']).any()
    ):
        return False

    if group['rpc_id'].duplicated().any():
        return False

    rpc_ids = set(group['rpc_id'].values)
    prefixes = set()
    for rpc_id in rpc_ids:
        while '.' in rpc_id:
            rpc_id = rpc_id.rsplit('.', 1)[0]
            prefixes.add(rpc_id)

    return prefixes.issubset(rpc_ids)

# Filter valid traces and save to Parquet
def filter_valid_traces(parquet_output, valid_output):
    print(f"Streaming valid traces from {parquet_output}")
    start_time = time.time()

    parquet_file = pq.ParquetFile(parquet_output)
    writer = None

    for i in range(parquet_file.num_row_groups):
        row_group = parquet_file.read_row_group(i)
        df = row_group.to_pandas().astype(str)

        # valid한 trace만 모아서 한 번에 쓰기
        valid_rows = []
        for _, group in df.groupby('traceid'):
            if sanity_check(group):
                valid_rows.append(group)

        if valid_rows:
            batch_df = pd.concat(valid_rows, axis=0)
            table = pa.Table.from_pandas(batch_df)

            if writer is None:
                writer = pq.ParquetWriter(valid_output, table.schema, compression='snappy')
            writer.write_table(table)

        print(f"[Row group {i}] ")

    if writer:
        writer.close()
        print(f"Saved valid traces to {valid_output}")
    else:
        print("No valid traces found.")

    print(f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")

csv_input = "./sample_data/data_origin.csv"
parquet_output = "./sample_data/data_origin.parquet"
valid_output = "./sample_data/valid_traces.parquet"

os.makedirs("./sample_data", exist_ok=True)
csv_to_parquet(csv_input, parquet_output)
filter_valid_traces(parquet_output, valid_output)
