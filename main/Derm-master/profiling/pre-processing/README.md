# Alibaba Pre-processing

python3 filter_disconnected_trace_data.py // Unknown, Unavailable data 삭제 및 CSV to Parquet
python3 split_trace_graph.py // Graph 분포 상위 36종 분류
python3 split_trace_argument.py // Argumnet 분포 상위 12종 분류

cd ..
python3 make_npy.py // Derm input Numpy, target Numpy 생성

