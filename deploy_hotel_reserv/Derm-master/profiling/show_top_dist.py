import numpy as np

# 파일 경로 설정
input_path = "data/graph_predict/all_input_Alibaba.npy"
target_path = "data/graph_predict/all_target_Alibaba.npy"

# NumPy 파일 로딩
input_data = np.load(input_path)
target_data = np.load(target_path)

# Shape 출력
print(f"Input shape: {input_data.shape}")
print(f"Target shape: {target_data.shape}")
