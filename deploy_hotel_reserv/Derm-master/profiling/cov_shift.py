import numpy as np
import os

# 경로 설정
input_path = "data/graph_predict/all_input_Alibaba_filtered.npy"
target_path = "data/graph_predict/all_target_Alibaba_filtered.npy"
save_dir = "ood_shift"
os.makedirs(save_dir, exist_ok=True)

# 데이터 로드
input_data = np.load(input_path)
target_data = np.load(target_path)

total_dim = input_data.shape[-1]  # 예: 9
graph_dim = target_data.shape[-1]  # 예: 4
load_dim = 1
arg_dim = total_dim - graph_dim - load_dim  # 예: 4

arg = input_data[:, :, :arg_dim]

# Gaussian noise 기반 covariate shift 생성
random_arg = arg + np.random.normal(loc=0.3, scale=0.05, size=arg.shape)
random_arg = np.clip(random_arg, 0, 1)  # 값 범위를 [0, 1]로 유지

gaussian_input = input_data.copy()
gaussian_input[:, :, :arg_dim] = random_arg

gaussian_target = target_data.copy()

# 저장
np.save(os.path.join(save_dir, "cov_input_gaussian.npy"), gaussian_input)
np.save(os.path.join(save_dir, "cov_target_gaussian.npy"), gaussian_target)

bias_input = input_data.copy()
bias_input[:, :, :arg_dim] = 0.0
bias_input[:, :, 0] = 0.95
if arg_dim > 1:
    bias_input[:, :, 1] = 0.05

bias_target = target_data.copy()

# 저장
np.save(os.path.join(save_dir, "cov_input_bias.npy"), bias_input)
np.save(os.path.join(save_dir, "cov_target_bias.npy"), bias_target)