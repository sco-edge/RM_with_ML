import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSV 읽기
df = pd.read_csv('./sample_data/traceid_num.csv', header=None, names=['service', 'traceidnum'], dtype=str)
df['traceidnum'] = pd.to_numeric(df['traceidnum'], errors='coerce')
df = df.dropna(subset=['traceidnum'])
df = df[df['traceidnum'] > 0]
df['traceidnum'] = df['traceidnum'].astype(int)

# 정렬 + CDF 계산
sorted_counts = np.sort(df['traceidnum'].values)
cdf = np.arange(1, len(sorted_counts)+1) / len(sorted_counts)

# 부드러운 CDF (선 그래프)
plt.figure(figsize=(10, 6))
plt.plot(sorted_counts, cdf, color='blue', linewidth=2, label='CDF')
plt.xscale('log')  # 로그 스케일로 확대
plt.xlabel('Number of trace IDs per service (log scale)')
plt.ylabel('Cumulative fraction of services')
plt.title('CDF of trace counts per service')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('traceid_cdf_cleaned.png')
plt.show()
