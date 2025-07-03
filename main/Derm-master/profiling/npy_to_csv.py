import numpy as np
import pandas as pd

input_npy_path = "data/graph_predict/all_input_train_ticket.npy"
target_npy_path = "data/graph_predict/all_target_train_ticket.npy"

input_array = np.load(input_npy_path)
target_array = np.load(target_npy_path)


input_flat = input_array.reshape(input_array.shape[0], -1)
target_flat = target_array.reshape(target_array.shape[0], -1)

input_df = pd.DataFrame(input_flat)
target_df = pd.DataFrame(target_flat)

input_df.to_csv("all_input.csv", index=False)
target_df.to_csv("all_target.csv", index=False)

print("CSV stored")
