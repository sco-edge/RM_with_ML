import numpy as np


inputs = np.load("data/graph_predict/all_input_train_ticket.npy")
targets = np.load("data/graph_predict/all_target_train_ticket.npy")

print(f"[INFO] Input shape: {inputs.shape} => N={inputs.shape[0]}, l={inputs.shape[1]}, D={inputs.shape[2]}")
print(f"[INFO] Target shape: {targets.shape}")

D = inputs.shape[2]
load_index = D - 1

# Check unique ARG and GRAPH class counts
arg_classes = set()
graph_classes = set()
for i in range(inputs.shape[0]):
    arg_classes.update(np.nonzero(inputs[i, 0])[0])  # arg dist
    graph_classes.update(np.nonzero(inputs[i, 1])[0])  # graph dist

print(f"[INFO] Estimated number of ARG classes: {len(arg_classes)}")
print(f"[INFO] Estimated number of GRAPH classes: {len(graph_classes)}")
print(f"[INFO] Load feature index: {load_index}")
print(f"[INFO] Sample load values: {[inputs[i, 2, load_index] for i in range(min(5, inputs.shape[0]))]}")

