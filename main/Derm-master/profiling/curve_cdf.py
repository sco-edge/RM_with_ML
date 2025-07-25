import matplotlib as mpl
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import entropy
import os
import time
from collections import namedtuple, defaultdict

mpl.rcParams.update(mpl.rcParamsDefault)

parent_services = [
    "frontend",
    "search",
]
child_services = [
    "geo",
    "profile",
    "rate",
    "recommendation",
    "reservation",
]
complex_non_shared = ["geo", "rate", "reservation", "search"]
simple_non_shared = ["recommendation"]

MSData = namedtuple(
    "MSData",
    [
        "simple_graph_ratio",
        "simple_invoke_times",
        "complex_graph_ratio",
        "complex_invoke_times",
        "container_number",
    ],
)

# (simple * simple_invoke_times + complex * complex_invoke_times) / container_number
ms_dict = {
    "frontend": 1.25,
    "geo": 1.0,
    "profile": 2.0,
    "rate": 1.0,
    "recommendation": 4.5,
    "reservation": 0.33,
    "search": 0.5,
}


def fit_cdf(x, c):
    return 1 - np.array(np.exp(-c * (x)))


def get_latency_e2e(path: str, graph_type: str, service_name: str):
    latency_list = pd.DataFrame()
    for epoch in range(0, 3):
        file_path = f"{path}/epoch_{epoch}/{graph_type}/raw_data.csv"
        data = pd.read_csv(file_path)
        data = data[~data["childMS"].isin(data["parentMS"])].reset_index()
        latency = (
            data.groupby(["traceId", "childMS"])["childDuration"].mean().reset_index()
        )
        latency_list = pd.concat([latency_list, latency], axis=0)
    latency_list = latency_list[latency_list["childMS"] == service_name][
        "childDuration"
    ]
    latency_list = latency_list.sort_values().reset_index(drop=True)
    latency_list = [i / 1000 for i in latency_list if i > 0]
    return latency_list


def get_latency(path: str, graph_type: str, service_name: str):
    latency_list = pd.DataFrame()
    for epoch in range(0, 3):
        file_path = f"{path}/epoch_{epoch}/{graph_type}/raw_data.csv"
        data = pd.read_csv(file_path)
        data = (
            data.groupby(["traceId", "parentId"])["childDuration"]
            .sum()
            .reset_index()
            .rename(columns={"childDuration": "childSum"})
            .merge(data, on=["traceId", "parentId"])
        )
        data = data.assign(
            exactParentDuration=data["parentDuration"] - data["childSum"]
        )
        data = data.drop_duplicates(["traceId", "parentId"])
        latency = (
            data.groupby(["traceId", "parentMS"])["exactParentDuration"]
            .mean()
            .reset_index()
        )
        latency_list = pd.concat([latency_list, latency], axis=0)
    latency_list = latency_list[latency_list["parentMS"] == service_name][
        "exactParentDuration"
    ]
    latency_list = latency_list.sort_values().reset_index(drop=True)
    latency_list = [i / 1000 for i in latency_list if i > 0]
    return latency_list


def fit(folder: str, workload: int, service_name: str):
    path = f"{folder}/{workload}"
    plt.figure()
    if service_name in parent_services:
        if service_name in complex_non_shared:
            latency_list = get_latency(path, "complex", service_name)
        elif service_name in simple_non_shared:
            latency_list = get_latency(path, "simple", service_name)
        else:
            simple_latency_list = get_latency(path, "simple", service_name)
            complex_latency_list = get_latency(path, "complex", service_name)
            latency_list = simple_latency_list + complex_latency_list
    else:
        if service_name in complex_non_shared:
            latency_list = get_latency_e2e(path, "complex", service_name)
        elif service_name in simple_non_shared:
            latency_list = get_latency_e2e(path, "simple", service_name)
        else:
            simple_latency_list = get_latency_e2e(path, "simple", service_name)
            complex_latency_list = get_latency_e2e(path, "complex", service_name)
            latency_list = simple_latency_list + complex_latency_list
    latency_list.sort()
    p95 = pd.Series(latency_list).quantile(0.95)
    latency_range = str(latency_list[0]) + "___" + str(latency_list[-1])
    skew = latency_list[0]
    latency_list = [i - latency_list[0] for i in latency_list]
    latency_step = (latency_list[-1] - latency_list[0]) / 100
    y_fit: list = np.histogram(
        latency_list, np.arange(latency_list[0], latency_list[-1], latency_step)
    )[0].tolist()
    y_fit.insert(0, 0)
    cdf = np.cumsum(y_fit)
    x_fit = np.histogram(
        latency_list, np.arange(latency_list[0], latency_list[-1], latency_step)
    )[1]
    y_fit = cdf / np.array(cdf[-1])
    x_fit = np.array(x_fit)
    data_num = len(x_fit)
    start = time.perf_counter()
    parameters, _ = curve_fit(f=fit_cdf, xdata=x_fit, ydata=y_fit, maxfev=500000)
    simulation = fit_cdf(x_fit, parameters[0])
    end = time.perf_counter()
    print(f"Train time: {end - start} seconds")
    ml = 1 / np.array(latency_list).mean()
    simulation_ml = fit_cdf(x_fit, ml)
    loss_fit = entropy(y_fit[1:], simulation[1:])
    loss_ml = entropy(y_fit[1:], simulation_ml[1:])
    plt.plot(
        x_fit,
        simulation,
        linestyle="--",
        linewidth=1,
        color="blue",
        label="fit_para = " + str(parameters[0]) + "\n KL = " + str(loss_fit),
    )
    plt.plot(
        x_fit,
        simulation_ml,
        linestyle="--",
        linewidth=1,
        color="red",
        label="ml_para = " + str((ml)) + "\n KL = " + str(loss_ml),
    )
    plt.grid(color="#C0C0C0", linestyle="-", linewidth=1, axis="y")
    plt.plot(
        x_fit, y_fit, linestyle="--", linewidth=1, color="green", label="groundtruths"
    )
    plt.legend()
    plt.title(service_name)
    os.system(f"mkdir -p figures/{service_name}")
    plt.savefig(f"figures/{service_name}/{workload}.png")
    plt.close()
    print(f"[{service_name}] @ workload={workload} → latency data points: {len(latency_list)}")
    if len(latency_list) < 10:
        print(f"⚠️ Too few data for {service_name} ({workload}) — possible curve_fit instability")

    return parameters[0], ml, loss_fit, loss_ml, data_num, latency_range, skew, p95


def collect_parameters(data_folder: str, service_name: str):
    paras_list = []
    ml_list = []
    loss_fit_list = []
    loss_ml_list = []
    workloads = []
    data_num_list = []
    latency_range_list = []
    skew_list = []
    p95_list = []
    workloads = [30, 60]
    for workload in workloads:
        paras, ml, loss_fit, loss_ml, data_num, latency_range, skew, p95 = fit(
            data_folder, workload, service_name
        )
        paras_list.append(paras)
        ml_list.append(ml)
        loss_fit_list.append(loss_fit)
        loss_ml_list.append(loss_ml)
        data_num_list.append(data_num)
        latency_range_list.append(latency_range)
        skew_list.append(skew)
        p95_list.append(p95)

    workload_ratio = ms_dict[service_name]
    avg_workloads = [round(workload * workload_ratio, 2) for workload in workloads]
    df = [
        paras_list,
        ml_list,
        loss_fit_list,
        loss_ml_list,
        avg_workloads,
        latency_range_list,
        skew_list,
        data_num_list,
        workloads,
        p95_list,
    ]
    df = pd.DataFrame(df).T
    df.columns = [
        "fit_pars",
        "ml_pars",
        "kl_fit",
        "kl_ml",
        "workloadPerPod",
        "latency_range",
        "skew",
        "data_num",
        "workloads",
        "p95",
    ]
    output_dir = os.path.abspath("data/hotel-reserv")
    os.makedirs(output_dir, exist_ok=True)
    df.sort_values(["workloadPerPod", "workloads"], inplace=True)
    df = df.drop(columns="workloads").reset_index(drop=True)
    df.to_csv(os.path.join(output_dir, f"{service_name}.csv"), index=False)
    


for i in parent_services:
    collect_parameters("/home/netlab/RM_with_ML/deploy_hotel_reserv/AE/data/data_hotel-reserv/offlineTestResult",i)
for i in child_services:
    collect_parameters("/home/netlab/RM_with_ML/deploy_hotel_reserv/AE/data/data_hotel-reserv/offlineTestResult",i)
    



    





