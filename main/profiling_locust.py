from dataCollector.OfflineProfilingDataCollector import OfflineProfilingDataCollector
from configs import log, GLOBAL_CONFIG, TESTING_CONFIG
from utils.others import parse_mem
import time
import os


def start_trace_collection(continues=False):
    os.system("mkdir -p tmp")
    os.system("mkdir -p log")

    if not continues:
        os.system(f"rm -rf {GLOBAL_CONFIG.data_path}")
        os.system(f"mkdir -p {GLOBAL_CONFIG.data_path}")

    dataCollector = OfflineProfilingDataCollector(
        GLOBAL_CONFIG.namespace,
        TESTING_CONFIG.collector_config.jaeger_host,
        TESTING_CONFIG.collector_config.entry_point,
        GLOBAL_CONFIG.prometheus_host,
        GLOBAL_CONFIG.nodes_for_test,
        GLOBAL_CONFIG.data_path,
        max_traces=TESTING_CONFIG.collector_config.max_traces,
        mointorInterval=TESTING_CONFIG.collector_config.monitor_interval,
        duration=TESTING_CONFIG.duration,
    )

    service = TESTING_CONFIG.services[0]
    repeat = TESTING_CONFIG.repeats[0]
    clientNum = TESTING_CONFIG.workload_config.services[service].max_clients

    testName = f"[{service}]{repeat}r{clientNum}c[locust]"
    test_data = {
        "repeat": repeat,
        "start_time": int(time.time()),
        "service": service,
        "cpu_inter": 0,
        "mem_inter": 0,
        "target_throughput": clientNum * TESTING_CONFIG.workload_config.services[service].throughput,
        "test_name": testName,
    }

    log.info(f"==> Waiting for Locust workload to run... (duration: {TESTING_CONFIG.duration}s)")
    time.sleep(TESTING_CONFIG.duration)  

    log.info("==> Start collecting traces")
    dataCollector.collect_data_async(test_data)
    dataCollector.wait_until_done()
    log.info("Trace collection completed.")


if __name__ == "__main__":
    start_trace_collection(continues=True)
