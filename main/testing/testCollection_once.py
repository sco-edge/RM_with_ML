import multiprocessing
import os
import re
import time
from typing import List
from box import Box
from configs import log
from dataCollector.OfflineProfilingDataCollector import OfflineProfilingDataCollector
from deployment.deployer import Deployer
from workloadGenerator.staticWorkload import StaticWorkloadGenerator
from workloadGenerator.trainticketWorkload import TrainticketWorkloadGenerator
# from infGenerator.busyInf import BusyInf
from utils.others import parse_mem

import configs

DEPLOYER = Deployer(
    configs.GLOBAL_CONFIG.namespace,
    configs.GLOBAL_CONFIG.pod_spec.cpu_size,
    configs.GLOBAL_CONFIG.pod_spec.mem_size,
    configs.GLOBAL_CONFIG.nodes_for_test,
    configs.GLOBAL_CONFIG.yaml_repo_path,
    configs.GLOBAL_CONFIG.app_img,
)


def full_init(app, port):
    os.system(f"mkdir tmp")
    os.system(f"mkdir log")
    DEPLOYER.full_init(app, configs.GLOBAL_CONFIG.nodes_for_infra, port)


def init_app(containers=None):
    os.system(f"mkdir tmp")
    os.system(f"mkdir log")
    # Initiallizing corresponding APP
    DEPLOYER.redeploy_app(containers)


def start_test(continues=False):
    """Test and collect data under different <cpu, memory> interference pair.
    To adjust test parameters, please check these configuration files:
    """
    os.system(f"mkdir tmp")
    os.system(f"mkdir log")
    if not continues:
        os.system(f"rm -rf {configs.GLOBAL_CONFIG.data_path}")
        os.system(f"mkdir -p {configs.GLOBAL_CONFIG.data_path}")
    # Prepare interference generator
    # interference_duration = str(
    #     configs.TESTING_CONFIG.duration
    #     * max(
    #         [
    #             x[1].max_clients
    #             for x in configs.TESTING_CONFIG.workload_config.services.items()
    #         ]
    #     )
    #     * 2
    # )
    # cpuInterGenerator = BusyInf(
    #     configs.GLOBAL_CONFIG.nodes_for_test,
    #     configs.TESTING_CONFIG.interference_config.cpu.cpu_size,
    #     configs.TESTING_CONFIG.interference_config.cpu.mem_size,
    #     "cpu",
    #     [f"{interference_duration}s"],
    # )
    # memoryInterGenerator = BusyInf(
    #     configs.GLOBAL_CONFIG.nodes_for_test,
    #     configs.TESTING_CONFIG.interference_config.mem.cpu_size,
    #     configs.TESTING_CONFIG.interference_config.mem.mem_size,
    #     "memory",
    #     ["300s", "anon", "300s"],
    # )

    # Prepare data collector
    dataCollector = OfflineProfilingDataCollector(
        configs.GLOBAL_CONFIG.namespace,
        configs.TESTING_CONFIG.collector_config.jaeger_host,
        configs.TESTING_CONFIG.collector_config.entry_point,
        configs.GLOBAL_CONFIG.prometheus_host,
        configs.GLOBAL_CONFIG.nodes_for_test,
        configs.GLOBAL_CONFIG.data_path,
        max_traces=configs.TESTING_CONFIG.collector_config.max_traces,
        mointorInterval=configs.TESTING_CONFIG.collector_config.monitor_interval,
        duration=configs.TESTING_CONFIG.duration,
    )

    # log.info("Deploying Application Once (No Interference)...")
    # for service in configs.TESTING_CONFIG.services:
    #     containers = configs.GLOBAL_CONFIG.replicas[service]
    #     DEPLOYER.deploy_app(containers)
    
    totalRound = (
        len(configs.TESTING_CONFIG.repeats)
        * sum(
            [
                x[1].max_clients
                for x in configs.TESTING_CONFIG.workload_config.services.items()
                if x[0] in configs.TESTING_CONFIG.services
            ]
        )
    )
    passedRound = 0
    usedTime = 0

    for service in configs.TESTING_CONFIG.services:
        containers = configs.GLOBAL_CONFIG.replicas[service]
        currentWorkloadConfig = configs.TESTING_CONFIG.workload_config.services[service]
        if configs.TESTING_CONFIG.workload_config.wrk_path == "python":
            workloadGenerator = TrainticketWorkloadGenerator(
                duration=configs.TESTING_CONFIG.duration,
                target_throughput=currentWorkloadConfig.throughput,  # 단일 client 기준
                client_num=1,  # generateWorkload 함수에서 clientNum이 또 들어감
                base_url=currentWorkloadConfig.url
            )
        else:
            workloadGenerator = StaticWorkloadGenerator(
                currentWorkloadConfig.thread_num,
                currentWorkloadConfig.connection_num,
                configs.TESTING_CONFIG.duration,
                currentWorkloadConfig.throughput,
                configs.TESTING_CONFIG.workload_config.wrk_path,
                currentWorkloadConfig.script_path,
                currentWorkloadConfig.url,
            )

        for repeat in configs.TESTING_CONFIG.repeats:
            for clientNum in range(1, currentWorkloadConfig.max_clients + 1):
                roundStartTime = time.time()
                testName = f"[{service}]{repeat}r{clientNum}c"

                log.info(
                    f"Repeat {repeat} of {service}: {clientNum} clients (No Interference)"
                )

                if passedRound != 0:
                    avgTime = usedTime / passedRound
                    log.info(
                        f"Used time: {timeParser(usedTime)}, "
                        f"Avg. round time: {timeParser(avgTime)}, "
                        f"Estimated time left: {timeParser(avgTime * (totalRound - passedRound))}"
                    )

                startTime = int(time.time())
                test_data = {
                    "repeat": repeat,
                    "start_time": startTime,
                    "service": service,
                    "cpu_inter": 0,
                    "mem_inter": 0,
                    "target_throughput": clientNum * currentWorkloadConfig.throughput,
                    "test_name": testName,
                }

                workloadGenerator.generateWorkload(testName, clientNum)
                dataCollector.collect_data_async(test_data)

                passedRound += 1
                usedTime += time.time() - roundStartTime
                    
    dataCollector.wait_until_done()


def timeParser(time):
    time = int(time)
    hours = format(int(time / 3600), "02d")
    minutes = format(int((time % 3600) / 60), "02d")
    secs = format(int(time % 3600 % 60), "02d")
    return f"{hours}:{minutes}:{secs}"
