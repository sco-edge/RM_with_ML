import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import time
from configs import log
from dataCollector.OfflineProfilingDataCollector import OfflineProfilingDataCollector
from deployment.deployer import Deployer
from workloadGenerator.staticWorkload import StaticWorkloadGenerator
# from infGenerator.busyInf import BusyInf



def timeParser(time):
    time = int(time)
    hours = format(int(time / 3600), "02d")
    minutes = format(int((time % 3600) / 60), "02d")
    secs = format(int(time % 3600 % 60), "02d")
    return f"{hours}:{minutes}:{secs}"


if __name__ == "__main__":
    cpuInstance=0
    memoryInstance=0
    parser = argparse.ArgumentParser(
        description="Offline Profiling, testing part",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--app",
        "-a",
        dest="app",
        help="Application name. (`hotel-reserv`, `social-network`, `media-microsvc`)",
        default="hotel-reserv",
    )
    parser.add_argument(
        "--service",
        "-s",
        dest="service",
        help=(
            "Service name. Different applications have different available services:\n"
            "\thotel-reserv: `Recommendation`, `Search`\n"
            "\tsocial-network: `UserTimeline`, `ComposePost`, `HomeTimeline`\n"
            "\tmedia-microsvc: `ComposeReview`"
            "\tFitting all services: `All`\n"
        ),
        default="All",
    )
    parser.add_argument(
        "--profiling-data",
        "-p",
        dest="profiling_data",
        help="Path to save profiling data",
        default="AE/data"
    )
    parser.add_argument(
        "--cpu",
        "-c",
        dest="cpu",
        help="Number of CPU interferences.",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--mem",
        "-m",
        dest="mem",
        help="Number of memory interferences.",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--repeats",
        "-r",
        dest="repeats",
        help="Number of repeats.",
        default=1,
        type=int,
    )
    args = parser.parse_args()
    app = args.app
    service = args.service
    cpu = args.cpu
    mem = args.mem
    repeats = args.repeats
    data_path = os.path.join(args.profiling_data, f"data_{app}")
    namespace = app
    if service == "All":
        service = {
            "hotel-reserv": ["Search","Recommendation"],
            "sociel-network": ["ComposePost", "UserTimeline", "HomeTimeline"],
            "media-microsvc": ["ComposeReview"],
        }[app]
    else:
        service = [service]
    jaeger_host = {
        "hotel-reserv": "http://10.0.1.106:30095",
        "sociel-network": "http://10.0.1.106:30094",
        "media-microsvc": "http://10.0.1.106:30093",
    }[app]
    entry_point = {
        "hotel-reserv": "frontend",
        "sociel-network": "nginx-web-server",
        "media-microsvc": "nginx",
    }[app]
    nodes_for_test = [
        "node6", "node7"
    ]
    yaml_repo = {
        "hotel-reserv": "yamlRepository/hotelReservation",
        "social-network": "yamlRepository/socialNetwork",
        "media-microsvc": "yamlRepository/mediaMicroservice",
    }[app]
    app_img = {
        "hotel-reserv": "nicklin9907/erms:hotelreservation-1.0",
        "social-network": "nicklin9907/erms:socialnetwork-1.1",
        "media-microsvc": "nicklin9907/erms:mediamicroservice-1.0",
    }[app]
    replica_configs = {
        "hotel-reserv": {
            "Search": {"frontend": 6, "reservation": 3, "search": 2, "geo": 1, "user": 1, "rate": 1},
            "Recommendation": {"frontend": 8, "profile": 5, "recommendation": 2, "geo": 1, "user": 1, "rate": 1},
        },
        "social-network": {
            "ComposePost": {
                "nginx-thrift": 30,
                "compose-post-service": 3,
                "text-service": 1,
            },
            "UserTimeline": {
                "nginx-thrift": 30,
                "user-timeline-service": 2,
                "post-storage-service": 6,
            },
            "HomeTimeline": {"nginx-thrift": 30, "post-storage-service": 3},
        },
        "media-microsvc": {
            "ComposeReview": {
                "nginx-web-server": 10,
                "compose-review-service": 3,
                "movie-id-service": 2,
            }
        },
    }[app]
    workload_configs = {
        "hotel-reserv": {
            "Search": {
                "thread_num": 7,
                "conn_num": 14,
                "throughput": 5,
                "scripts": "wrk2/scripts/hotel-reservation/search.lua",
            },
            "Recommendation": {
                "thread_num": 7,
                "conn_num": 14,
                "throughput": 5,
                "scripts": "wrk2/scripts/hotel-reservation/recommendation.lua",
            },
        },
        "social-network": {
            "ComposePost": {
                "thread_num": 1,
                "conn_num": 2,
                "throughput": 4,
                "scripts": "wrk2/scripts/social-network/compose-post.lua",
            },
            "HomeTimeline": {
                "thread_num": 2,
                "conn_num": 3,
                "throughput": 6,
                "scripts": "wrk2/scripts/social-network/read-home-timeline.lua",
            },
            "UserTimeline": {
                "thread_num": 2,
                "conn_num": 4,
                "throughput": 7,
                "scripts": "wrk2/scripts/social-network/read-user-timeline.lua",
            },
        },
        "media-microsvc": {
            "ComposeReview": {
                "thread_num": 2,
                "conn_num": 3,
                "throughput": 6,
                "scripts": "wrk2/scripts/media-microservice/compose-review.lua",
            }
        },
    }[app]
    url = {
        "hotel-reserv": "http://10.0.1.106:30096",
        "social-network": "http://10.0.1.106:30628",
        "media-microsvc": "http://10.0.1.106:30092",
    }[app]
    DEPLOYER = Deployer(namespace, 0.1, "100Mi", nodes_for_test, yaml_repo, app_img)

    os.system(f"mkdir tmp")
    os.system(f"mkdir log")
    os.system(f"rm -rf {data_path}")
    os.system(f"mkdir -p {data_path}")
    # Prepare interference generator
    # cpuInterGenerator = BusyInf(nodes_for_test, 0.4, "10Mi", "cpu", ["36000s"])
    # memoryInterGenerator = BusyInf(
    #     nodes_for_test, 0.01, "800Mi", "memory", ["36000s", "wired", "100000s"]
    # )

    # Prepare data collector
    dataCollector = OfflineProfilingDataCollector(
        namespace,
        jaeger_host,
        entry_point,
        "http://10.0.1.106:30090",
        nodes_for_test,
        data_path,
        duration=80,
    )

    # Time estimation
    totalRound = repeats * 20 * len(service) * cpu * mem
    passedRound = 1
    usedTime = 0
    # Test each service in the configuration
    for cpuInstance in range(1):
        for memoryInstance in range(1):
            for clientNum in [3, 6]:  # 3과 6 → 30, 60 throughput
                for repeat in range(repeats):  # epoch_0, epoch_1, epoch_2
                    DEPLOYER.delete_app()
                    for svc in service:
                        containers = replica_configs[svc]
                        currentWorkloadConfig = workload_configs[svc]
                        workloadGenerator = StaticWorkloadGenerator(
                            currentWorkloadConfig["thread_num"],
                            currentWorkloadConfig["conn_num"],
                            40,
                            currentWorkloadConfig["throughput"],
                            "wrk2/wrk",
                            currentWorkloadConfig["scripts"],
                            url,
                        )
                    log.info("Deploying Application...")
                    
                    merged_containers = {}
                    for svc in service:
                        merged_containers.update(replica_configs[svc])
                    DEPLOYER.deploy_app(merged_containers)
                    dataCollector.wait_until_done()

                    roundStartTime = time.time()
                    if passedRound != 0:
                        avgTime = usedTime / passedRound
                        log.info(
                            f"Used time: {timeParser(usedTime)}, "
                            f"Avg. round time: {timeParser(avgTime)}, "
                            f"Left time estimation: {timeParser(avgTime * (totalRound - passedRound))}"
                        )

                    startTime = int(time.time())

                    for svc in service:
                        currentWorkloadConfig = workload_configs[svc]
                        testName = f"[{svc}]{repeat}r{clientNum}c[{cpuInstance}u,{memoryInstance}m]"
                        workloadGenerator = StaticWorkloadGenerator(
                            currentWorkloadConfig["thread_num"],
                            currentWorkloadConfig["conn_num"],
                            40,
                            currentWorkloadConfig["throughput"],
                            "wrk2/wrk",
                            currentWorkloadConfig["scripts"],
                            url,
                        )
                        log.info(f"Running workload for {svc}")
                        print("Round:",passedRound)
                        workloadGenerator.generateWorkload(testName, clientNum)

                    test_data = {
                        "repeat": repeat,
                        "start_time": startTime,
                        "service": "All",
                        "cpu_inter": cpuInstance * 0.4,
                        "mem_inter": memoryInstance * 800,
                        "target_throughput": clientNum * sum([workload_configs[s]["throughput"] for s in service]),
                        "test_name": f"All-{repeat}-{clientNum}",
                    }
                    dataCollector.collect_data_async(test_data)

                    passedRound += 1
                    usedTime += time.time() - roundStartTime
            
        

    dataCollector.wait_until_done()
    DEPLOYER.delete_app()
   
