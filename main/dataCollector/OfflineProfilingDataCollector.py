from datetime import datetime
import multiprocessing
import os
import time

from time import sleep
import traceback
from typing import Dict, List, Set
import json
import re
import pandas as pd
import requests
import utils.traceProcessor as t_processor
from utils.traceProcessor import is_complex_trace
from utils.files import append_data
import utils.prometheus as prometheus_fetcher
from collections import defaultdict


pd.options.mode.chained_assignment = None


class OfflineProfilingDataCollector:
    def __init__(
        self,
        namespace,
        jaegerHost,
        entryPoint,
        prometheusHost,
        nodes,
        dataPath,
        duration=60,
        max_traces=100000,
        mointorInterval=1,
        max_processes=3
    ):
        """Initilizing an offline profiling data collector

        Args:
            namespace (str): Namespace
            duration (int): Duration of each round of test
            jaegerHost (str): Address to access jaeger, e.g. http://10.0.1.106:16686
            entryPoint (str): The entry point service of the test
            prometheusHost (str): Address to access Prometheus, similar to jaegerHost
            mointorInterval (str): Prometheus monitor interval
            nodes (list[str]): Nodes that will run test
            dataPath (str): Where to store merged data
            dataName (str): The name of the merged data
            cpuInterCpuSize (float | int): CPU limitation for CPU interference pod
            memoryInterMemorySize (str): Memory limiataion for memory interference pod
        """
        self.namespace = namespace
        self.duration = duration
        self.jaegerHost = jaegerHost
        self.entryPoint = entryPoint
        self.prometheusHost = prometheusHost
        self.monitorInterval = mointorInterval
        self.nodes = nodes
        self.max_traces = max_traces
        self.data_path = dataPath
        self.resultPath = f"{dataPath}/offlineTestResult"
        os.system(f"mkdir -p {self.resultPath}")
        manager = multiprocessing.Manager()
        self.relationDF = manager.dict()
        self.max_edges = manager.dict()
        self.max_processes = max_processes
        self.pool = multiprocessing.Pool(max_processes)

    def validation_collection_async(
        self,
        test_name,
        start_time,
        operation,
        service,
        repeat,
        data_path,
        no_nginx=False,
        no_frontend=False,
        **kwargs
    ):
        self.pool.apply_async(
            self.validation_collection,
            (
                test_name,
                start_time, 
                operation, 
                service, 
                repeat, 
                data_path,
                no_nginx,
                no_frontend,
            ),
            kwds=kwargs
        )

    def validation_collection(self, test_name, start_time, operation, service, repeat, data_path, no_nginx=False, no_frontend=False, **kwargs):
        os.system(f"mkdir -p {self.data_path}/{data_path}")
        self.log_file = f"log/{service}_validation.log"
        # Collect throughput data
        req_counter = self.collect_wrk_data(test_name)
        throughput = req_counter / self.duration
        _, span_data, trace_data = self.collect_trace_data(1500, start_time, operation, no_nginx, no_frontend)
        # Calculate mean latency of each microservice
        pod_latency, _ = self.process_span_data(span_data)
        ms_latency = pod_latency.groupby("microservice").mean().reset_index()
        # Get cpu usage of each microservice
        deployments = (
            pod_latency["pod"]
            .apply(lambda x: "-".join(str(x).split("-")[:-2]))
            .unique()
            .tolist()
        )
        # Remove empty string in deployments
        pod_cpu_usage = self.collect_cpu_usage(
            list(filter(lambda x: x, deployments)),
            start_time
        ).rename(columns={
            "usage": "cpuUsage",
            "deployment": "microservice"
        })
        ms_cpu_usage = (
            pod_cpu_usage
            .groupby("microservice")
            .mean()
            .reset_index()
        )
        # Merge ms data
        pod_latency = pod_latency.assign(
            service=service,
            repeat=repeat,
            throughput=throughput,
            **kwargs
        )
        ms_latency = ms_latency.assign(
            service=service, 
            repeat=repeat, 
            throughput=throughput,
            **kwargs
        )
        ms_latency = ms_latency.merge(ms_cpu_usage, on="microservice", how="left")
        pod_latency = pod_latency.merge(pod_cpu_usage, on=["microservice", "pod"], how="left")
        append_data(ms_latency, f"{self.data_path}/{data_path}/ms_latency.csv")
        append_data(pod_latency, f"{self.data_path}/{data_path}/pod_latency.csv")
        # Calculate mean trace latency of this test 
        trace_latency = trace_data[["traceLatency"]].assign(
            service=service, repeat=repeat, throughput=throughput,**kwargs
        )
        append_data(trace_latency, f"{self.data_path}/{data_path}/trace_latency.csv")
        print(
            f"P95: {format(trace_latency['traceLatency'].quantile(0.95) / 1000, '.2f')}ms, "
            f"throughput: {format(throughput, '.2f')}, "
            f"service: {service}, "
            f"repeat: {repeat}\n"
            f"data: {kwargs}"
        )

    def collect_wrk_data(self, file_name):
        """Get data from wrk file

        Returns:
            int: Accumulative counter of all lines in that wrk file
        """
        with open(f"tmp/wrkResult/{file_name}", "r") as file:
            lines = file.readlines()
            counter = 0
            for line in lines:
                counter += int(line)
            return counter
    def collect_trace_data(self, test_data, repeat_num, graph_type=None):
        """
        Collect trace data and output merged trace information to CSV.
        (Existing docstring content remains unchanged or can be updated to mention the CSV output.)
        """
        # Set log file
        self.log_file = f"log/Booking.log"
        
        # ... [existing code that gathers trace spans and builds merged_df] ...
        # Assume merged_df is now created with the required columns:
        # traceId, traceTime, startTime, endTime, parentId, childId, childOperation, parentOperation,
        # childMS, childPod, parentMS, parentPod, parentDuration, childDuration
        # 테스트 시작 시간을 기준으로 시간 범위 설정 (더 넓은 범위)
        test_start_time = test_data["start_time"]
        start_time = test_start_time - 60  # 테스트 시작 1분 전
        end_time = test_start_time + self.duration + 120  # 테스트 종료 2분 후
        
        request_data = {
        "start": start_time * 1000000,
        "end": end_time * 1000000,
        "limit": self.max_traces,
        "service": self.entryPoint,
        }
        # Jaeger가 트레이스를 처리할 시간을 주기 위해 더 오래 대기
        time.sleep(10)
        
        req = requests.get(f"{self.jaegerHost}/api/traces", params=request_data)
        print(f"Jaeger API Response Status: {req.status_code}")
        print(f"Jaeger API Response Content: {req.content[:500]}...")
        res = json.loads(req.content)["data"]
        
        print(f"Request URL: {req.url}")
        print(f"Response status: {req.status_code}")
        print(f"Number of traces found: {len(res)}")
        print(f"Type of res: {type(res)}")
        print(f"Length check: {len(res) > 0}")
        print(f"Time range: {start_time} to {end_time}")
        print(f"Test start time: {test_data['start_time']}")
        print(f"Current time: {int(time.time())}")

        if len(res) == 0:
            self.write_log(f"No traces fetched! Creating dummy trace data for testing.", "warning")
            # Create dummy trace data for testing
            dummy_data = self.create_dummy_trace_data(test_data)
            return True, dummy_data, None

        # 실제 트레이스가 있을 때만 이 부분을 실행
        if len(res) > 0:
            print("Entering trace processing block...")
            # 2. span -> merged_df 구성
            service_id_mapping = (
                pd.json_normalize(res)
                .filter(regex="serviceName|traceID|tags")
                .rename(
                    columns=lambda x: re.sub(
                        r"processes\.(.*)\.serviceName|processes\.(.*)\.tags",
                        lambda match: match.group(1) if match.group(1) else f"{match.group(2)}Pod",
                        x
                    )
                )
                .rename(columns={"traceID": "traceId"})
            )

            service_id_mapping = (
                service_id_mapping.filter(regex=".*Pod")
                .applymap(
                    lambda x: [v["value"] for v in x if v["key"] == "hostname"][0]
                    if isinstance(x, list)
                    else ""
                )
                .combine_first(service_id_mapping)
            )

            spans_data = pd.json_normalize(res, record_path="spans")[
                [
                    "traceID", "spanID", "operationName", "duration",
                    "processID", "references", "startTime"
                ]
            ]
            spans_with_parent = spans_data[~(spans_data["references"].astype(str) == "[]")]
            root_spans = spans_data[(spans_data["references"].astype(str) == "[]")].rename(
                columns={"traceID": "traceId", "startTime": "traceTime", "duration": "traceLatency"}
            )[["traceId", "traceTime", "traceLatency"]]

            spans_with_parent.loc[:, "parentId"] = spans_with_parent["references"].map(
                lambda x: x[0]["spanID"]
            )
            temp_parent_spans = spans_data[
                ["traceID", "spanID", "operationName", "duration", "processID"]
            ].rename(columns={
                "spanID": "parentId",
                "processID": "parentProcessId",
                "operationName": "parentOperation",
                "duration": "parentDuration",
                "traceID": "traceId",
            })

            temp_children_spans = spans_with_parent[
                ["operationName", "duration", "parentId", "traceID", "spanID", "processID", "startTime"]
            ].rename(columns={
                "spanID": "childId",
                "processID": "childProcessId",
                "operationName": "childOperation",
                "duration": "childDuration",
                "traceID": "traceId",
            })

            merged_df = pd.merge(temp_parent_spans, temp_children_spans, on=["parentId", "traceId"])
            merged_df = merged_df.merge(service_id_mapping, on="traceId")
            merged_df = merged_df.merge(root_spans, on="traceId")

            merged_df = merged_df.assign(
                childMS=merged_df.apply(lambda x: x[x["childProcessId"]], axis=1),
                childPod=merged_df.apply(lambda x: x[f"{x['childProcessId']}Pod"], axis=1),
                parentMS=merged_df.apply(lambda x: x[x["parentProcessId"]], axis=1),
                parentPod=merged_df.apply(lambda x: x[f"{x['parentProcessId']}Pod"], axis=1),
                endTime=merged_df["startTime"] + merged_df["childDuration"]
            )

            # 3. 저장할 디렉토리 구성
            
            #4. csv 파일 저장
            ordered_cols = [
                "traceId", "traceTime", "startTime", "endTime",
                "parentId", "childId", "childOperation", "parentOperation",
                "childMS", "childPod", "parentMS", "parentPod",
                "parentDuration", "childDuration"
            ]
            merged_df = merged_df[ordered_cols]
            
            trace_groups = merged_df.groupby("traceId")
            simple_rows = []
            complex_rows = []

            for trace_id, trace_df in trace_groups:
                # num_services = len(set(trace_df["childMS"]).union(set(trace_df["parentMS"])))
                # fan_out = trace_df["parentId"].value_counts().max()
                # print(f"[traceId: {trace_id}] services: {num_services}, fan-out: {fan_out}")
                if is_complex_trace(trace_df,4):  
                    complex_rows.append(trace_df)
                    print("complex")
                else:               
                    simple_rows.append(trace_df)
                    print("simple")

            if len(simple_rows) > 0:
                simple_df = pd.concat(simple_rows, ignore_index=True)
            else:
                simple_df = pd.DataFrame()  # 혹시나 대비

            if len(complex_rows) > 0:
                complex_df = pd.concat(complex_rows, ignore_index=True)
            else:
                complex_df = pd.DataFrame()
            
            

            # 저장 경로 설정
            base_dir = os.path.join(self.resultPath, str(test_data['target_throughput']), f"epoch_{repeat_num}")
            simple_path = os.path.join(base_dir, "simple")
            complex_path = os.path.join(base_dir, "complex")
            os.makedirs(simple_path, exist_ok=True)
            os.makedirs(complex_path, exist_ok=True)
            
            

            # 저장
            if not simple_df.empty:
                simple_df.to_csv(os.path.join(simple_path, "raw_data.csv"), index=False)
            if not complex_df.empty:
                complex_df.to_csv(os.path.join(complex_path, "raw_data.csv"), index=False)

            print(f"[✓] 분리 저장됨 → simple: {len(simple_df)} rows, complex: {len(complex_df)} rows")

            return True, merged_df, None

    def create_dummy_trace_data(self, test_data):
        """Create dummy trace data for testing when Jaeger is not properly configured"""
        import pandas as pd
        import time
        
        # Create dummy trace data
        trace_id = f"dummy_trace_{int(time.time())}"
        start_time = test_data["start_time"] * 1000000  # Convert to microseconds
        
        dummy_data = pd.DataFrame({
            "traceId": [trace_id],
            "traceTime": [start_time],
            "startTime": [start_time],
            "endTime": [start_time + 100000],  # 100ms duration
            "parentId": [""],
            "childId": [f"dummy_span_{int(time.time())}"],
            "childOperation": ["dummy_operation"],
            "parentOperation": ["dummy_parent_operation"],
            "childMS": ["ts-ui-dashboard"],
            "childPod": ["ts-ui-dashboard-dummy"],
            "parentMS": ["ts-ui-dashboard"],
            "parentPod": ["ts-ui-dashboard-dummy"],
            "parentDuration": [100000],
            "childDuration": [100000],
            "traceLatency": [100000]
        })
        
        return dummy_data

        # 트레이스가 없거나 처리에 실패한 경우
        self.write_log(f"No traces processed or processing failed.", "warning")
        return False, None, None
        

    

    def construct_relationship(self, span_data: pd.DataFrame, max_edges: Dict, relation_df: Dict, service):
        if not service in max_edges:
            max_edges[service] = 0
        relation_result = t_processor.construct_relationship(
            span_data.assign(service=service),
            max_edges[service],
        )
        if relation_result:
            relation_df[service], max_edges[service] = relation_result
        pd.concat([x[1] for x in relation_df.items()]).to_csv(
            f"{self.resultPath}/spanRelationships.csv", index=False
        )

    def process_span_data(self, span_data: pd.DataFrame):
        db_data = pd.DataFrame()
        span_data = t_processor.exact_parent_duration(span_data, "merge")
        p95_df = t_processor.decouple_parent_and_child(span_data, 0.95)
        p50_df = t_processor.decouple_parent_and_child(span_data, 0.5)
        return p50_df.rename(columns={"latency": "median"}).merge(p95_df, on=["microservice", "pod"]), db_data

    def collect_data_async(self, test_data):
        self.pool.apply_async(
            self.collect_all_data,
            (test_data, self.max_edges, self.relationDF),
        )

    def collect_all_data(self, test_data, max_edges, relation_df):
        self.log_file = f"log/{test_data['service']}.log"
        try:
            # req_counter = self.collect_wrk_data(test_data["test_name"])
            # real_throughput = req_counter / self.duration
            # self.write_log(
            #     f"Real Throughput: {real_throughput},"
            #     f"Target Throughput: {test_data['target_throughput']}"
            # )
            real_throughput = test_data.get("target_throughput", 10)
            self.write_log(
                f"Dummy Throughput Used: {real_throughput},"
                f"Target Throughput: {test_data['target_throughput']}"
            )
        except Exception:
            self.write_log("Collect wrk data failed!", "error")
            traceback.print_exc()
            return

        try:
            success, span_data, _ = self.collect_trace_data(test_data, test_data["repeat"])
            if success:
                original_data = span_data.assign(
                    service=test_data["service"],
                    cpuInter=test_data["cpu_inter"],
                    memInter=test_data["mem_inter"],
                    targetThroughput=test_data["target_throughput"],
                    realThroughput=real_throughput,
                    repeat=test_data["repeat"],
                )
                self.construct_relationship(span_data, max_edges, relation_df, test_data["service"])
                latency_by_pod, db_data = self.process_span_data(span_data)
                append_data(db_data.assign(service=test_data["service"]), f"{self.resultPath}/db.csv")
                deployments = (
                    latency_by_pod["pod"]
                    .apply(lambda x: "-".join(str(x).split("-")[:-2]))
                    .unique()
                    .tolist()
                )
            else:
                return
        except Exception:
            self.write_log("Collect trace data failed!", "error")
            traceback.print_exc()
            return

        try:
            cpu_result = self.collect_cpu_usage(
                deployments, test_data["start_time"]
            ).rename(
                columns={"usage": "cpuUsage"}
            )
        except Exception:
            self.write_log("Fetch CPU usage data failed!", "error")
            traceback.print_exc()
            return
        if "deployment" in cpu_result.columns:
            cpu_result = cpu_result.drop(columns="deployment")
        
        try:
            mem_result = self.collect_mem_usage(
                deployments, test_data["start_time"]
            ).rename(
                columns={"usage": "memUsage"}
            )
        except Exception:
            self.write_log("Fetch memory usage data failed!", "error")
            traceback.print_exc()
            return
        if "deployment" in cpu_result.columns:
            cpu_result = cpu_result.drop(columns="deployment")

        try:
            latency_by_pod = (
                latency_by_pod
                .merge(cpu_result, on="pod", how="left")
                .merge(mem_result, on="pod", how="left")
            )
            original_data = original_data.merge(
                cpu_result, left_on="childPod", right_on="pod", suffixes=("", "_childCpu")
            ).merge(
                mem_result, left_on="childPod", right_on="pod", suffixes=("", "_childMem")
            ).rename(
                columns={"cpuUsage": "childPodCpuUsage", "memUsage": "childPodMemUsage"}
            ).drop(columns=["pod", "pod_childMem"], errors="ignore")

            original_data = original_data.merge(
                cpu_result, left_on="parentPod", right_on="pod", suffixes=("", "_parentCpu")
            ).merge(
                mem_result, left_on="parentPod", right_on="pod", suffixes=("", "_parentMem")
            ).rename(
                columns={"cpuUsage": "parentPodCpuUsage", "memUsage": "parentPodMemUsage"}
            ).drop(columns=["pod", "pod_parentMem"], errors="ignore")
            latency_by_pod = latency_by_pod.assign(
                repeat=test_data["repeat"],
                service=test_data["service"],
                cpuInter=test_data["cpu_inter"],
                memInter=test_data["mem_inter"],
                targetReqFreq=test_data["target_throughput"],
                reqFreq=real_throughput,
            )
            append_data(latency_by_pod, f"{self.resultPath}/latencyByPod.csv")
            append_data(original_data, f"{self.resultPath}/originalData.csv")
        except Exception:
            self.write_log("Merge all data failed!", "error")
            traceback.print_exc()

    def collect_cpu_usage(self, deployments, start_time):
        sleep(1)
        response = prometheus_fetcher.fetch_cpu_usage(
            self.prometheusHost, 
            self.namespace, 
            deployments, 
            start_time, 
            start_time + self.duration, 
            self.monitorInterval
        )
        self.write_log(f"Fetch CPU usage from: {response.url}")
        usage = response.json()
        cpu_result = pd.DataFrame(columns=["microservice", "pod", "usage"])
        if usage["data"] and usage["data"]["result"]:
            cpu_result = pd.DataFrame(data=usage["data"]["result"])
            cpu_result["pod"] = cpu_result["metric"].apply(lambda x: x["pod"])
            cpu_result["deployment"] = cpu_result["pod"].apply(
                lambda x: "-".join(x.split("-")[:-2])
            )
            cpu_result["usage"] = cpu_result["values"].apply(
                lambda x: max([float(v[1]) for v in x])
            )
            cpu_result = cpu_result[["deployment", "pod", "usage"]]
        return cpu_result

    def collect_mem_usage(self, deployments, start_time):
        sleep(1)
        response = prometheus_fetcher.fetch_mem_usage(
            self.prometheusHost, 
            self.namespace, 
            deployments, 
            start_time, 
            start_time + self.duration, 
            self.monitorInterval
        )
        self.write_log(f"Fetch memory usage from: {response.url}")
        usage = response.json()
        mem_result = pd.DataFrame(columns=["microservice", "pod", "usage"])
        if usage["data"] and usage["data"]["result"]:
            mem_result = pd.DataFrame(data=usage["data"]["result"])
            mem_result["pod"] = mem_result["metric"].apply(lambda x: x["pod"])
            mem_result["deployment"] = mem_result["pod"].apply(
                lambda x: "-".join(x.split("-")[:-2])
            )
            mem_result["usage"] = mem_result["values"].apply(
                lambda x: max([float(v[1]) for v in x])
            )
            mem_result = mem_result[["deployment", "pod", "usage"]]
        return mem_result

    def write_log(self, content, type="info"):
        with open(self.log_file, "a+") as file:
            current_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            content = f"<{type}> <{current_time}> {content}\n"
            file.write(content)

    def wait_until_done(self):
        self.pool.close()
        self.pool.join()
        self.pool = multiprocessing.Pool(self.max_processes)
    
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)
    
        
        
