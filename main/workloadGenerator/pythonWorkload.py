from threading import Thread
from time import time, sleep
from train-ticket-auto-query import Query  

class PythonWorkloadGenerator:
    def __init__(self, duration, target_throughput, client_num, base_url):
        self.duration = duration
        self.target_throughput = target_throughput
        self.client_num = client_num
        self.base_url = base_url

    def generateWorkload(self, testName, clientNum):
        threads = []
        for _ in range(clientNum):
            t = Thread(target=self.run_client)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    def run_client(self):
        q = Query(self.base_url)
        if not q.login():
            return
        start_time = time()
        while time() - start_time < self.duration:
            try:
                trip_ids = q.query_high_speed_ticket()
                if trip_ids:
                    q.preserve("Shang Hai", "Su Zhou", trip_ids)
            except Exception as e:
                print(f"Exception in workload: {e}")
            sleep(1.0 / self.target_throughput)  # QPS 제어
