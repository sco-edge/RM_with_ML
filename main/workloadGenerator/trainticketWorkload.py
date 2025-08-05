from threading import Thread
from time import time, sleep
from train_ticket_auto_query.queries import Query  
from train_ticket_auto_query.scenarios import query_and_preserve  # 내부에서 랜덤한 시나리오 실행
import logging

class TrainticketWorkloadGenerator:
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
            logging.warning("Login failed")
            return
        start_time = time()
        while time() - start_time < self.duration:
            try:
                query_and_preserve(q)
            except Exception as e:
                logging.warning(f"Exception in workload: {e}")
                import traceback
                logging.warning(f"Traceback: {traceback.format_exc()}")
            sleep(1.0 / self.target_throughput)  # QPS 제어
