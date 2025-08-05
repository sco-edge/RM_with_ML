from testing.testCollection_once import full_init, start_test
from configs import app
import logging

# 로깅 레벨을 DEBUG로 설정
logging.basicConfig(level=logging.DEBUG)

ports = {"social": 30628, "media": 30092, "hotel": 30096, "train": 32677}
app= "train"
# full_init(app, ports[app])

# continues=True: Program will append to the end of existing data
# continues=False: Program will overwrite existing data.
start_test(continues=True)