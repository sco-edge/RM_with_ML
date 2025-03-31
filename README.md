# Problem of Machine Learning approaches for Resource mangement system
------------

## Plans
Step 1: Clustering VMs and deploy Deathstarbench(ASPLOS'19)
Step 2: Run Erms(ASPLOS'23)
Step 3: Implement Deployment and Resource Provisioning for Derm(ISCA'24)

### Week 1: Clustering with Vagrant
Control Plane: 2 CPU cores and 2GB RAM
Three Slave nodes: 1 CPU cores and 1.5GB RAM

Start with vagrant up with Vagrantfile

Issue 1: MAC OS Architecture (aarch64) is not match with Docker Images in Deathstarbench
=> Run it with Linux systems (12CPU core, RAM 15GB)

Start with minikube

Issue 1:
yamlRepository/hotelReservation/non-test/consul_Deployment.yaml
image: consul:latest -> image: consul:1.8.4

### Week2: Execute Erms with Minikube
Minikube(Control Plane with Pods): CPU 16 RAM 32GB

#### Deploy
source ksh/bin/activate

pip install -r requirement.txt(버전 문제로 인해 개별적인 설치 진행)

kubectl create namespace hotel-reserv(namespace를 자동으로 생성을 안해줌, readme나 issue에는 없음)

python3 main.py

(main.py -> testCollection.py -> full_init() -> deployer.py -> hotel-global.yaml)

#### Testing
kubectl create namespace interference(CPU, Memory inteference 발생용 pod가 배포될 namespace)

chmod +x wrk2/wrk    (워크로드 실행 권한 부여)

python3 profiling.py

full_init(app, port number 명시) && start_test -> testCollection.py -> hotel-testing.yaml

hotel-testing.yaml에서 service종류를 명시 ex) Search, Recommendation, Login, Reservation등
이 각 부하는 wrk2/scripts/hotel-reservation/ 에 있는 각 lua script들로 실행됨

####Issues 
Issue1: libssl version conflict in wrk2

wget http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.24_amd64.deb
sudo dpkg -i libssl1.1_1.1.1f-1ubuntu2.24_amd64.deb

Issue2: Connect refuse for worload

hotel-testing.yaml
localhost -> 192.168.49.2

Issue3: delete namespace error

kubectl get namespace hotel-reserv -o json > tmp-ns.json
sed -i '/"kubernetes"/d' tmp-ns.json
kubectl replace --raw "/api/v1/namespaces/hotel-reserv/finalize" -f ./tmp-ns.json


### Derm

#### fitting and learning
python3 ./AE/scripts/print_ms_info.py  config값 계산(컨테이너수는 직접 pod확인해서 세야 함)

python3 ./Derm-master/profiling/curve_cdf.py (서비스 별 fitting에 필요한 파라미터 계산)

