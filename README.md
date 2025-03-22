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
