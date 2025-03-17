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

ISSUE 1: MAC OS Architecture (aarch64) is not match with Docker Images in Deathstarbench 
=> Run it with Linux systems (12CPU core, RAM 15GB)
