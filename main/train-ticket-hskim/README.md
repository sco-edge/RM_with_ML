train-ticket microservices benchmark - kubernetes deployment

0. base source : vaqurkhan
https://github.com/vaquarkhan/train-ticket/tree/master

1. jar build
mvn clean package

o Development Tool/Environment
Docker version 28.0.1, build 068a01e
spring boot framework 1.5.3.RELEASE
javac 1.8.0_442
Apache Maven 3.9.9 (8e8579a9e76f7d015ee5ec7bfcdc97d260186937)

javac path: /usr/lib/jvm/java-8-openjdk-amd64/bin/javac
Maven home: /usr/opt/apache-maven-3.9.9
Java Development Kit (JDK)	Maven 3.9+ requires JDK 8 or above to execute.

2. docker image build
docker compose build
with docker-compose.yml

3. docker image register(push into dockethub)
docker tag ts/ts-login-service swlove2024/ts-login-service
docker push swlove2024/ts-login-service

4. kubernetes deployment
deployment, serviec .yaml 

5. istio, jaeger tracing
curl -L https://istio.io/downloadIstio | ISTIO_VERSION=1.25.2 TARGET_ARCH=x86_64 sh -
cd istio-1.25.2
export PATH=$PWD/bin:$PATH

// istio install
istioctl install --set profile=demo --set meshConfig.defaultConfig.tracing.zipkin.address="jaeger-collector.istio-system.svc.cluster.local:9411" --set meshConfig.defaultConfig.tracing.sampling=100 -y

// jaeger install
kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.26/samples/addons/jaeger.yaml

■ tracing.yaml
apiVersion: install.istio.io/v1alpha1
kind: IstioOperator
spec:
  meshConfig:
    enableTracing: true
    defaultConfig:
      tracing: # <---
        zipkin: 
          address: "jaeger-collector.istio-system.svc.cluster.local:9411"
    extensionProviders:
    - name: jaeger
      opentelemetry:
        port: 4317
        service: jaeger-collector.istio-system.svc.cluster.local

$ istioctl install -f ./tracing.yaml --skip-confirmation
 
■ Enable tracing by applying the following configuration:
$ kubectl apply -f - <<EOF
apiVersion: telemetry.istio.io/v1
kind: Telemetry
metadata:
  name: mesh-default
  namespace: istio-system
spec:
  tracing:
  - providers:
    - name: jaeger
EOF


■ confirm the value of address: and extensionProviders: 
kubectl edit configmap istio -n istio-system 

apiVersion: v1
data:
  mesh: |-
    accessLogFile: /dev/stdout
    defaultConfig:
      discoveryAddress: istiod.istio-system.svc:15012
      tracing:
        sampling: 100
        zipkin:  # <---
          address: jaeger-collector.istio-system.svc.cluster.local:9411  # <---
    defaultProviders:
      metrics:
      - prometheus
    enablePrometheusMerge: true
    extensionProviders:
    - envoyOtelAls:
        port: 4317
        service: opentelemetry-collector.observability.svc.cluster.local
      name: otel
    - name: jaeger    # <---
      opentelemetry:  # <---
        port: 4317       # <---
        service: jaeger-collector.istio-system.svc.cluster.local  # <---
    rootNamespace: istio-system
    trustDomain: cluster.local
  meshNetworks: 'networks: {}'
kind: ConfigMap
metadata:
.......






