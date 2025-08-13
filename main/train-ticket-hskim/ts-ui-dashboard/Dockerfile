
FROM opentracing/nginx-opentracing:latest

# 환경변수 설정
ENV JAEGER_AGENT_HOST=jaeger
ENV JAEGER_AGENT_PORT=6831
ENV JAEGER_SAMPLE_RATIO=1.0
ENV OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317/v1/traces
ENV OTEL_SERVICE_NAME=ts-ui-dashboard
ENV OTEL_TRACES_SAMPLER=always_on
ENV OTEL_TRACES_SAMPLER_ARG=1.0

# nginx 설정 복사
COPY nginx.conf /etc/nginx/nginx.conf
COPY static /usr/share/nginx/html
COPY jaeger-config.json /etc/jaeger-config.json

# 기본 설정 파일 삭제 (nginx.conf를 덮어쓰는 것을 방지)
RUN rm -f /etc/nginx/conf.d/default.conf

EXPOSE 8080

CMD ["nginx", "-g", "daemon off;"]