#!/bin/bash

# 개선된 병렬 빌드 스크립트 v2
# 사용법: ./parallel_build_v2.sh [max_jobs]

set -e

# 최대 동시 작업 수 (기본값: 4)
MAX_JOBS=${1:-4}
BASE_DIR="/home/masternode/RM_with_ML/main/train-ticket-hskim"
LOG_DIR="$BASE_DIR/build_logs_v2"
BUILD_LOG="$LOG_DIR/build_summary.log"
COMPLETED_FILE="$LOG_DIR/completed_services.txt"
FAILED_FILE="$LOG_DIR/failed_services.txt"

# 로그 디렉토리 생성
mkdir -p "$LOG_DIR"

# 재빌드할 서비스 목록 (MongoDB/MySQL 제외)
SERVICES=(
    "ts-admin-basic-info-service"
    "ts-admin-order-service"
    "ts-admin-route-service"
    "ts-admin-travel-service"
    "ts-admin-user-service"
    "ts-assurance-service"
    "ts-basic-service"
    "ts-cancel-service"
    "ts-config-service"
    "ts-consign-price-service"
    "ts-consign-service"
    "ts-contacts-service"
    "ts-execute-service"
    "ts-food-map-service"
    "ts-food-service"
    "ts-inside-payment-service"
    "ts-news-service"
    "ts-notification-service"
    "ts-order-other-service"
    "ts-order-service"
    "ts-payment-service"
    "ts-preserve-other-service"
    "ts-preserve-service"
    "ts-price-service"
    "ts-rebook-service"
    "ts-route-plan-service"
    "ts-route-service"
    "ts-seat-service"
    "ts-security-service"
    "ts-station-service"
    "ts-ticket-office-service"
    "ts-ticketinfo-service"
    "ts-train-service"
    "ts-travel-plan-service"
    "ts-travel-service"
    "ts-travel2-service"
    "ts-ui-dashboard"
    "ts-voucher-service"
)

# 이미 완료된 서비스들 제외
if [ -f "$COMPLETED_FILE" ]; then
    COMPLETED_SERVICES=($(cat "$COMPLETED_FILE"))
    echo "📋 이미 완료된 서비스들: ${COMPLETED_SERVICES[*]}"
    
    # 완료된 서비스들을 제외한 새로운 목록 생성
    REMAINING_SERVICES=()
    for service in "${SERVICES[@]}"; do
        if [[ ! " ${COMPLETED_SERVICES[@]} " =~ " ${service} " ]]; then
            REMAINING_SERVICES+=("$service")
        fi
    done
    SERVICES=("${REMAINING_SERVICES[@]}")
    echo "🔄 남은 서비스 수: ${#SERVICES[@]}"
fi

echo "🚀 병렬 빌드 v2 시작 - 최대 동시 작업: $MAX_JOBS" | tee "$BUILD_LOG"
echo "📅 시작 시간: $(date)" | tee -a "$BUILD_LOG"
echo "📊 총 서비스 수: ${#SERVICES[@]}" | tee -a "$BUILD_LOG"
echo "==================================" | tee -a "$BUILD_LOG"

# 빌드 함수
build_service() {
    local service=$1
    local service_dir="$BASE_DIR/$service"
    local log_file="$LOG_DIR/${service}_build.log"
    
    echo "🔨 빌드 시작: $service" | tee -a "$BUILD_LOG"
    
    if [ ! -d "$service_dir" ]; then
        echo "❌ 디렉토리 없음: $service" | tee -a "$BUILD_LOG"
        echo "$service" >> "$FAILED_FILE"
        return 1
    fi
    
    cd "$service_dir"
    
    # Maven 빌드
    if mvn clean install -DskipTests > "$log_file" 2>&1; then
        echo "✅ 빌드 성공: $service" | tee -a "$BUILD_LOG"
        
        # Docker 이미지 생성
        local version="v2"
        if docker build -t "ksh6283/$service:$version" . >> "$log_file" 2>&1; then
            echo "🐳 Docker 이미지 생성 성공: $service:$version" | tee -a "$BUILD_LOG"
            
            # Docker Hub 푸시 (재시도 로직 포함)
            local retry_count=0
            local max_retries=3
            
            while [ $retry_count -lt $max_retries ]; do
                if docker push "ksh6283/$service:$version" >> "$log_file" 2>&1; then
                    echo "📤 Docker Hub 푸시 성공: $service:$version" | tee -a "$BUILD_LOG"
                    echo "$service" >> "$COMPLETED_FILE"
                    return 0
                else
                    ((retry_count++))
                    echo "⚠️  Docker Hub 푸시 실패 (시도 $retry_count/$max_retries): $service" | tee -a "$BUILD_LOG"
                    if [ $retry_count -lt $max_retries ]; then
                        sleep 10
                    fi
                fi
            done
            
            echo "❌ Docker Hub 푸시 최종 실패: $service" | tee -a "$BUILD_LOG"
            echo "$service" >> "$FAILED_FILE"
            return 1
        else
            echo "❌ Docker 이미지 생성 실패: $service" | tee -a "$BUILD_LOG"
            echo "$service" >> "$FAILED_FILE"
            return 1
        fi
    else
        echo "❌ Maven 빌드 실패: $service" | tee -a "$BUILD_LOG"
        echo "$service" >> "$FAILED_FILE"
        return 1
    fi
}

# 병렬 실행을 위한 작업 큐
job_queue=()
completed=0
failed=0

# 모든 서비스에 대해 빌드 작업 추가
for service in "${SERVICES[@]}"; do
    # 현재 실행 중인 작업 수가 최대치에 도달하면 대기
    while [ ${#job_queue[@]} -ge $MAX_JOBS ]; do
        # 완료된 작업 확인
        for i in "${!job_queue[@]}"; do
            if ! kill -0 "${job_queue[$i]}" 2>/dev/null; then
                # 작업 완료, 큐에서 제거
                wait "${job_queue[$i]}"
                exit_code=$?
                if [ $exit_code -eq 0 ]; then
                    ((completed++))
                else
                    ((failed++))
                fi
                unset "job_queue[$i]"
            fi
        done
        
        # 배열 재정렬
        job_queue=("${job_queue[@]}")
        
        # 잠시 대기
        sleep 2
    done
    
    # 새 작업 시작
    build_service "$service" &
    job_queue+=($!)
    echo "🔄 작업 시작: $service (PID: $!)" | tee -a "$BUILD_LOG"
done

# 남은 모든 작업 완료 대기
echo "⏳ 모든 작업 완료 대기 중..." | tee -a "$BUILD_LOG"
for job in "${job_queue[@]}"; do
    wait "$job"
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        ((completed++))
    else
        ((failed++))
    fi
done

# 최종 결과 출력
echo "==================================" | tee -a "$BUILD_LOG"
echo "🎉 병렬 빌드 완료!" | tee -a "$BUILD_LOG"
echo "📅 완료 시간: $(date)" | tee -a "$BUILD_LOG"
echo "✅ 성공: $completed" | tee -a "$BUILD_LOG"
echo "❌ 실패: $failed" | tee -a "$BUILD_LOG"
echo "📊 총 서비스: ${#SERVICES[@]}" | tee -a "$BUILD_LOG"

if [ $failed -gt 0 ]; then
    echo "⚠️  실패한 서비스들:" | tee -a "$BUILD_LOG"
    if [ -f "$FAILED_FILE" ]; then
        cat "$FAILED_FILE" | tee -a "$BUILD_LOG"
    fi
fi

echo "📁 상세 로그: $LOG_DIR" | tee -a "$BUILD_LOG"
echo "✅ 완료된 서비스: $COMPLETED_FILE" | tee -a "$BUILD_LOG"
echo "❌ 실패한 서비스: $FAILED_FILE" | tee -a "$BUILD_LOG"
