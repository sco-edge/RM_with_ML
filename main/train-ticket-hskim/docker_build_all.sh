#!/bin/bash

# 모든 서비스의 Docker 이미지 생성 및 푸시 스크립트
# 사용법: ./docker_build_all.sh

set -e

BASE_DIR="/home/masternode/RM_with_ML/main/train-ticket-hskim"
LOG_DIR="$BASE_DIR/docker_logs"
BUILD_LOG="$LOG_DIR/docker_build_summary.log"

# 로그 디렉토리 생성
mkdir -p "$LOG_DIR"

# Docker 이미지 생성 및 푸시할 서비스 목록
SERVICES=(
    "ts-login-service"
    "ts-register-service"
    "ts-sso-service"
    "ts-verification-code-service"
    "ts-contacts-service"
    "ts-order-service"
    "ts-order-other-service"
    "ts-config-service"
    "ts-station-service"
    "ts-train-service"
    "ts-travel-service"
    "ts-travel2-service"
    "ts-preserve-service"
    "ts-preserve-other-service"
    "ts-basic-service"
    "ts-ticketinfo-service"
    "ts-price-service"
    "ts-notification-service"
    "ts-security-service"
    "ts-inside-payment-service"
    "ts-execute-service"
    "ts-payment-service"
    "ts-rebook-service"
    "ts-cancel-service"
    "ts-route-service"
    "ts-assurance-service"
    "ts-seat-service"
    "ts-travel-plan-service"
    "ts-route-plan-service"
    "ts-food-map-service"
    "ts-food-service"
    "ts-consign-price-service"
    "ts-consign-service"
    "ts-admin-order-service"
    "ts-admin-basic-info-service"
    "ts-admin-route-service"
    "ts-admin-travel-service"
    "ts-admin-user-service"
)

echo "🐳 Docker 이미지 생성 및 푸시 시작" | tee "$BUILD_LOG"
echo "📅 시작 시간: $(date)" | tee -a "$BUILD_LOG"
echo "📊 총 서비스 수: ${#SERVICES[@]}" | tee -a "$BUILD_LOG"
echo "==================================" | tee -a "$BUILD_LOG"

completed=0
failed=0

for service in "${SERVICES[@]}"; do
    service_dir="$BASE_DIR/$service"
    log_file="$LOG_DIR/${service}_docker.log"
    
    echo "🔨 Docker 이미지 생성 시작: $service" | tee -a "$BUILD_LOG"
    
    if [ ! -d "$service_dir" ]; then
        echo "❌ 디렉토리 없음: $service" | tee -a "$BUILD_LOG"
        ((failed++))
        continue
    fi
    
    cd "$service_dir"
    
    # Docker 이미지 생성
    version="v2"
    if docker build -t "ksh6283/$service:$version" . > "$log_file" 2>&1; then
        echo "✅ Docker 이미지 생성 성공: $service:$version" | tee -a "$BUILD_LOG"
        
        # Docker Hub 푸시
        if docker push "ksh6283/$service:$version" >> "$log_file" 2>&1; then
            echo "📤 Docker Hub 푸시 성공: $service:$version" | tee -a "$BUILD_LOG"
            ((completed++))
        else
            echo "❌ Docker Hub 푸시 실패: $service" | tee -a "$BUILD_LOG"
            ((failed++))
        fi
    else
        echo "❌ Docker 이미지 생성 실패: $service" | tee -a "$BUILD_LOG"
        ((failed++))
    fi
    
    echo "---" | tee -a "$BUILD_LOG"
done

# 최종 결과 출력
echo "==================================" | tee -a "$BUILD_LOG"
echo "🎉 Docker 빌드 완료!" | tee -a "$BUILD_LOG"
echo "📅 완료 시간: $(date)" | tee -a "$BUILD_LOG"
echo "✅ 성공: $completed" | tee -a "$BUILD_LOG"
echo "❌ 실패: $failed" | tee -a "$BUILD_LOG"
echo "📊 총 서비스: ${#SERVICES[@]}" | tee -a "$BUILD_LOG"
echo "📁 상세 로그: $LOG_DIR" | tee -a "$BUILD_LOG"
