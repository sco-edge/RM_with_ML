#!/bin/bash
prepare_dir() {
    mkdir -p data/NodeMetrics
}

# $1 = start_day, $2 = end_day
# $3 = start_hour, $4 = end_hour
fetch_data() {
    # NodeMetrics만!
    file_prefix="data/MSRTMCR/MSRTMCR"
    remote_prefix="MCRRTUpdate/MCRRTUpdate"
    ratio=720  # 12시간 단위

    start_min=$(($1 * 24 * 60 + $3 * 60))
    end_min=$(($2 * 24 * 60 + $4 * 60))

    start_idx=$(($start_min / $ratio))
    end_idx=$(($end_min / $ratio))
    if [ $(($end_min % $ratio)) -ne 0 ]; then
        end_idx=$(($end_idx))
    fi

    for idx in $(seq $start_idx $end_idx); do
        file_name="${file_prefix}_$idx.tar.gz"
        remote_path="${remote_prefix}_$idx.tar.gz"
        url="https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2022MicroservicesTraces/$remote_path"
        echo "[INFO] Downloading Node index $idx (12시간 단위)"
        wget -c --retry-connrefused --tries=0 --timeout=50 -O "$file_name" "$url"
    done
}

# 파라미터 파싱
for ARGUMENT in "$@"; do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE="${ARGUMENT:${#KEY}+1}"
    export "$KEY"="$VALUE"
done

# 날짜 파싱
start_day=$(expr $(echo $start_date | cut -f1 -dd) + 0)
start_hour=$(expr $(echo $start_date | cut -f2 -dd) + 0)
end_day=$(expr $(echo $end_date | cut -f1 -dd) + 0)
end_hour=$(expr $(echo $end_date | cut -f2 -dd) + 0)

# 실행
prepare_dir
fetch_data $start_day $end_day $start_hour $end_hour
