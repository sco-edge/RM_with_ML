#!/bin/bash

prepare_dir() {
    mkdir -p data/CallGraph
}

# $1 = start_day, $2 = end_day
# $3 = start_hour, $4 = end_hour
fetch_data() {
    file_prefix="data/CallGraph/CallGraph"
    remote_prefix="CallGraph/CallGraph"
    ratio=3  # 3분 단위로 저장됨

    start_min=$(($1 * 24 * 60 + $3 * 60))
    end_min=$(($2 * 24 * 60 + $4 * 60))

    start_idx=$(($start_min / $ratio))
    end_idx=$(($end_min / $ratio - 1))

    for idx in $(seq $start_idx $end_idx); do
        file_name="${file_prefix}_$idx.tar.gz"
        remote_path="${remote_prefix}_$idx.tar.gz"
        url="https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2022MicroservicesTraces/$remote_path"
        wget -c --retry-connrefused --tries=0 --timeout=50 -O $file_name $url
    done
}

# Parse arguments (e.g., start_date=0d0 end_date=1d0)
for ARGUMENT in "$@"; do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE="${ARGUMENT:${#KEY}+1}"
    export "$KEY"="$VALUE"
done

# Extract numeric day/hour
start_day=$(expr $(echo $start_date | cut -f1 -dd) + 0)
start_hour=$(expr $(echo $start_date | cut -f2 -dd) + 0)
end_day=$(expr $(echo $end_date | cut -f1 -dd) + 0)
end_hour=$(expr $(echo $end_date | cut -f2 -dd) + 0)

# Run
prepare_dir
fetch_data $start_day $end_day $start_hour $end_hour
