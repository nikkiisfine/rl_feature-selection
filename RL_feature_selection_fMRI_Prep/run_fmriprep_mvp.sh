#!/usr/bin/env bash
set -euo pipefail

# 1) BIDS 根目录（绝对路径）
BIDS_DIR="/Users/daxiaoniqiu/anaconda3/AI4H/pythonProject/myvenv/thesis/thesis/AD_MVP/BIDS"

# 2) 输出目录（绝对路径）
OUT_DIR="/Users/daxiaoniqiu/anaconda3/AI4H/pythonProject/myvenv/thesis/thesis/AD_MVP/mvp_fmriprep_out"

# 3) 工作目录（绝对路径，SSD）
WORK_DIR="/Users/daxiaoniqiu/anaconda3/AI4H/pythonProject/myvenv/thesis/thesis/AD_MVP/mvp_fmriprep_work"

# 4) FreeSurfer license
FS_LICENSE="/Users/daxiaoniqiu/anaconda3/AI4H/pythonProject/myvenv/thesis/thesis/license.txt"

# 5) 先跑少量 subjects 做 QC（把 sub- 后面的 ID 写进来）
SUBJECTS=($(ls ${BIDS_DIR} | grep '^sub-' | sed 's/sub-//'))
BATCH_SIZE=5

# 6) 线程/内存
NTHREADS="6"
OMP_NTHREADS="4"
MEM_MB="12000"

# 7) fMRIPrep 镜像版本
FMRIPREP_IMG="nipreps/fmriprep:24.1.0"
PLATFORM="--platform linux/amd64"


# ======================================
echo "Scanning subjects from BIDS..."
TOTAL=${#SUBJECTS[@]}

echo "Found ${TOTAL} subjects"


mkdir -p ${OUT_DIR}/logs
mkdir -p ${WORK_DIR}


# =========================
# ===== 批处理主循环 =====
# =========================

start=0

while [ $start -lt $TOTAL ]; do

    batch=("${SUBJECTS[@]:$start:$BATCH_SIZE}")

    echo ""
    echo "======================================="
    echo "Running batch: ${batch[*]}"
    echo "======================================="

    LOGFILE="${OUT_DIR}/logs/batch_${start}.log"

    docker run --rm -it \
        ${PLATFORM} \
        -v ${BIDS_DIR}:/data:ro \
        -v ${OUT_DIR}:/out \
        -v ${WORK_DIR}:/work \
        -v ${FS_LICENSE}:/opt/freesurfer/license.txt:ro \
        ${FMRIPREP_IMG} \
        /data /out participant \
        --participant-label ${batch[*]} \
        --work-dir /work \
        --fs-license-file /opt/freesurfer/license.txt \
        --fs-no-reconall \
        --ignore fieldmaps \
        --output-spaces MNI152NLin2009cAsym \
        --nthreads ${NTHREADS} \
        --omp-nthreads ${OMP_NTHREADS} \
        --mem_mb ${MEM_MB} \
        --resource-monitor \
        --stop-on-first-crash \
        2>&1 | tee ${LOGFILE}

    start=$((start + BATCH_SIZE))

done

echo ""
echo "✅ ALL BATCHES FINISHED!"
