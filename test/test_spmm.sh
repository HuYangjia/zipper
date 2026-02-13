#!/bin/bash
#!/bin/bash
set -euo pipefail

CONFIG="data.json"
PYTHON="${PYTHON:-python}"
FILE_NAME="${FILE_NAME:-spmm_time_test.py}"  


DEVICE="${DEVICE:-cuda}"
ITERS="${ITERS:-100}"
WARMUP="${WARMUP:-20}"
S="${S:-1}"              # 这里沿用你原来的 S，映射为 spmm 的 N
SEED="${SEED:-0}"

# 你要测的 density 列表（可改成多个）
# DENSITY_LIST=(0.01 0.05 0.1 0.2)
DENSITY_LIST=(0.01)

RESULT_LOG="benchlog_spmm.txt"
FINAL_OUT="result_spmm.tsv"

: > "$RESULT_LOG"
: > "$FINAL_OUT"

# 输出列：适配 spmm_bench.py 的打印内容
echo -e "item\tmatrix\tM\tK\tN\tdensity\tnnz\tnnz_per_row\tavg_ms\tp50_ms\tp90_ms\tp99_ms\ttflops" >> "$FINAL_OUT"

# 从 data.json 读出：ITEM, MATRIX, ROWS(M), COLS(K)
jq -r '
  to_entries[] |
  .key as $item |
  .value | to_entries[] |
  "\($item)\t\(.key)\t\(.value[0])\t\(.value[1])"
' "$CONFIG" | while IFS=$'\t' read -r ITEM MATRIX M K; do

  N="$S"   # spmm_bench.py 需要 N；这里直接用你原来的 S

  echo "==== item=$ITEM matrix=$MATRIX M=$M K=$K N=$N device=$DEVICE iters=$ITERS warmup=$WARMUP ====" | tee -a "$RESULT_LOG"

  for density in "${DENSITY_LIST[@]}"; do
    echo "--- density=$density ---" | tee -a "$RESULT_LOG"

    # 运行一次 bench，并解析输出三行
    cur_M="" cur_K="" cur_N="" cur_density="" cur_nnz="" cur_nnz_row=""
    avg="" p50="" p90="" p99="" tflops=""

    "$PYTHON" "$FILE_NAME" \
      --M "$M" --K "$K" --N "$N" \
      --density "$density" \
      --iters "$ITERS" --warmup "$WARMUP" --seed "$SEED" \
      2>&1 | tee -a "$RESULT_LOG" | while IFS= read -r line; do

        # 1) 解析第一行：
        # [spmm_csr_fp16_wxt] M=... K=... N=... density=... nnz=... nnz/row=...
        if [[ "$line" =~ ^\[spmm_csr_fp16_wxt\]\ M=([0-9]+)\ K=([0-9]+)\ N=([0-9]+)\ density=([0-9.eE+-]+)\ nnz=([0-9]+)\ nnz/row=([0-9]+) ]]; then
          cur_M="${BASH_REMATCH[1]}"
          cur_K="${BASH_REMATCH[2]}"
          cur_N="${BASH_REMATCH[3]}"
          cur_density="${BASH_REMATCH[4]}"
          cur_nnz="${BASH_REMATCH[5]}"
          cur_nnz_row="${BASH_REMATCH[6]}"
          continue
        fi

        # 2) 解析第二行：
        # time ms: avg=...  p50=...  p90=...  p99=...
        if [[ "$line" =~ ^[[:space:]]*time\ ms:\ avg=([0-9.]+)[[:space:]]+p50=([0-9.]+)[[:space:]]+p90=([0-9.]+)[[:space:]]+p99=([0-9.]+) ]]; then
          avg="${BASH_REMATCH[1]}"
          p50="${BASH_REMATCH[2]}"
          p90="${BASH_REMATCH[3]}"
          p99="${BASH_REMATCH[4]}"
          continue
        fi

        # 3) 解析第三行：
        # approx TFLOPS (2*nnz*N): ...
        if [[ "$line" =~ ^[[:space:]]*approx\ TFLOPS.*:\ ([0-9.]+) ]]; then
          tflops="${BASH_REMATCH[1]}"

          # 一旦拿到 tflops，说明本轮关键信息齐了，写 TSV
          echo -e "${ITEM}\t${MATRIX}\t${cur_M}\t${cur_K}\t${cur_N}\t${cur_density}\t${cur_nnz}\t${cur_nnz_row}\t${avg}\t${p50}\t${p90}\t${p99}\t${tflops}" >> "$FINAL_OUT"
          continue
        fi
      done
  done
done

echo "Done."
echo "  Full log   : $RESULT_LOG"
echo "  Summary TSV: $FINAL_OUT"
