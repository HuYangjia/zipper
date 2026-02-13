#!/bin/bash
set -euo pipefail

CONFIG="data.json"
PYTHON="${PYTHON:-python}"
FILE_NAME="${FILE_NAME:-quant_time_test.py}"

ITERS="${ITERS:-100}"
WARMUP="${WARMUP:-20}"
S="${S:-0}"        # S>0: 固定 N=S；S=0: 用 data.json 里的 M 作为 N
SEED="${SEED:-0}"

RESULT_LOG="benchlog_quant.txt"
FINAL_OUT="result_quant.tsv"

: > "$RESULT_LOG"
: > "$FINAL_OUT"

# 输出列：适配 quantize_bench.py 的打印内容
echo -e "item\tmatrix\tN\tK\tavg_ms\tp50_ms\tp90_ms\tp99_ms\tgbps_in" >> "$FINAL_OUT"

# 从 data.json 读出：ITEM, MATRIX, ROWS(M), COLS(K)
jq -r '
  to_entries[] |
  .key as $item |
  .value | to_entries[] |
  "\($item)\t\(.key)\t\(.value[0])\t\(.value[1])"
' "$CONFIG" | while IFS=$'\t' read -r ITEM MATRIX M K; do

  # quantize 的输入是 [N, K]
  if (( S > 0 )); then
    N="$S"
  else
    N="$M"
  fi

  echo "==== item=$ITEM matrix=$MATRIX N=$N K=$K iters=$ITERS warmup=$WARMUP ====" | tee -a "$RESULT_LOG"

  # int4 pack 通常要求 K 为偶数（你的 wrapper 也会检查）
  if (( K % 2 != 0 )); then
    echo "[SKIP] K=$K is odd (int4 pack requires even K)" | tee -a "$RESULT_LOG"
    continue
  fi

  "$PYTHON" "$FILE_NAME" \
    --N "$N" --K "$K" \
    --iters "$ITERS" --warmup "$WARMUP" --seed "$SEED" \
    2>&1 | tee -a "$RESULT_LOG" | while IFS= read -r line; do

      # 解析第一行：
      # [quantize_w4a4_act] N=... K=... dtype=fp16
      if [[ "$line" =~ ^\[quantize_w4a4_act\]\ N=([0-9]+)\ K=([0-9]+) ]]; then
        cur_N="${BASH_REMATCH[1]}"
        cur_K="${BASH_REMATCH[2]}"
        continue
      fi

      # 解析第二行：
      # time ms: avg=...  p50=...  p90=...  p99=...  (iters=..., warmup=...)
      if [[ "$line" =~ ^[[:space:]]*time\ ms:\ avg=([0-9.]+)[[:space:]]+p50=([0-9.]+)[[:space:]]+p90=([0-9.]+)[[:space:]]+p99=([0-9.]+) ]]; then
        avg="${BASH_REMATCH[1]}"
        p50="${BASH_REMATCH[2]}"
        p90="${BASH_REMATCH[3]}"
        p99="${BASH_REMATCH[4]}"
        continue
      fi

      # 解析第三行：
      # approx input GB/s: ...
      if [[ "$line" =~ ^[[:space:]]*approx\ input\ GB/s:\ ([0-9.]+) ]]; then
        gbps="${BASH_REMATCH[1]}"

        # 拿到第三行就可以写 TSV（这一行写在 pipe 的 while 里，变量都可见）
        echo -e "${ITEM}\t${MATRIX}\t${cur_N}\t${cur_K}\t${avg}\t${p50}\t${p90}\t${p99}\t${gbps}" >> "$FINAL_OUT"
        continue
      fi

    done
done

echo "Done."
echo "  Full log   : $RESULT_LOG"
echo "  Summary TSV: $FINAL_OUT"
