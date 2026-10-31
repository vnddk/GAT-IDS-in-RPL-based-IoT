#!/usr/bin/env bash
# Chạy t-SNE + ma trận nhầm lẫn cho MỌI mô hình trên CẢ BỐN bộ dữ liệu.
#
#   · EdgeGAT và NE-GAT : t-SNE TỪNG TẦNG (h1..hK + biểu diễn trước phân loại)
#   · Mô hình còn lại    : t-SNE TẦNG CUỐI
#   · Ma trận nhầm lẫn   : cho tất cả, không trừ mô hình nào
#
# Sửa bốn biến dưới cho khớp máy của bạn rồi chạy:
#     bash scripts/run_all_viz_4ds.sh
set -u

DATA_RADAR=${DATA_RADAR:-data/radar_chrono}
DATA_UOS=${DATA_UOS:-data/uos_chrono}
DATA_BEH=${DATA_BEH:-data/rplbeh_chrono}
DATA_GUARD=${DATA_GUARD:-data/rplguard_chrono}

CKPT_RADAR=${CKPT_RADAR:-checkpoints_radar}
CKPT_UOS=${CKPT_UOS:-checkpoints_uos}
CKPT_BEH=${CKPT_BEH:-checkpoints_rplbeh}
CKPT_GUARD=${CKPT_GUARD:-checkpoints_rplguard}

# Thư mục checkpoint GAT-Transformer SAU TINH CHỈNH (để trống nếu chưa chạy)
GT_RADAR=${GT_RADAR:-ck_gtids_radar}
GT_UOS=${GT_UOS:-ck_gtids_uos}
GT_BEH=${GT_BEH:-}
GT_GUARD=${GT_GUARD:-}

OUT=${OUT:-viz}
NS=${NS:-3000}

run_one () {
  local tag="$1" data="$2" ckpt="$3" gt="$4"
  echo ""
  echo "=============================================================="
  echo ">>> BỘ: $tag"
  echo "=============================================================="
  if [ ! -f "$data/meta.json" ]; then
    echo "    BỎ QUA: không thấy $data/meta.json (chưa chuyển đổi dữ liệu?)"
    return
  fi
  local gtarg=""
  [ -n "$gt" ] && [ -d "$gt" ] && gtarg="--gt-dir $gt"
  python scripts/run_all_viz.py \
      --data-dir "$data" --ckpt-dir "$ckpt" \
      --out-dir "$OUT/$tag" --tag "$tag" \
      --n-samples "$NS" $gtarg
}

run_one radar    "$DATA_RADAR" "$CKPT_RADAR" "$GT_RADAR"
run_one uos      "$DATA_UOS"   "$CKPT_UOS"   "$GT_UOS"
run_one rplbeh   "$DATA_BEH"   "$CKPT_BEH"   "$GT_BEH"
run_one rplguard "$DATA_GUARD" "$CKPT_GUARD" "$GT_GUARD"

echo ""
echo "=============================================================="
echo "XONG. Kết quả nằm trong: $OUT/{radar,uos,rplbeh,rplguard}/"
echo "  cm_<bộ>_<mô hình>.png    ma trận nhầm lẫn"
echo "  cm_<bộ>_<mô hình>.csv    ma trận nhầm lẫn dạng số"
echo "  tsne_<bộ>_<mô hình>.png  t-SNE"
echo "  viz_summary_<bộ>.json    tóm tắt accuracy"
echo "=============================================================="
