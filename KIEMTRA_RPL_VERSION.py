"""Kiểm tra cột RPL_VERSION có mang tín hiệu không.

Log kiểm toán cho thấy version_mean / nbr_version_avg / diff_version đều có
nMI ~ 0.000 — tức KHÔNG mang thông tin. Script này xác định nguyên nhân:
cột RPL_VERSION trong RADAR có thực sự biến thiên ở file Version hay không.

Chạy:  python KIEMTRA_RPL_VERSION.py --raw-dir data/radar
"""
import argparse, glob, os
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--raw-dir", default="data/radar")
args = ap.parse_args()

pats = [os.path.join(args.raw_dir, "**", "*ersion*", "**", "*.csv")]
files = []
for p in pats:
    files += glob.glob(p, recursive=True)
if not files:
    print("Không tìm thấy file Version trong", args.raw_dir); raise SystemExit

print("Kiểm tra %d file Version:\n" % len(files))
for f in sorted(files)[:5]:
    df = pd.read_csv(f, low_memory=False)
    df.columns = [c.strip().rstrip(",") for c in df.columns]
    if "RPL_VERSION" not in df.columns:
        print("  %-40s KHÔNG có cột RPL_VERSION" % os.path.basename(f)); continue
    v = pd.to_numeric(df["RPL_VERSION"], errors="coerce").dropna()
    uniq = sorted(v.unique().tolist())
    print("  %-40s giá trị=%s std=%.3f %s" % (
        os.path.basename(f), uniq[:6], v.std() if len(v) > 1 else 0,
        "<== HẰNG SỐ, vô dụng" if (len(uniq) <= 1) else "<== CÓ biến thiên"))

print("\nKẾT LUẬN:")
print("  • Nếu tất cả HẰNG SỐ -> 3 đặc trưng version là rác, nên BỎ (d_n 21->18)")
print("  • Nếu CÓ biến thiên -> giữ lại; chúng sẽ có ích sau khi sửa nhãn version")
