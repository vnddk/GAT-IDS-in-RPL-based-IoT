"""TỔNG HỢP KẾT QUẢ NHIỀU HẠT GIỐNG — báo cáo trung bình ± độ lệch chuẩn.

Chạy:
    python scripts/aggregate_seeds.py --dirs ck_A_s42,ck_A_s43,ck_A_s44 --name A
    python scripts/aggregate_seeds.py --dirs ck_D_s42,ck_D_s43,ck_D_s44 --name D --compare A

VÌ SAO BẮT BUỘC: sáu cấu hình đã thử nghiệm cách nhau chỉ 0,0086 macro-F1,
trong khi biến thiên do hạt giống của mô hình GNN thường vào khoảng
0,005–0,015. Với một hạt giống duy nhất, KHÔNG THỂ kết luận cấu hình nào tốt
hơn. Đây không phải chi tiết kỹ thuật nhỏ — nó quyết định toàn bộ Chương 4 và
Chương 5 của luận văn có bảo vệ được hay không.

Script đọc tệp test_metrics.json mà test_all.py sinh ra trong từng thư mục
điểm kiểm tra, rồi tính trung bình, độ lệch chuẩn và khoảng tin cậy 95%.
"""
import os, sys, json, argparse, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.utils.common import get_logger

log = get_logger()

# hệ số t hai phía, mức 95%, cho n nhỏ (bậc tự do n−1)
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
       8: 2.365, 9: 2.306, 10: 2.262}


def read_metrics(d, model="EdgeGAT"):
    """Đọc kết quả từ một thư mục điểm kiểm tra."""
    for cand in [os.path.join(d, model, "test_metrics.json"),
                 os.path.join(d, "test_metrics.json"),
                 os.path.join(d, "test_comparison.json")]:
        if not os.path.exists(cand):
            continue
        with open(cand, encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):                       # dạng list
            for r in obj:
                if r.get("name") == model:
                    return r
            return None
        if model in obj and isinstance(obj[model], dict):   # dict theo tên model
            return obj[model]
        if "macro_f1" in obj:                                # dạng phẳng
            return obj
        return None
    return None


def summarize(vals, label, unit=""):
    v = np.asarray(vals, dtype=float)
    n = len(v)
    mean = v.mean()
    sd = v.std(ddof=1) if n > 1 else 0.0
    if n > 1:
        t = T95.get(n, 2.0)
        half = t * sd / math.sqrt(n)
    else:
        half = 0.0
    log.info("   %-14s %.4f ± %.4f%s   (KTC 95%%: %.4f – %.4f, n=%d)",
             label, mean, sd, unit, mean - half, mean + half, n)
    return mean, sd, half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True,
                    help="Danh sách thư mục điểm kiểm tra, cách nhau bởi dấu phẩy")
    ap.add_argument("--model", default="EdgeGAT")
    ap.add_argument("--name", default="cấu hình")
    ap.add_argument("--compare-dirs", default=None,
                    help="Danh sách thư mục của cấu hình thứ hai để so sánh")
    ap.add_argument("--compare-name", default="cấu hình B")
    ap.add_argument("--out", default="seed_summary.json")
    args = ap.parse_args()

    def collect(dirs):
        f1, acc, per = [], [], {}
        for d in [x.strip() for x in dirs.split(",") if x.strip()]:
            r = read_metrics(d, args.model)
            if r is None:
                log.warning("Bỏ qua %s — không tìm thấy kết quả kiểm thử", d)
                continue
            f1.append(r["macro_f1"]); acc.append(r["accuracy"])
            for k, v in (r.get("per_class_f1") or {}).items():
                per.setdefault(k, []).append(v)
        return f1, acc, per

    f1a, acca, pera = collect(args.dirs)
    if not f1a:
        log.error("Không đọc được kết quả nào. Chạy test_all.py cho từng thư mục trước.")
        return

    log.info("═══ %s  (n = %d hạt giống) ═══", args.name, len(f1a))
    log.info("   macro-F1 từng lần: %s", ", ".join("%.4f" % v for v in f1a))
    m_f1, s_f1, h_f1 = summarize(f1a, "macro-F1")
    m_ac, s_ac, h_ac = summarize(acca, "accuracy")

    if pera:
        log.info("\n── F1 THEO LỚP (trung bình ± độ lệch chuẩn) ──")
        log.info("   %-22s %18s %10s", "lớp", "F1", "biến thiên")
        for k in sorted(pera, key=lambda x: np.mean(pera[x])):
            v = np.asarray(pera[k], float)
            sd = v.std(ddof=1) if len(v) > 1 else 0.0
            flag = "  ← rất biến động" if sd > 0.03 else ""
            log.info("   %-22s %10.4f ± %.4f %8s%s", k, v.mean(), sd, "", flag)

    out = {args.name: {"macro_f1_mean": m_f1, "macro_f1_std": s_f1,
                       "accuracy_mean": m_ac, "accuracy_std": s_ac,
                       "n_seeds": len(f1a), "macro_f1_runs": f1a}}

    # ── So sánh hai cấu hình ──
    if args.compare_dirs:
        f1b, accb, _ = collect(args.compare_dirs)
        if f1b:
            log.info("\n═══ %s  (n = %d hạt giống) ═══", args.compare_name, len(f1b))
            log.info("   macro-F1 từng lần: %s", ", ".join("%.4f" % v for v in f1b))
            m2, s2, h2 = summarize(f1b, "macro-F1")
            out[args.compare_name] = {"macro_f1_mean": m2, "macro_f1_std": s2,
                                      "n_seeds": len(f1b), "macro_f1_runs": f1b}

            log.info("\n── KẾT LUẬN THỐNG KÊ ──")
            diff = m_f1 - m2
            lo1, hi1 = m_f1 - h_f1, m_f1 + h_f1
            lo2, hi2 = m2 - h2, m2 + h2
            overlap = not (hi1 < lo2 or hi2 < lo1)
            log.info("   Chênh lệch trung bình: %+.4f macro-F1", diff)
            log.info("   KTC 95%% %s: %.4f – %.4f", args.name, lo1, hi1)
            log.info("   KTC 95%% %s: %.4f – %.4f", args.compare_name, lo2, hi2)
            if overlap:
                log.info("   → HAI KHOẢNG CHỒNG NHAU: chưa đủ bằng chứng kết luận")
                log.info("     cấu hình nào tốt hơn. Nên chọn theo tiêu chí KHÁC:")
                log.info("     số tham số, độ tách cụm (silhouette), hoặc độ đơn giản.")
            else:
                better = args.name if diff > 0 else args.compare_name
                log.info("   → KHÔNG chồng nhau: %s tốt hơn có ý nghĩa thống kê.", better)
            out["comparison"] = {"diff": diff, "ci_overlap": bool(overlap)}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log.info("\nĐã lưu -> %s", args.out)
    log.info("\nGHI CHÚ CHO LUẬN VĂN: báo cáo dạng 'macro-F1 = %.4f ± %.4f (n=%d)'",
             m_f1, s_f1, len(f1a))


if __name__ == "__main__":
    main()
