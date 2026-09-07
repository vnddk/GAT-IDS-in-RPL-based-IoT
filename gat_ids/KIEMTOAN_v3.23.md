# Kiểm toán framework v3.23 — gỡ RPL-IDS-Beh + rà lỗi

## A. Đã gỡ bỏ RPL-IDS-Beh

| Việc | Chi tiết |
|---|---|
| Xoá tệp | `src/data/rplbeh_builder.py`, `configs/rplbeh.yaml`, `RPLBEH_TICHHOP.md` |
| Hoàn nguyên `src/data/loader.py` | bỏ import, mục `DATASET_CLASSES["rplbeh"]`, nhánh `if dataset == "rplbeh"`, tham số `use_rates` |
| Hoàn nguyên `scripts/convert_data.py` | bỏ cờ `--random-split`, `--raw-counters`, nhánh `split_rplbeh_by_sim`, nhánh tên đặc trưng trong kiểm toán rò rỉ |

Kiểm chứng: `grep -rn "rplbeh\|RPLBEH"` trên toàn cây → **0 kết quả** trong mã
nguồn. Còn đúng một chỗ nhắc tên bài báo: `src/data/radar_builder.py` dòng 238,
trích **triết lý** "tấn công RPL = node nói dối về vai trò trong DODAG" của
Nassrullah & Alisa 2025 làm căn cứ thiết kế đặc trưng RADAR. Đây là trích dẫn
học thuật, không phải phụ thuộc mã — **giữ nguyên**.

**Hai sửa lỗi phát hiện trong đợt RPL-IDS-Beh được GIỮ LẠI**, vì chúng là lỗi
của framework chứ không thuộc dataset đó:

1. `smote_augment_graphs` trong `loader.py` dựng `Data` mới và làm mất mọi
   thuộc tính phụ → PyG `collate` ném `KeyError` khi batch trộn graph đã
   augment với graph gốc. **Ảnh hưởng nhánh `iotrpl`** (`src_file` biến mất).
2. `feature_audit.py` in cứng `"H(16 lớp)"` và câu kết `"theo cả hai tiêu chí"`
   trong khi quyết định gắn cờ chỉ dựa vào một tiêu chí.

---

## B. Lỗi tìm thêm trong đợt kiểm toán này

### B1. `format_report` GIẤU lớp bị bỏ sót hoàn toàn — NGHIÊM TRỌNG

`src/utils/metrics.py`. Bản cũ:

```python
if p == 0 and r == 0 and f == 0:
    continue          # bỏ qua lớp không có mẫu
```

Chú thích nói "lớp không có mẫu", nhưng một lớp **có mẫu** mà model bỏ sót
**hoàn toàn** cũng có P=R=F1=0 — và bị xoá khỏi bảng. Đó đúng là trường hợp
cần nhìn thấy nhất (lớp thiểu số không được phát hiện lần nào) lại là trường
hợp bị ẩn.

Minh chứng trên dữ liệu synthetic: bảng cũ chỉ in **2 trong 6 lớp**, ẩn mất
Sinkhole (8 mẫu), Blackhole (8), Flooding (59), Rank (8) — tất cả đều có mẫu
thật trong tập test.

`macro_f1` trong `compute_metrics` vẫn tính **đúng** (có cộng các số 0 đó), nên
lỗi nằm ở phần hiển thị. Nhưng hậu quả thật: bảng in ra mâu thuẫn với macro-F1
và dẫn tới kết luận sai về lớp yếu — đúng chủ đề trung tâm của luận văn.

**Sửa:** quyết định theo số mẫu thật lấy từ confusion matrix. Lớp có mẫu thì
luôn hiện, thêm cột `Support`, đánh dấu `<-- BỎ SÓT HOÀN TOÀN`, và một dòng
tổng kết liệt kê các lớp không được phát hiện lần nào. Chỉ lớp thực sự vắng
mặt trong `y_true` mới bị ẩn.

### B2. `predict.py` bỏ qua Yeo-Johnson khi chuẩn hoá — NGHIÊM TRỌNG, ÂM THẦM

`scripts/predict.py`, hàm `_load_scaler`. Bản cũ luôn dựng `GraphScaler`
(z-score) và chỉ nạp `mean`/`std`, **bỏ qua λ Yeo-Johnson**. Với cấu hình mặc
định (`use_power: true`), mẫu mới sẽ được chuẩn hoá **khác** với lúc train →
dự đoán sai mà **không báo lỗi**.

### B3. `predict.py` không nạp nổi scaler

Cùng hàm: `torch.load(..., weights_only=True)`. `scaler.pt` chứa `ndarray` của
numpy chứ không chỉ tensor → `UnpicklingError`. Nghĩa là với cấu hình mặc định,
`predict.py` **chưa từng chạy được lần nào**. Lỗi B3 che lỗi B2: sửa xong B3 mà
không sửa B2 thì script chạy trơn tru nhưng cho kết quả sai.

**Sửa (B2+B3):** `weights_only=False`, và chọn lớp scaler theo cờ `use_power`
đã lưu trong `scaler.pt`, khôi phục cả `x_lam`/`e_lam`.

### B4. Pipeline 3 bước không chạy được với dữ liệu synthetic

`main.py` có đường synthetic (qua `get_datasets`), nhưng `convert_data.py` gọi
thẳng `load_from_csv` — hàm này trả rỗng cho `dataset=synthetic` → `Không tạo
được graph` rồi thoát. Hệ quả: **không có cách nào chạy thử convert → train_all
→ test_all khi chưa có CSV thật**, dù framework có sẵn bộ sinh synthetic.

**Sửa:** thêm nhánh fallback dùng chung `generate_dataset`, để hai đường vào
hành xử giống nhau.

### B5. Bảng so sánh in cứng "6 MODELS"

`scripts/test_all.py` dòng 183 in `"BẢNG SO SÁNH 6 MODELS"` trong khi bảng thực
tế có 7 dòng (RF, XGBoost, LightGBM, MLP, GCN, E-GraphSAGE, EdgeGAT). **Sửa:**
đếm động số model có kết quả.

### B6. Vòng federated cuối thường không được log

`src/federated/fed.py`: điều kiện `rnd % 5 == 0 or rnd == 1`. Khi `rounds`
không chia hết cho 5 (ví dụ 60 thì được, 3 thì không), vòng cuối không in ra
nên không thấy trạng thái hội tụ. **Sửa:** thêm `or rnd == fcfg["rounds"]`.

### B7. Docstring lệch code trong kiểm toán rò rỉ

`feature_audit.py`: docstring ghi ngưỡng `F1 nhị phân > 0.95`, code dùng `0.90`.
**Sửa:** docstring theo code.

---

## C. Đã kiểm tra và KHÔNG có lỗi

| Hạng mục | Kết quả |
|---|---|
| Biên dịch toàn bộ (`compileall`) | sạch |
| Import 10 module lõi | sạch |
| `--help` của 13 script | sạch |
| Chống rò rỉ chuẩn hoá | scaler `fit` **chỉ trên train**, transform cả ba tập — đúng |
| Resample chỉ trên train | đúng; `orig_counts` tính **trước** resample rồi truyền vào loss (SMOTE và CB-Focal không còn triệt tiêu nhau) |
| `macro_f1` | tính trên các lớp **có mẫu**, có cộng F1=0 của lớp bị bỏ sót — đúng |
| `split_graphs` chuyển graph train→test | thứ tự `pop` giảm dần, không lệch chỉ số — đúng |
| Tái lập theo seed | chạy `train.py` hai lần cùng seed → macro-F1 **giống hệt** (0,0843) |
| Đường federated | chạy thông (fedprox + Dirichlet 3 client) |
| Ablation `--no-edge` | chạy thông |
| `confusion_matrix.py`, `train.py`, `train_all.py --models all`, `test_all.py` | chạy thông |

**Lưu ý:** mọi số trong bảng trên đến từ dữ liệu **synthetic** với 4–6 epoch —
chỉ dùng để chứng minh mã chạy đúng, **tuyệt đối không** có ý nghĩa về chất
lượng mô hình. Đừng trích vào luận văn.

---

## D. Điều đáng chú ý, KHÔNG sửa (cần anh quyết)

1. **EMA khởi tạo từ trọng số ngẫu nhiên.** `EMA.shadow` sao chép model lúc
   khởi tạo, decay 0.995 → khoảng 200 bước mới "quên" hết trọng số ngẫu nhiên
   ban đầu. Với 120 epoch thì vô hại, nhưng với run ngắn (< 30 epoch) val bị
   kéo xuống mạnh. Đây là lý do các run 4–6 epoch ở trên có val rất thấp.
   Sửa được bằng hiệu chỉnh chệch (bias correction) như Adam, nhưng sẽ **đổi
   số liệu của mọi thí nghiệm đã chạy** — nên tôi để nguyên và báo anh.

2. **Federated dùng `Adam`, centralized dùng `AdamW`**; federated cũng không có
   EMA, không có scheduler. So sánh centralized với federated vì thế không phải
   so sánh cùng điều kiện. Nếu luận văn có bảng đối chiếu hai chế độ, cần nói
   rõ điểm này hoặc đồng bộ bộ tối ưu.

3. **`GraphScaler`/`GraphPowerScaler` sửa `g.x` tại chỗ** (`transform` gán trực
   tiếp vào graph). Gọi `transform` hai lần trên cùng danh sách sẽ chuẩn hoá
   chồng. Hiện không chỗ nào gọi hai lần nên chưa thành lỗi.
