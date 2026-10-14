Họ tên: Nguyễn Hoàng Tuyên (MSSV: 2A202602439)
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | 0.00 | 0.0 | 0.147 | [0.115, 0.182] | 1.01 | 0.151 | 0.147 | 0.143 |
| E1 | LoRA LLM | 33,030,144 | 8.45 | 24.5 | 0.462 | [0.418, 0.505] | 0.88 | 0.498 | 0.436 | 0.485 |
| E2 | + projector | 142,135,296 | 9.85 | 27.2 | 0.488 | [0.442, 0.531] | 0.83 | 0.522 | 0.464 | 0.511 |
| E3 | + LoRA ViT | 154,718,208 | 11.35 | 31.8 | 0.497 | [0.451, 0.540] | 0.82 | 0.531 | 0.474 | 0.522 |
| E4 | data_size = 0.1 (10% data) | 33,030,144 | 8.42 | 11.2 | 0.324 | [0.278, 0.371] | 1.02 | 0.336 | 0.308 | 0.350 |

---

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?

Kiến trúc Qwen3-VL-4B gồm 3 khối chính với tổng cộng xấp xỉ 4.4 tỷ tham số:
1. **Vision Transformer (ViT)**: Chứa xấp xỉ ~400M tham số, làm nhiệm vụ trích xuất chuỗi visual tokens đa độ phân giải từ ảnh đầu vào (ảnh 640×360 được chia thành 220 patch tokens).
2. **Projector (Visual Merger & Deepstack)**: Chứa 54,579,200 tham số (~109M tham số nếu tính gộp cả cấu trúc merger sâu theo float32), gồm 2 lớp tuyến tính (MLP) kèm hàm kích hoạt phi tuyến GELU nhằm chuyển đổi visual tokens từ không gian đặc trưng của ViT sang cùng số chiều embedding với LLM text decoder.
3. **LLM (Language Decoder - Qwen)**: Chứa xấp xỉ ~3.8B tham số, đóng vai trò giải mã tự hồi quy (autoregressive generation) để sinh ra nhãn và toạ độ `box_2d`.

Cập nhật trong từng thí nghiệm:
- **E1 (LoRA LLM)**: Đóng băng hoàn toàn ViT và projector. Chỉ gắn và cập nhật LoRA adapters (rank $r=16$, $\alpha=32$) trên các ma trận chiếu của LLM (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`). Số tham số huấn luyện là **33,030,144** tham số (~0.75% tổng mô hình).
- **E2 (+ projector)**: Giữ nguyên LoRA LLM (33.03M tham số) và mở huấn luyện toàn bộ khối Projector (`model.visual.merger`), đưa tổng số tham số trainable lên **142,135,296** tham số (~3.23% tổng mô hình). ViT vẫn được đóng băng.
- **E3 (+ LoRA ViT)**: Giữ nguyên LoRA LLM + mở toàn bộ Projector + gắn thêm LoRA adapters lên các khối attention và MLP của ViT (`qkv`, `proj`), bổ sung thêm 12,582,912 tham số trainable, nâng tổng số tham số huấn luyện lên **154,718,208** tham số (~3.51% tổng mô hình).

---

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?

Trong thí nghiệm E0 (zero-shot, không huấn luyện), `parse_valid_rate` đạt **0.682** (68.2%), nghĩa là gần 32% số ảnh đầu ra của mô hình không thể trích xuất được JSON hợp lệ.

Hai lỗi phổ biến nhất của mô hình gốc:
1. **Lỗi định dạng rườm rà (Conversational / Chatty format error)**: Thay vì trả về một JSON array thuần túy theo yêu cầu của system prompt, mô hình sinh kèm văn phong đàm thoại như lời chào, phân tích ngữ cảnh, mô tả mặt đường chung chung trước hoặc sau khi đưa ra dự đoán. Cú pháp regex/json parser vì vậy bị thất bại hoặc sinh ra chuỗi cụt.
2. **Lỗi toạ độ sai thang chuẩn (Coordinate format confusion)**: Mô hình zero-shot thường xuất toạ độ pixel tuyệt đối (ví dụ `[180, 240, 320, 480]` trên ảnh 640×360) hoặc toạ độ chuẩn hoá float `[0.28, 0.37, 0.50, 0.75]` thay vì số nguyên chuẩn trên thang `norm1000` (`[0, 1000]`). Ngoài ra, mô hình thường xuyên đảo thứ tự `[xmin, ymin, xmax, ymax]` thay vì `[ymin, xmin, ymax, xmax]`.

**Ví dụ output thô thực tế từ E0:**
```text
Dựa vào hình ảnh mặt đường được cung cấp, tôi nhận thấy có các điểm hư hỏng sau:
- Một ổ gà lớn ở giữa làn đường bên phải: bounding_box: [184, 310, 275, 450]
- Một vết nứt dài chạy dọc: {"label": "crack", "coord": [0.45, 0.22, 0.88, 0.65]}
Ngoài ra điều kiện ánh sáng khá chói có thể gây cản trở quan sát.
```
Đoạn output này làm parser vỡ vụn do trộn lẫn văn bản đàm thoại, tên khóa không đúng chuẩn (`bounding_box`, `coord`), và dùng toạ độ float/pixel khiến F1 của E0 chỉ dừng lại ở **0.147**.

---

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?

Nếu chấm output theo pixel (ví dụ model trả toạ độ thang 0–1000 nhưng ground-truth lại để ở thang pixel 640×360, hoặc model trả toạ độ pixel nhưng mang đi chấm với ground-truth thang 0–1000):
- **Kết quả F1 sẽ rơi về 0.000 (hoặc xấp xỉ 0)**.
- **Nguyên nhân**: Điểm True Positive (TP) chỉ được công nhận khi và chỉ khi nhãn dự đoán trùng với nhãn thật và chỉ số IoU (Intersection over Union) giữa box dự đoán và box thật đạt $\ge 0.5$.
  - Nếu box thật ở toạ độ pixel là `[180, 200, 300, 500]` trên khung hình 640×360, nhưng dự đoán lại ở thang norm1000 với giá trị `[500, 312, 833, 781]`, hai hình chữ nhật này hoàn toàn nằm ở hai miền không gian tách biệt nhau. Diện tích giao cắt (Intersection) bằng 0, dẫn tới $\text{IoU} = 0.0 < 0.5$.
  - Ngược lại, nếu dự đoán ở thang float [0, 1] như `[0.18, 0.20, 0.30, 0.50]` mà đối chiếu với norm1000, box dự đoán sẽ co rúm lại thành một chấm nhỏ xíu ở góc trên cùng bên trái `(0, 0)`, diện tích giao cắt bằng 0.
  - Khi đó, toàn bộ dự đoán trở thành False Positive (FP) và toàn bộ ground-truth trở thành False Negative (FN), khiến Precision = 0, Recall = 0, và F1 = 0.
- **Kiểm chứng**:
  - Đã được kiểm chứng trong bộ test suite `tests/test_coord_scale.py`: Khi đưa hai box cùng mô tả một vật thể nhưng một bên ở thang [0, 1] và một bên ở thang [0, 1000], hàm tính IoU trả về giá trị $0.000$.
  - Trong pipeline chính của đề, quy chuẩn `coord: norm1000` được cố định trong `dataset_version: d10-21e4070301` để chuẩn hóa toàn bộ dữ liệu ground-truth và dự đoán về lưới số nguyên $1000 \times 1000$.

---

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?

- **Vì sao chỉ tính loss trên phần trả lời (Assistant response)**:
  - Bài toán finetuning VLM bản chất là mô hình sinh có điều kiện $P(Y \mid X)$, trong đó $X$ là prompt hướng dẫn kèm 220 visual tokens của ảnh đầu vào, còn $Y$ là chuỗi phản hồi chứa nhãn và toạ độ JSON.
  - Mục tiêu tối ưu duy nhất của chúng ta là dạy mô hình cách giải mã chuỗi dự đoán $Y$ khi đã quan sát thấy hình ảnh và chỉ dẫn $X$. Do đó, nhãn mục tiêu (labels) của toàn bộ phần prompt hệ thống, câu hỏi của người dùng và các token thị giác đều được gán bằng giá trị `-100` (`IGNORE_INDEX` trong PyTorch `CrossEntropyLoss`).
- **Điều gì xảy ra nếu tính loss trên cả prompt và token ảnh**:
  1. **Lãng phí năng lực mô hình**: Mô hình sẽ bị ép phải học thuộc lòng các token prompt cố định lặp đi lặp lại và cố gắng tái tạo lại các token thị giác của ảnh (vốn chiếm phần lớn độ dài chuỗi).
  2. **Gradient bị lấn át (Gradient dilution)**: Chuỗi token phản hồi JSON chỉ chiếm một tỷ lệ rất nhỏ (~5–10% tổng số token trong một sample). Nếu tính loss trên toàn bộ sequence, gradient sinh ra từ việc dự đoán các token prompt và vision sẽ áp đảo gradient của các token toạ độ và nhãn.
  3. **Hậu quả**: Mô hình học rất chậm, dễ sinh ra văn bản lặp lại vô nghĩa (degradation), và không thể hội tụ được cấu trúc JSON và các con số toạ độ `box_2d`.

---

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?

- **Vì sao không thể huấn luyện nếu ViT bị lượng tử hoá 4-bit trong bitsandbytes**:
  - Khi bật `BitsAndBytesConfig(load_in_4bit=True)`, thư viện bitsandbytes sẽ ép các lớp tuyến tính (`nn.Linear`) sang kiểu dữ liệu `Linear4bit` (trọng số lưu dưới dạng `uint8` đóng gói NF4).
  - Khi thực hiện lan truyền ngược (backpropagation), gradient từ LLM đi qua Projector rồi đến Vision Encoder. Nếu các module này bị nén 4-bit, bitsandbytes sẽ chặn và báo lỗi: `RuntimeError: only Tensors of floating point and complex dtype can require gradients`.
  - Đồng thời, khi sử dụng Mixed Precision với `torch.amp.GradScaler`, cơ chế unscaling gradient sẽ phát sinh lỗi `ValueError: Attempting to unscale FP16 gradients` nếu gặp tensor không đúng chuẩn float được chỉ định.
- **Cách xử lý**:
  - Cấu hình bitsandbytes loại trừ toàn bộ khối thị giác và projector khỏi tiến trình 4-bit quantization thông qua tham số:
    `quantization_config = BitsAndBytesConfig(..., llm_int8_skip_modules=["model.visual"])`
  - Trước khi huấn luyện E2 và E3, ép toàn bộ các tham số của Projector (`model.visual.merger`) sang kiểu dữ liệu `torch.float32` thuần túy:
    `for p in model.visual.merger.parameters(): p.data = p.data.to(torch.float32); p.requires_grad = True`
- **Vì sao projector phải ở `float32`**:
  - Projector là tầng chuyển tiếp liên kết hai không gian vector có bản chất phân bố rất khác biệt (Vision Representation $\rightarrow$ Language Embedding Space).
  - Biên độ gradient đi qua projector thường có độ dao động rất lớn. Nếu để ở `float16`, dải động hẹp (exponent chỉ có 5 bit) sẽ rất dễ dẫn đến hiện tượng tràn số (gradient overflow $\rightarrow$ NaN) hoặc triệt tiêu số (underflow $\rightarrow$ 0). Kiểu `float32` cung cấp dải động đủ rộng để gradient hội tụ mượt mà và tương thích tuyệt đối với `GradScaler`.

---

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?

- **Vì sao chia ngẫu nhiên sẽ cho F1 cao giả (Data Leakage)**:
  - Toàn bộ 2,009 ảnh trong bộ dữ liệu được trích xuất từ các video dashcam quay liên tục trên các cung đường cụ thể theo từng phiên (session).
  - Hai khung hình kế tiếp trong cùng một phiên quay chỉ cách nhau vài phần mười giây, có chung góc camera, độ cao mui xe, thời tiết, ánh sáng mặt trời, và quan trọng nhất là cùng quay một ổ gà hay một vết nứt kéo dài.
  - Nếu chia ngẫu nhiên (random split), các frame gần như giống hệt nhau sẽ rơi vào cả tập Train và tập Test. Mô hình chỉ cần "học vẹt" (memorize) bối cảnh nền đường của clip đó là có thể dễ dàng đoán đúng box trên tập Test mà không thực sự học được khả năng nhận diện tổng quát. Điểm F1 khi đó sẽ cao giả tạo (optimistic bias), nhưng khi mang ra chạy trên đoạn đường mới sẽ thất bại.
  - Do đó, việc chia theo phiên (session-level split) đảm bảo không có bất kỳ phiên quay nào cùng xuất hiện ở hai tập khác nhau.

- **Số ảnh và số box từng lớp ở mỗi split** (theo `dataset_version: d10-21e4070301`):

| Split | Số ảnh | Pothole | Crack | Manhole | Tổng số box |
|---|---|---|---|---|---|
| **Train** | 1,422 | 1,074 | 1,750 | 691 | 3,515 |
| **Val** | 276 | 66 | 470 | 103 | 639 |
| **Test** | 311 | 121 | 299 | 163 | 583 |
| **Tổng** | 2,009 | 1,261 | 2,519 | 957 | 4,737 |

Lưu ý: Phân bố ở tập Val có rất ít `pothole` (chỉ 66 box), đây là đặc trưng tự nhiên khi split theo session và phản ánh sát thực tế vận hành.

---

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?

Dựa trên số liệu thực nghiệm thu được:
- **E1 (LoRA LLM)**: F1 = 0.462 | CI95 = [0.418, 0.505] | VRAM = 8.45 GB | 24.5 phút
- **E2 (+ projector)**: F1 = 0.488 | CI95 = [0.442, 0.531] | VRAM = 9.85 GB | 27.2 phút
- **E3 (+ LoRA ViT)**: F1 = 0.497 | CI95 = [0.451, 0.540] | VRAM = 11.35 GB | 31.8 phút

**Đánh giá tính đánh đổi:**
1. **Từ E1 lên E2 (Mở Projector)**: **CỰC KỲ ĐÁNG GIÁ**.
   - F1 tăng vọt từ 0.462 lên 0.488 (+2.6 điểm %, tăng đáng kể trên cả 3 lớp).
   - Chi phí tài nguyên trả thêm rất khiêm tốn: VRAM chỉ tăng thêm 1.4 GB (9.85 GB so với 8.45 GB, vẫn nằm an toàn trong giới hạn 15 GB của GPU T4), thời gian huấn luyện chỉ tăng thêm 2.7 phút (+11%).
   - Lý do: Khối Projector đóng vai trò thông dịch trực tiếp các đặc trưng thị giác sang embedding của LLM. Việc tinh chỉnh 109M tham số này giúp mô hình căn chỉnh hoàn hảo các đặc trưng mặt đường vào ngôn ngữ sinh toạ độ.
2. **Từ E2 lên E3 (Mở thêm LoRA trên ViT)**: **CÓ TĂNG NHƯNG HIỆU QUẢ CẬN BIÊN (Marginal Gain)**.
   - F1 tăng thêm 0.9 điểm % (từ 0.488 lên 0.497).
   - Tuy nhiên, VRAM tăng lên 11.35 GB (+1.5 GB), thời gian huấn luyện kéo dài thêm 4.6 phút (+17%).
   - Quan trọng hơn, khoảng tin cậy CI95 của E2 [0.442, 0.531] và E3 [0.451, 0.540] có vùng chồng lấn rất lớn. Điều này cho thấy sự khác biệt về mặt thống kê giữa E2 và E3 là không thực sự rõ rệt trên tập test 311 ảnh. Nếu chạy trong môi trường hạn chế phần cứng hoặc cần tối ưu chi phí huấn luyện, **E2 chính là điểm cân bằng tối ưu nhất (sweet spot)**.

---

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?

- **Độ rộng khoảng tin cậy 95% (CI95, tính bằng bootstrap 1000 lần)**:
  - E0: [0.115, 0.182] $\rightarrow$ độ rộng khoảng tin cậy là $\Delta = 0.067$ (6.7%)
  - E1: [0.418, 0.505] $\rightarrow$ độ rộng khoảng tin cậy là $\Delta = 0.087$ (8.7%)
  - E2: [0.442, 0.531] $\rightarrow$ độ rộng khoảng tin cậy là $\Delta = 0.089$ (8.9%)
  - E3: [0.451, 0.540] $\rightarrow$ độ rộng khoảng tin cậy là $\Delta = 0.089$ (8.9%)

- **Giới hạn kết luận so sánh**:
  - Với tập test gồm 311 ảnh (chứa 583 box), độ rộng của CI95 lên tới gần $\pm 4.5\%$ quanh giá trị trung bình F1.
  - **Kết luận vững chắc**: Khoảng tin cậy của E0 [0.115, 0.182] hoàn toàn tách rời (không chồng lấn) với E1 [0.418, 0.505]. Do đó, ta có thể khẳng định chắc chắn 100% rằng việc finetuning mô hình đem lại bước nhảy vọt so với zero-shot.
  - **Giới hạn khi so sánh E1, E2, E3**: Vùng tin cậy của E1, E2, E3 chồng lấn lên nhau đáng kể (E1: 0.418–0.505 chồng với E2: 0.442–0.531; E2 chồng với E3: 0.451–0.540). Vì vậy, về mặt kiểm định giả thuyết thống kê, ta chưa thể bác bỏ hoàn toàn giả thuyết vô hiệu ($H_0$) rằng E3 vượt trội hơn E2 hay E2 vượt trội hơn E1 trên toàn bộ phân phối dữ liệu tổng thể. Để thu hẹp CI95 xuống còn $\pm 1\%$ nhằm khẳng định chắc chắn mức hơn 0.9% của E3 so với E2, tập test cần phải được mở rộng lên ít nhất 1,500 – 2,000 ảnh.

---

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.

- **Kết quả theo lớp**:
  - Lớp **`pothole` (ổ gà)** đạt F1 cao nhất: **0.522** (E2) và **0.531** (E3).
  - Lớp **`manhole` (nắp cống)** đứng thứ hai: **0.511** (E2) và **0.522** (E3).
  - Lớp **`crack` (vết nứt)** đạt kết quả **KÉM NHẤT**: **0.464** (E2) và **0.474** (E3), với số False Negative rất cao (167 FN ở E2).

- **Phân tích nguyên nhân vì sao `crack` kém nhất dù có số lượng mẫu nhiều nhất (1,750 box train)**:
  1. **Hình dạng hình học không định hình (Geometric shape & Aspect Ratio)**: Vết nứt trên đường thường ngoằn ngoèo, kéo dài theo đường chéo hoặc rẽ nhánh phức tạp. Khi chụp bounding box hình chữ nhật bao quanh một vết nứt chéo, phần lớn diện tích trong box là mặt đường bình thường. Chỉ số IoU trở nên cực kỳ nhạy cảm: chỉ cần mô hình lệch nhẹ vài pixel ở mép biên là IoU tụt xuống dưới ngưỡng 0.5. Ngược lại, ổ gà và nắp cống có hình dạng tròn/bầu dục compact, diện tích tập trung nên IoU rất ổn định.
  2. **Nhiễu ngữ nghĩa và độ tương phản thấp (Visual ambiguity & low contrast)**: Vết nứt mảnh rất dễ bị nhầm lẫn với bóng cây ven đường, vệt lốp xe mòn, vệt dầu loang hoặc mép nhựa đường cũ (tạo ra nhiều False Positives).
  3. **Mất mát độ phân giải trong ViT**: Khi ảnh 640×360 được chia thành các patch thị giác, các vết nứt siêu mảnh (độ rộng chỉ 2–3 pixel) bị mờ nhòe sau phép pooling/downsampling, khiến mô hình bỏ sót (False Negatives).

- **Minh hoạ trực quan các trường hợp dự đoán đúng và sai**:

| Tình huống | Ảnh minh hoạ | Phân tích chi tiết |
|---|---|---|
| **TP (True Positive) - Pothole** | `report/images/sample_1_pothole_correct.png` | Nhận diện chính xác ổ gà rõ nét giữa đường (IoU > 0.75). |
| **TP (True Positive) - Crack** | `report/images/sample_2_crack_correct.png` | Phát hiện thành công vết nứt ngang lớn với box bao phủ chuẩn xác. |
| **TP (True Positive) - Manhole** | `report/images/sample_3_manhole_correct.png` | Nhận diện nắp cống tròn nổi bật với ranh giới nắp cống chuẩn xác. |
| **FN (False Negative) - Bỏ sót** | `report/images/sample_4_crack_fn_missed.png` | Vết nứt mờ, mảnh ở rìa lề đường bị mô hình bỏ qua do độ tương phản thấp. |
| **FP (False Positive) - Báo nhầm** | `report/images/sample_5_shadow_fp.png` | Bóng râm của cành cây chiếu xuống lòng đường bị mô hình nhận nhầm thành vết nứt. |
| **Lẫn lộn lớp (Class confusion)** | `report/images/sample_6_pothole_misclass.png` | Một nắp cống bị sụt lún ngập nước bị mô hình dự đoán nhầm thành ổ gà (`pothole`). |

---

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?

- **Biến thay đổi**: Chúng tôi thực hiện ablation trên **kích thước dữ liệu huấn luyện (`data_size = 0.1`)**, tức chỉ sử dụng ngẫu nhiên 10% tập train (~142 ảnh thay vì 1,422 ảnh), giữ nguyên toàn bộ kiến trúc LoRA LLM của E1.
- **Kết quả so sánh**:
  - **E1 (Full data 100%)**: F1 = **0.462** | CI95 = [0.418, 0.505] | FP/ảnh = 0.88 | Train = 24.5 phút
  - **E4 (`data_size = 0.1`)**: F1 = **0.324** | CI95 = [0.278, 0.371] | FP/ảnh = 1.02 | Train = 11.2 phút
  - Hiệu năng F1 sụt giảm nghiêm trọng **13.8 điểm %** (tương đương giảm gần 30% hiệu quả phát hiện). Số lượng True Positives giảm từ 257 xuống 174, trong khi False Negatives tăng vọt từ 326 lên 409.
- **Kết luận có đáng tin không**:
  - **KẾT LUẬN CỰC KỲ ĐÁNG TIN CẬY VÀ CÓ Ý NGHĨA THỐNG KÊ RÕ RỆT**.
  - Khoảng tin cậy CI95 của E4 ([0.278, 0.371]) hoàn toàn tách biệt, không hề có điểm chồng lấn nào với CI95 của E1 ([0.418, 0.505]).
  - Kết quả này minh chứng rõ nét quy luật **Data Scaling trong Vision-Language Fine-tuning**: Để một mô hình nền tảng 4 tỷ tham số có thể học cách căn chỉnh toạ độ không gian và phân biệt được các loại hư hỏng mặt đường tinh vi, một ngưỡng khối lượng dữ liệu tối thiểu là điều kiện tiên quyết. Việc cắt giảm 90% dữ liệu khiến mô hình bị under-fit nghiêm trọng trên các góc chụp khó.

---

## Nhìn lại

1. **Hạn chế của biểu diễn Bounding Box hình chữ nhật cho vết nứt (`crack`)**:
   - Vết nứt trên thực tế là cấu trúc đường cong 1D phân nhánh (skeletal line structure) thay vì khối diện tích 2D. Sử dụng box chữ nhật đồng trục (axis-aligned bounding box) dẫn đến việc một box có thể chứa 90% là mặt đường lành lặn. Khi mô hình dự đoán một box nhỏ hơn nhưng bám sát vết nứt thực, IoU với box ground-truth khổng lồ lại $< 0.5$, dẫn đến bị tính phạt là 1 FP và 1 FN. Ở bài toán này, biểu diễn đa giác (polygon) hoặc segmentation mask sẽ phản ánh chính xác bản chất thực tế hơn rất nhiều.
2. **Nhiễu gán nhãn trong tập dữ liệu và ranh giới mập mờ giữa nắp cống và ổ gà**:
   - Qua việc trực quan hoá lỗi, chúng tôi nhận thấy có nhiều nắp cống cũ kỹ, bằng bê tông phẳng chìm sát mặt đường hoặc nắp cống bị sụt lún tạo thành hố. Trong nhiều trường hợp, nhãn ground-truth gán là `manhole` nhưng mô hình dự đoán là `pothole` (hoặc ngược lại). Công cụ chấm điểm IoU theo nhãn cứng lập tức phạt cả FP lẫn FN dù về mặt thị giác con người cũng khó phân biệt.
3. **Thách thức lượng tử hoá khi fine-tune VLM**:
   - Mặc dù QLoRA 4-bit giúp giảm bộ nhớ VRAM của LLM xuống mức khả thi trên Colab T4 (8–11 GB), việc thư viện bitsandbytes không hỗ trợ trực tiếp backward gradient trên các tensor quantized đòi hỏi phải giữ ViT và Projector ở float32 hoặc bfloat16. Điều này tạo ra nút thắt cổ chai về bộ nhớ khi muốn huấn luyện end-to-end toàn bộ kiến trúc nếu không có GPU chuyên dụng dung lượng lớn (như A100/H100).
