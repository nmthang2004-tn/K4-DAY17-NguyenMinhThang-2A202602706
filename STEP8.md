# Bước 8 - Phân tích kết quả

Kết quả dưới đây được lấy khi chạy `python src/benchmark.py` từ trạng thái không có thư mục `state/`.

| Bộ benchmark | Agent | Agent tokens | Prompt tokens | Recall | Quality | Memory growth | Compactions |
|---|---|---:|---:|---:|---:|---:|---:|
| Standard | Baseline | 1.945 | 16.498 | 0,00 | 0,46 | 0 bytes | 0 |
| Standard | Advanced | 1.267 | 18.501 | 0,43 | 0,69 | 7 bytes | 0 |
| Long-Context Stress | Baseline | 255 | 22.253 | 0,00 | 0,37 | 0 bytes | 0 |
| Long-Context Stress | Advanced | 143 | 15.577 | 0,33 | 0,62 | 146 bytes | 1 |

## 1. Vì sao Advanced có recall tốt hơn Baseline?

Baseline chỉ giữ lịch sử hội thoại trong RAM theo `thread_id`. Các câu hỏi recall được đặt ở một thread mới, vì vậy Baseline không còn ngữ cảnh của thread trước và đạt recall 0,00 ở cả hai bộ benchmark.

Advanced có thêm persistent memory. Nó trích xuất các thông tin ổn định của người dùng, ghi chúng vào `state/profiles/<user_id>/User.md`, rồi nạp hồ sơ này khi trả lời ở thread mới. Nhờ bộ nhớ dùng chung giữa các phiên, Advanced đạt recall 0,43 ở Standard và 0,33 ở Long-Context Stress. Recall chưa đạt 1,00 vì cơ chế trích xuất theo luật chỉ lưu và truy xuất được một phần các fact trong dữ liệu.

## 2. Vì sao Advanced có thể tốn hơn ở hội thoại ngắn?

Ở Standard, Advanced xử lý 18.501 prompt tokens, cao hơn 2.003 tokens (khoảng 12,1%) so với 16.498 của Baseline. Nguyên nhân là mỗi lượt Advanced phải nạp thêm `User.md` bên cạnh lịch sử gần đây. Hội thoại còn ngắn nên chưa vượt ngưỡng compact (`Compactions = 0`), vì vậy chưa có phần tiết kiệm token nào bù lại chi phí của persistent memory.

Đây là chi phí đánh đổi để có khả năng nhớ xuyên phiên. Chỉ số agent tokens của Advanced vẫn thấp hơn trong lần chạy này (1.267 so với 1.945), nhưng điều đó không phủ nhận chi phí prompt tăng thêm; hai chỉ số đo hai phần khác nhau của quá trình xử lý.

## 3. Vì sao compact giúp Advanced có lợi thế ở hội thoại dài?

Nếu luôn gửi lại toàn bộ lịch sử, lượng context phải xử lý sẽ tăng dần sau mỗi lượt. Baseline làm như vậy nên dùng 22.253 prompt tokens trong Long-Context Stress.

Advanced theo dõi kích thước lịch sử. Khi vượt ngưỡng, nó tóm tắt các message cũ và chỉ giữ nguyên văn các message gần nhất. Benchmark ghi nhận một lần compact; prompt tokens còn 15.577, giảm 6.676 tokens (khoảng 30,0%) so với Baseline. Tóm tắt giữ lại thông tin chính trong khi loại bỏ phần văn bản dài, lặp lại, nhờ đó lợi ích tăng rõ khi hội thoại dài.

## 4. File memory tăng trưởng ra sao và có rủi ro gì?

Baseline không ghi persistent memory nên memory growth bằng 0. Advanced tăng 7 bytes ở Standard và 146 bytes ở Long-Context Stress. File tăng khi hệ thống phát hiện fact mới và ghi vào `User.md`; vì vậy số lượng fact, độ dài fact và thời gian sử dụng càng lớn thì file càng có xu hướng phình ra.

Các rủi ro chính gồm:

- thông tin cũ, trùng lặp hoặc không còn đúng tích lũy lâu dài;
- trích xuất nhầm câu nói đùa, thông tin tạm thời hoặc dữ kiện nhiễu thành fact;
- chỉnh sửa không đúng có thể tạo ra các fact mâu thuẫn và làm recall sai;
- file lớn dần làm tăng prompt tokens ở mọi lượt, giảm lợi ích của compact;
- dữ liệu hồ sơ có thể chứa thông tin cá nhân, nên cần kiểm soát truy cập và vòng đời lưu trữ.

Để hạn chế các rủi ro này, hệ thống nên cập nhật fact theo khóa thay vì chỉ nối thêm, loại trùng, dùng ngưỡng tin cậy trước khi ghi, hỗ trợ decay/xóa thông tin cũ, giới hạn kích thước hồ sơ và không lưu dữ liệu nhạy cảm nếu không cần thiết.
