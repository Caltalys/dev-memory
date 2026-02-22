---
description: Tự động tạo note kinh nghiệm sau mỗi vấn đề đã xử lý
---

# Workflow: Tạo Note Kinh Nghiệm Phát Triển

Sử dụng workflow này khi đã giải quyết xong một lỗi, một vấn đề kỹ thuật, hoặc hoàn thành một tính năng quan trọng để lưu lại bài học.

### Các bước thực hiện:

1. **Xác định thông tin**:
    - Tiêu đề note (Ngắn gọn, phản ánh đúng lỗi/vấn đề).
    - Các thành phần quan trọng: Triệu chứng, Nguyên nhân, và Giải pháp chi tiết.

2. **Dùng template**:
    - Đọc nội dung mẫu tại `data/notes/template.md`.
    - Tạo file mới tại `data/notes/YYYY-MM-DD-<ten-file-slug>.md` (Nên có prefix ngày tháng để dễ quản lý).

3. **Điền thông tin theo cấu trúc Semantic Sectioning**:
    - **QUAN TRỌNG**: Giữ đúng các Header `##` cấp 2 vì hệ thống sử dụng chúng để chia chunk và phân loại dữ liệu (`Triệu chứng`, `Nguyên nhân`, `Giải pháp`, `Bài học`).
    - Cập nhật Frontmatter: `tags`, `project`, `date`.
    - Đảm bảo có Code Snippet trong các block ` ``` ` để hệ thống không cắt ngang mã nguồn.

4. **Xác nhận**:
    - Thông báo cho người dùng đã tạo note kèm đường dẫn file.
    - Gợi ý người dùng kiểm tra lại nội dung để đảm bảo tính chính xác cho RAG.
