---
description: Tự động tạo note kinh nghiệm sau mỗi vấn đề đã xử lý
---

# Workflow: Tạo Note Kinh Nghiệm Phát Triển

Sử dụng workflow này khi đã giải quyết xong một lỗi, một vấn đề kỹ thuật, hoặc hoàn thành một tính năng quan trọng để lưu lại bài học.

### Các bước thực hiện:

1. **Xác định thông tin**:
    - Tiêu đề note (ngắn gọn, súc tích).
    - Bối cảnh (Project, Tech Stack).
    - Vấn đề cụ thể đã gặp.
    - Giải pháp chi tiết đã áp dụng.
    - Bài học rút ra.

2. **Dùng template**:
    - Đọc nội dung mẫu tại `data/notes/template.md`.
    - Tạo file mới tại `data/notes/<ten-file-slug>.md`.

3. **Điền thông tin**:
    - Cập nhật Frontmatter: `tags`, `project`, `date`.
    - Viết nội dung dựa trên các Heading có sẵn trong template.
    - Đảm bảo có Code Snippet minh họa nếu có.

4. **Xác nhận**:
    - Thông báo cho người dùng đã tạo note kèm đường dẫn file.
