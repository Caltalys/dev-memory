---
tags: [pip, python, troubleshooting, cuda, torch]
project: "DevMemory Pro"
date: 2026-02-21
---

# Fix Pip AssertionError & No Space Left (CUDA/Torch Issue)

## Bối cảnh (Context)
Trong quá trình cài đặt môi trường phát triển cho dự án DevMemory Pro (Local RAG), tôi đã gặp phải lỗi cài đặt thư viện thông qua `pip`.

## Vấn đề (Problem)
1. **AssertionError**: Khi chạy `pip install`, trình giải quyết phụ thuộc của `pip` (phiên bản 22.0.2) gặp lỗi nội bộ.
2. **No Space Left on Device**: Khi cài đặt `sentence-transformers`, `pip` mặc định tải xuống `torch` phiên bản có hỗ trợ CUDA (NVIDIA), dung lượng hơn 2GB, làm đầy ổ cứng (chỉ còn ~4GB trống).

## Giải pháp (Solution)
1. **Nâng cấp pip**: Cập nhật `pip` lên phiên bản mới nhất (26.0.1) để sửa lỗi `AssertionError`.
2. **Dọn dẹp bộ nhớ**:
   - Gỡ bỏ các gói `nvidia-*` đã tải dở.
   - Chạy `pip cache purge` để giải phóng dung lượng đĩa.
3. **Cài đặt bản CPU-only**: Sử dụng index URL của PyTorch dành riêng cho CPU và cập nhật `requirements.txt`.

```bash
# Cài đặt bản CPU
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Cập nhật requirements.txt
# Thêm --extra-index-url https://download.pytorch.org/whl/cpu ở đầu file
```

## Bài học (Lesson Learned)
- Luôn đảm bảo `pip` ở phiên bản mới nhất khi làm việc với các thư viện AI/ML phức tạp.
- Đối với các môi trường không có GPU hoặc dung lượng đĩa hạn chế, cần chỉ định rõ phiên bản `+cpu` của PyTorch để tránh tải các thư viện CUDA nặng nề.
- Sử dụng `--extra-index-url` trong `requirements.txt` là cách tốt nhất để cấu hình nguồn tải cho team.

## Tham khảo (References)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [Pip Issue Tracker](https://github.com/pypa/pip/issues)
