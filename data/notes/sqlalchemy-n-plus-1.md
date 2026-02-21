---
tags: [python, fastapi, performance]
project: dev-memory
date: 2026-02-21
---

# Xử Lý Lỗi N+1 Trong SQLAlchemy

## Bối cảnh (Context)
Dự án web app FastAPI + SQLAlchemy. Khi load danh sách user kèm orders, query chạy N+1 lần.

## Vấn đề (Problem)
Lazy loading mặc định trong SQLAlchemy khiến mỗi lần access `user.orders` lại trigger một query riêng.
Với 100 users → 101 queries.

## Giải pháp (Solution)
Dùng `selectinload` hoặc `joinedload` khi query:

```python
from sqlalchemy.orm import selectinload

# Thay vì:
users = db.query(User).all()

# Dùng eager loading:
users = db.query(User).options(selectinload(User.orders)).all()
```

Với relationship phức tạp hơn, dùng `joinedload`:
```python
users = (
    db.query(User)
    .options(joinedload(User.orders).selectinload(Order.items))
    .all()
)
```

## Bài học (Lesson Learned)
- Luôn kiểm tra SQL log khi phát triển: `echo=True` trong engine
- `selectinload` tốt hơn `joinedload` cho collection (1-to-many)
- `joinedload` phù hợp với many-to-one hoặc one-to-one

## Tham khảo (References)
- https://docs.sqlalchemy.org/en/14/orm/loading_relationships.html
