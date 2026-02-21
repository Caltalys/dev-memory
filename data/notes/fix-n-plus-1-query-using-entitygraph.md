---
tags: [spring-boot, jpa, performance]
project: "order-service"
date: 2026-02-21
---

# Fix N+1 Query với EntityGraph

## Bối cảnh (Context)
Order service sử dụng Spring Data JPA với quan hệ OneToMany giữa Order và OrderItem.

## Vấn đề (Problem)
Khi load danh sách Order, hệ thống phát sinh N+1 query làm giảm performance.

## Giải pháp (Solution)
Sử dụng @EntityGraph để fetch join.

```java
@EntityGraph(attributePaths = {"items"})
List<Order> findAll();
```

## Bài học (Lesson Learned)

- Luôn bật log SQL khi dev
- Kiểm tra Lazy/Eager hợp lý
- EntityGraph dễ maintain hơn custom JPQL

## Tham khảo (References)

https://docs.spring.io/spring-data/jpa/docs/current/reference/html/#jpa.entity-graph

---