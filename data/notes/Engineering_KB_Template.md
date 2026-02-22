---
created_at: YYYY-MM-DD
status: draft
updated_at: YYYY-MM-DD
version: 1.0.0
---

# \[ENGINEERING KB\] \<Tiêu đề rõ ràng, mô tả đúng vấn đề kỹ thuật\>

------------------------------------------------------------------------

## 1. Executive Summary

> Tóm tắt 5--10 dòng: - Vấn đề là gì? - Ảnh hưởng production ra sao? -
> Quyết định kiến trúc là gì? - Recommendation chính thức của team?

------------------------------------------------------------------------

## 2. Context (Bối cảnh)

### 2.1 Business Context

-   Service thuộc domain nào?
-   Ảnh hưởng tới luồng nghiệp vụ nào?
-   SLA/SLO liên quan?

### 2.2 Technical Context

-   Stack sử dụng (Spring Boot version, DB, Kafka, Redis...)
-   Kiến trúc (monolith/microservice/event-driven)
-   Môi trường (K8s, on-prem, cloud)

------------------------------------------------------------------------

## 3. Problem Statement (Vấn đề)

### 3.1 Mô tả chi tiết

-   Vấn đề xảy ra khi nào?
-   Frequency?
-   Ảnh hưởng dữ liệu / performance / security?

### 3.2 Symptoms (Triệu chứng)

-   Log mẫu
-   Exception mẫu
-   Metric bất thường

### 3.3 Root Cause Analysis

-   Phân tích kỹ thuật
-   Sai assumption nào?
-   Sai design ở đâu?

------------------------------------------------------------------------

## 4. Decision (Quyết định chính thức của team)

> Đây là phần QUAN TRỌNG NHẤT

-   Team thống nhất chọn giải pháp gì?
-   Vì sao chọn giải pháp này?
-   Không được làm theo cách nào?

------------------------------------------------------------------------

## 5. Solution Design

### 5.1 High-Level Approach

-   Kiến trúc tổng thể
-   Pattern áp dụng (Outbox, Saga, CQRS, Retry, Circuit Breaker...)

### 5.2 Detailed Implementation

``` java
// Code chuẩn production
```

``` yaml
# config chuẩn production
```

### 5.3 Data Flow (Nếu có)

-   Request flow
-   Transaction boundary
-   Event publishing flow

------------------------------------------------------------------------

## 6. Trade-offs Analysis

  Option   Ưu điểm   Nhược điểm   Khi nào dùng
  -------- --------- ------------ --------------

Phải ghi rõ vì sao KHÔNG chọn các option khác.

------------------------------------------------------------------------

## 7. Anti-Patterns (Không được làm)

-   ❌ ...
-   ❌ ...
-   ❌ ...

------------------------------------------------------------------------

## 8. Production Checklist

### 8.1 Code Level

-   [ ] Exception handling chuẩn
-   [ ] Logging chuẩn hóa
-   [ ] Timeout cấu hình
-   [ ] Validation đầy đủ

### 8.2 Infra Level

-   [ ] Config production khác dev
-   [ ] Resource limit set
-   [ ] Liveness/Readiness probe

### 8.3 Security

-   [ ] Không lộ sensitive data
-   [ ] AuthZ/AuthN validate
-   [ ] Dependency scan

------------------------------------------------------------------------

## 9. Testing Strategy

### 9.1 Unit Test

-   Coverage yêu cầu ≥ ?

### 9.2 Integration Test

-   Dùng Testcontainers?
-   Có test concurrency?

### 9.3 Chaos / Failure Test

-   DB down?
-   Kafka delay?
-   Network timeout?

------------------------------------------------------------------------

## 10. Observability & Monitoring

### 10.1 Logging

-   Format chuẩn (JSON?)
-   traceId bắt buộc?

### 10.2 Metrics

-   Latency
-   Error rate
-   Throughput

### 10.3 Alerting

-   Alert khi nào?
-   Threshold?

------------------------------------------------------------------------

## 11. Performance Considerations

-   Bottleneck ở đâu?
-   Connection pool config?
-   Caching strategy?

------------------------------------------------------------------------

## 12. Security Considerations

-   Injection risk?
-   Data exposure?
-   Token handling?
-   Encryption at rest / transit?

------------------------------------------------------------------------

## 13. Migration / Rollout Plan

-   Backward compatibility?
-   Feature flag?
-   Blue/Green hay Rolling update?
-   Data migration script?

------------------------------------------------------------------------

## 14. Rollback Plan

-   Nếu deployment fail?
-   Nếu data corrupt?
-   Nếu event duplication?

------------------------------------------------------------------------

## 15. Postmortem Notes (Nếu liên quan incident)

-   Incident ID
-   Timeline
-   Impact
-   Action items

------------------------------------------------------------------------

## 16. Lessons Learned

-   Điều gì team đã hiểu sai?
-   Điều gì cần standard hóa?
-   Cần bổ sung guideline nào?

------------------------------------------------------------------------

## 17. Related Documents

-   ADR-xxx
-   PR link
-   Incident report
-   External reference

------------------------------------------------------------------------

# Engineering Standards

## Definition of Done (KB Article)

Một KB được coi là hoàn chỉnh khi:

-   [ ] Có Decision rõ ràng
-   [ ] Có Trade-off analysis
-   [ ] Có Production checklist
-   [ ] Có Anti-pattern
-   [ ] Có Testing strategy
-   [ ] Có Rollback plan
-   [ ] Được reviewer approve

------------------------------------------------------------------------

# Versioning Policy

-   Major: Thay đổi kiến trúc
-   Minor: Cập nhật guideline
-   Patch: Sửa chính tả / bổ sung ví dụ

------------------------------------------------------------------------

# Ownership

-   Owner chịu trách nhiệm cập nhật khi:
    -   Upgrade framework
    -   Kiến trúc thay đổi
    -   Incident mới xảy ra

Nếu không có owner → KB bị deprecated.
