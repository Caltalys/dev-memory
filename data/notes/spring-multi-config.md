---
tags: [spring-boot, configuration, properties]
project: "microservice-auth"
date: 2026-02-21
---

# Quản lý cấu hình với @ConfigurationProperties

## Bối cảnh (Context)
Dự án microservice-auth cần quản lý nhiều cấu hình liên quan đến JWT, Redis và Kafka.

## Vấn đề (Problem)
Sử dụng @Value gây rải rác cấu hình và khó maintain khi số lượng properties tăng lên.

## Giải pháp (Solution)
Sử dụng @ConfigurationProperties để bind cấu hình theo nhóm.

```java
@Configuration
@ConfigurationProperties(prefix = "app.jwt")
@Getter
@Setter
public class JwtProperties {
    private String secret;
    private long expiration;
}
```
application.yml:
```
app:
  jwt:
    secret: my-secret
    expiration: 3600
```

## Bài học (Lesson Learned)

- Group cấu hình giúp code clean hơn
- Dễ validate với @Validated
- Dễ test và mock

## Tham khảo (References)

https://docs.spring.io/spring-boot/docs/current/reference/html/features.html#features.external-config

---