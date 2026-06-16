# 비용 추정 — 당일 배포/삭제 기준 (australiaeast, 2026-06 가격대)

전제: 랩 세션 1회 = 클러스터 가동 약 3시간, 당일 RG 삭제.

| 항목 | 단가 (대략) | 사용량 | 비용 |
|---|---|---|---|
| Databricks 워크스페이스 (Premium) | 워크스페이스 자체는 무료 — 컴퓨트 사용 시에만 과금 | — | $0 |
| DBU (All-Purpose Compute, Premium) | ~USD 0.55/DBU | DS3_v2 single node = 0.75 DBU/h × 3h | ~USD 1.24 |
| VM (Standard_DS3_v2, 4 vCPU 14GB) | ~AUD 0.42/h | 3h × 1 node | ~AUD 1.26 |
| ADLS Gen2 (Standard LRS) | ~AUD 0.03/GB/월 | <1GB, 1일 | ~AUD 0.01 |
| Access Connector | 무료 | — | $0 |
| 트랜잭션/이그레스 | 미미 | — | <AUD 0.10 |

**합계: 약 AUD 3.5–5 (USD 2.5–3.5)** / 세션

## 비용이 튀는 함정 (주의)

1. **클러스터 auto-terminate 미설정** — 30분으로 반드시 설정. 밤새 켜두면 ~AUD 25/일.
2. **멀티노드 기본값** — 클러스터 생성 시 기본이 2–8 worker autoscale. 반드시 "Single node" 선택.
3. **Managed RG 잔존** — 워크스페이스 리소스를 지우면 managed RG는 자동 삭제되지만,
   `az group delete`가 실패한 채 방치되면 managed RG 내 NAT GW/IP가 과금될 수 있음.
   teardown 후 `az group list --query "[?contains(name,'dbx-churn')]"` 로 잔존 확인.
4. **Premium SKU DBU 단가** — Standard보다 ~50% 비쌈. UC/RBAC 데모가 목적이므로 Premium 유지하되 세션을 짧게.

## 더 아끼려면

- Spot VM 사용 시 VM 비용 60–80% 절감 (랩 용도로 적합)
- 14일 Databricks 무료 평가판 워크스페이스면 DBU 요금 $0 (VM 비용만 발생)
