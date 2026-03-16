# Legal Rewrite V2

Updated: `2026-03-16`

## 1. Muc tieu

Rewrite V2 duoc thiet ke de giai quyet 3 van de cua rewrite cu:

- model tu y viet lai cau hoi va drift sang luat khong lien quan
- model suy luan phap ly qua som ngay o buoc retrieval rewrite
- subquery sinh ra qua chung chung, khong giu du `main_act`, chu the, dieu kien

Rewrite V2 khong con coi LLM la bo may paraphrase. LLM chi dong vai tro `objective extractor`.

## 2. Nguyen tac thiet ke

Rewrite V2 khoa 5 nguyen tac:

1. `surface-role only`
- Khong ket luan ai pham toi, ai vo toi, ai chac chan chiu trach nhiem.
- Chi trich xuat nhung gi nguoi dung neu trong cau hoi.

2. `main_act first`
- `main_act` phai la hanh vi gay hau qua phap ly chinh.
- Khong duoc chon tinh tiet phu nhu `tu thien`, `nop thue`, `me don than`, `hoan canh kho khan` lam hanh vi chinh.

3. `question-unit first`
- Moi y hoi chinh cua user duoc tach thanh `1 question unit`.
- Moi `question unit` sinh toi da `1 retrieval query`.

4. `named law is weak hint`
- Neu user neu `Luat Lao dong`, `Bo luat Hinh su`, `Nghi dinh ...`, rewrite chi giu no o dang `named_law_hint`.
- Ten luat khong duoc phep de main_act va khong duoc tro thanh retrieval anchor chinh.

5. `original query is mandatory anchor`
- Query goc luon duoc giu lai trong pool retrieval.
- Neu rewrite fail quality gate, retrieval quay ve query goc.

## 3. Luong xu ly

```text
user query
-> complexity gate
-> qwen3.5:9b extraction (JSON only)
-> normalize + merge heuristic
-> question units
-> deterministic query synthesis
-> quality gate
-> retrieval query set
```

## 4. Runtime hien tai

Model rewrite mac dinh:
- profile: `qwen3_5_9b_ollama_remote`
- provider: `ollama_generate`
- endpoint: `http://222.255.250.24:11434`
- model: `qwen3.5:9b`

Rewrite call:
- `task="rewrite"`
- `think=false`
- `temperature=0.0`
- structured output qua JSON schema

## 5. Extraction schema

Rewrite V2 dung schema sau:

- `needs_decomposition: boolean`
- `main_act: string`
- `surface_actors[]`
- `conditions[]`
- `explicit_legal_axes[]`
- `question_focus: string`
- `clause_units[]`
- `named_law_hints[]`
- `question_units[]`

### 5.1 surface_actors

Moi actor co:
- `name`
- `actor_type`
- `source_span`

`actor_type` duoc khoa trong tap:
- `ca_nhan`
- `to_chuc`
- `quan_ly`
- `nhan_vien`
- `nguoi_thuc_hien_duoc_neu`
- `nguoi_khac`
- `khong_ro`

### 5.2 conditions

Moi condition co:
- `kind`
- `value`
- `source_span`

`kind` duoc khoa trong tap:
- `tuoi`
- `thoi_diem`
- `so_lan`
- `so_tien`
- `hau_qua`
- `quan_he_lao_dong`
- `khac`

### 5.3 question_units

Day la trung tam cua Rewrite V2.

Moi `question_unit` co:
- `question_label`
- `actor_refs[]`
- `act_span`
- `condition_refs[]`
- `legal_axis`
- `user_asked_text`
- `intent`

`intent` duoc khoa trong tap:
- `xac_dinh_chu_the`
- `dieu_kien_ap_dung`
- `trach_nhiem_hinh_su`
- `trach_nhiem_dan_su`
- `trach_nhiem_hanh_chinh`
- `boi_thuong`

## 6. Complexity gate

Rewrite chi chay khi query du phuc tap. Gate se bat rewrite khi co it nhat mot trong cac dau hieu sau:

- query dai
- query co nhieu menh de
- query co nhieu chu the
- query co nhieu dieu kien
- query co nhieu y hoi
- query co cau truc nhu `hay ca hai`, `ai vi pham ai`, `thuoc ve ai`
- query co named law hint va nhieu actor

Neu query don gian:
- `rewrite_status = skipped`
- `retrieval_query_set = [original query]`

## 7. Deterministic synthesis

Rewrite V2 khong cho LLM tu viet subquery cuoi cung.

Code se sinh retrieval query theo thu tu:
- `main_act`
- `question_units`
- `actor_refs`
- `condition_refs`
- `named_law_hints` neu lien quan va khong lan at `main_act`

### 7.1 Rule sinh query

- moi `question_unit` sinh `1 query`
- toi da `3 rewrite queries`
- `original query` luon duoc append o cuoi
- tong query trong `retrieval_query_set` toi da `4`

### 7.2 Vi du

Case moi truong co 2 y hoi:

```text
Cau hoi goc:
Cong ty A ... xa thang nuoc thai chua xu ly ra song ...
Thanh tich tu thien co duoc tru vao tien phat khong?
Nhan vien lam theo chi dao cap tren co duoc mien trach nhiem hinh su khong?

Question units:
1. tru vao tien phat
2. mien trach nhiem hinh su

Retrieval queries:
1. Tinh tiet duoc xem xet giam hoac tru tien phat doi voi hanh vi xa thang nuoc thai chua xu ly ra song cua Cong ty A; tinh tiet duoc neu: ...
2. Dieu kien xem xet mien hoac giam trach nhiem hinh su doi voi nhan vien ... khi xa thang nuoc thai chua xu ly ra song; dieu kien: ...; nguoi dung neu them Luat Lao dong
3. original query
```

## 8. Quality gate

Rewrite V2 reject ket qua neu gap mot trong cac loi sau:

- `main_act` rong
- `main_act` generic kieu `hanh vi duoc neu`
- query khong chua `main_act`
- query bi drift topic
- named law tro thanh retrieval anchor chinh
- query suy luan som `co toi`, `pham toi`, `vo toi`, `nan nhan`
- query khong cover du `question_units`
- query lam roi so quan trong

Neu fail:
- `rewrite_status = rejected`
- `retrieval_query_set = [original query]`
- `rewrite_debug.quality_fail_reasons` se giai thich ly do reject

## 9. Trace moi duoc expose

Search va Chat payload hien tai expose them:

- `rewrite_context.main_act`
- `rewrite_context.conditions`
- `rewrite_context.clause_units`
- `rewrite_context.named_law_hints`
- `rewrite_context.question_units`
- `rewrite_context.decomposition_policy = objective_question_unit`
- `rewrite_debug.main_act_candidate_reasons`
- `retrieval_query_plan[]`

## 10. QUERY_CONTEXT dua vao answer

Rewrite V2 dua vao answer prompt mot block co cau truc:

- `ORIGINAL_QUERY`
- `MAIN_ACT`
- `SURFACE_ACTORS`
- `KEY_CONDITIONS`
- `NAMED_LAW_HINTS`
- `QUESTION_UNITS`
- `RETRIEVAL_SUBQUERIES_USED`

Luu y:
- `QUERY_CONTEXT` chi de hieu y dinh
- `EVIDENCE_CONTEXT` moi la can cu phap ly
- `NAMED_LAW_HINTS` khong phai can cu

## 11. Gioi han hien tai

Rewrite V2 da giai quyet tot hon cac loi rewrite cu, nhung chua giai quyet toan bo bai toan retrieval.

Van con 3 gioi han:
- top-1 retrieval co the van lech mien
- named law hint co the lam retrieval nghieng ve mien user neu retrieval/rerank chua du manh
- answer co the phai abstain `CHUA_DU_CAN_CU` neu evidence ve khong dung

Noi cach khac:
- rewrite da dung hon
- retrieval van la bottleneck tiep theo

## 12. File lien quan

- `src/legal_query_rewriter.py`
- `src/services/query_rewrite_service.py`
- `src/legal_rag_pipeline.py`
- `src/services/remote_llm_client.py`
- `src/adapters/backend_adapter.py`
- `web_app/backend/main.py`
- `web_app/frontend/src/components/LegalLabArea.jsx`
- `web_app/frontend/src/components/MessageBubble.jsx`