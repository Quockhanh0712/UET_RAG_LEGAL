# Refactor Cleanup Inventory

Updated: `2026-04-03`

## Muc tieu

Doc nay tong hop:

- ket qua doc lai docs plan sau refactor
- danh sach file da duoc don gon trong dot nay
- ly do cleanup de giam tech debt va drift

## Docs plan review

Da dong bo lai bo plan theo trang thai hoan tat 8/8:

- `00_index.md`
  - cap nhat status hien tai
  - bo baseline fail cu thanh baseline history (da close)
- `03_master_plan.md`
  - giu phase map + bo sung execution status
- `04_completion_report_2026_04_03.md`
  - chot validation va ket qua regression

## Cleanup da ap dung (source code)

### 1) Query rewrite DTO cleanup (single source of truth)

- file:
  - `src/legal_query_rewriter.py`
  - `src/legal_rag/rewrite/models.py`
  - `tests/unit/test_rewrite_modularization.py`
- thay doi:
  - bo dto class duplicate trong facade `legal_query_rewriter.py`
  - facade gio re-export truc tiep DTO tu `src/legal_rag/rewrite/models.py`
  - them test contract de khoa behavior:
    - `test_public_rewrite_dtos_are_reexported_from_rewrite_models`
- tac dong:
  - giam nguy co drift giua facade va module rewrite moi
  - de bao tri hon khi thay doi DTO

### 2) User store compatibility + log clarity

- file:
  - `web_app/backend/user_store.py`
- thay doi:
  - sparse vector setup khong hard-code `models.Modifier.IDF`
  - xu ly idempotent khi collection da ton tai (conflict 409)
  - log message chinh xac (`created` vs `already exists`)
- tac dong:
  - on dinh hon tren cac version `qdrant_client` khac nhau
  - giam false alarm trong logs

### 3) Model profile listing contract cleanup

- file:
  - `src/services/model_profile_registry.py`
  - `web_app/backend/main.py`
- thay doi:
  - `list_model_profiles()` mac dinh tra ve full preset inventory
  - call-site API runtime dung `enabled_only=True` ro rang
- tac dong:
  - test inventory va runtime behavior khong con mau thuan

## Verification

Da chay regression sau cleanup:

```bash
pytest -q tests/unit/test_rewrite_modularization.py tests/unit/test_legal_query_rewriter.py tests/unit/test_query_rewrite_service.py tests/unit/test_model_profile_registry.py tests/integration/test_api.py::TestDocumentAPI::test_list_documents
```

Result:

- `14 passed`

Wide system check:

```bash
pytest -q tests/unit tests/integration
```

Result:

- `78 passed, 20 skipped`

## Follow-up de xet trong sprint sau (khong block)

- pydantic protected namespace warnings (`model_*`)
- sqlalchemy warning (`declarative_base`)
- swig deprecation warnings
