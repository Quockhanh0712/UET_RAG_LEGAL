# Backend + Frontend Smoke 10 (run_20260314_backend_frontend_smoke10_v2)

## Summary
- `search_queries_total` = `10`
- `search_ok` = `10`
- `search_top1_citation_non_empty` = `10`
- `search_at_least_3_results` = `10`
- `search_p50_client_ms` = `5501.07`
- `search_p95_client_ms` = `6253.65`
- `matrix_ok` = `4`
- `frontend_ok` = `True`
- `backend_ok` = `True`
- `collection` = `legal_documents__truro7_vn_law_embedding__d768`
- `backend_mode` = `shared_legal_pipeline`

## Search Smoke
- `Q1` status=`ok` http=`200` results=`5` top1_doc=`28/2004/TT-BTNMT` top1_path=`Document`
- `Q2` status=`ok` http=`200` results=`5` top1_doc=`61/2010/QH12` top1_path=`Dieu 1`
- `Q3` status=`ok` http=`200` results=`5` top1_doc=`28/2004/TT-BTNMT` top1_path=`Document`
- `Q4` status=`ok` http=`200` results=`5` top1_doc=`61/2010/QH12` top1_path=`Dieu 1`
- `Q5` status=`ok` http=`200` results=`5` top1_doc=`61/2010/QH12` top1_path=`Dieu 1`
- `Q6` status=`ok` http=`200` results=`5` top1_doc=`61/2010/QH12` top1_path=`Dieu 1`
- `Q7` status=`ok` http=`200` results=`5` top1_doc=`28/2004/TT-BTNMT` top1_path=`Document`
- `Q8` status=`ok` http=`200` results=`5` top1_doc=`28/2004/TT-BTNMT` top1_path=`Document`
- `Q9` status=`ok` http=`200` results=`5` top1_doc=`28/2004/TT-BTNMT` top1_path=`Document`
- `Q10` status=`ok` http=`200` results=`5` top1_doc=`62/2010/QH12` top1_path=`Dieu 1`

## Matrix Checks
- `search_legal_model_rewrite` status=`ok` http=`200` model=`legal_tgi_remote` mode=`rewrite_combine` fallback=`False` rewrite_status=`ok`
- `search_qwen_rewrite` status=`ok` http=`200` model=`qwen3_8b_ollama_remote` mode=`rewrite_combine` fallback=`False` rewrite_status=`ok`
- `chat_qwen_original` status=`ok` http=`200` model=`qwen3_8b_ollama_remote` mode=`original` fallback=`False` rewrite_status=`disabled`
- `chat_legal_model_rewrite` status=`ok` http=`200` model=`legal_tgi_remote` mode=`rewrite_combine` fallback=`False` rewrite_status=`ok`