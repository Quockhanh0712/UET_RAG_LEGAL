# Chainlit Hybrid Smoke (10 queries)

- run_id: `run_20260313_chainlit_shared_runtime_smoke`
- created_at: `2026-03-13T15:27:34.778183+00:00`
- model: `truro7/vn-law-embedding`
- collection: `legal_documents__truro7_vn_law_embedding__d768`
- collection_status: `green`
- top_k: `5`
- use_reranker: `True`
- reranker_model: `AITeamVN/Vietnamese_Reranker`
- reranker_alpha: `1.0`
- query_rewriter_enabled: `False`
- query_rewriter_force: `False`
- use_graph: `False`
- require_active_only: `True`

## Summary

- query_total: `10`
- status_ok: `10`
- fallback_count: `0`
- hard_fail_count: `0`
- retrieval_p50_ms: `3675.80`
- retrieval_p95_ms: `3940.24`
- generation_p50_ms: `92443.89`
- generation_p95_ms: `166754.89`
- rerank_used_count: `10`
- rerank_p50_ms: `1907.82`
- rerank_p95_ms: `2075.86`
- rewrite_used_count: `0`
- rewrite_p50_ms: `0.00`
- rewrite_p95_ms: `0.00`

## Gate Checks

- at_least_3_results_each_query: `True`
- top1_citation_non_empty_gte_9: `True`
- llm_non_empty_or_fallback_gte_9: `True`
- reranker_used_when_enabled: `True`

### Q1. Cong an xa xu phat loi khong mang bang lai xe co dung khong?
- status: `ok`
- retrieval_time_ms: `6663.75`
- rewrite_time_ms: `0.00`
- rewrite_used: `False`
- rewrite_status: `disabled`
- rewrite_query_count: `1`
- retrieval_union_count: `100`
- rerank_time_ms: `5190.23`
- used_reranker: `True`
- generation_time_ms: `85592.19`
- error: ``
- rewritten_queries: `['Cong an xa xu phat loi khong mang bang lai xe co dung khong?']`
- missing_numbers: `[]`
- per_subquery_hits: `[{'query': 'Cong an xa xu phat loi khong mang bang lai xe co dung khong?', 'hits': 100}]`
- R1: score=`0.0000` doc_id=`28/2004/TT-BTNMT` path=`Document` type=`FallbackChunk`
  citation: Thông tư 28/2004/TT-BTNMT - THÔNG TƯ Về việc hướng dẫn thực hiện thống kê, kiểm kê đất đai và xây dựng bản đồ hiện trạng sử dụng đất
- R2: score=`0.0000` doc_id=`28/2004/TT-BTNMT` path=`Document` type=`FallbackChunk`
  citation: Thông tư 28/2004/TT-BTNMT - THÔNG TƯ Về việc hướng dẫn thực hiện thống kê, kiểm kê đất đai và xây dựng bản đồ hiện trạng sử dụng đất
- R3: score=`0.0000` doc_id=`20/2016/TT-BLĐTBXH` path=`Dieu 3` type=`ProvisionChunk`
  citation: Thông tư 20/2016/TT-BLĐTBXH - Bộ luật lao động ngày 18 tháng 6 năm 2012 | Dieu 3

### Q2. Phai thuc hien thao tac nap mau vao binh chua va xu ly mau so bo bang hoa chat nhu the nao?
- status: `ok`
- retrieval_time_ms: `3577.48`
- rewrite_time_ms: `0.00`
- rewrite_used: `False`
- rewrite_status: `disabled`
- rewrite_query_count: `1`
- retrieval_union_count: `100`
- rerank_time_ms: `1877.16`
- used_reranker: `True`
- generation_time_ms: `91727.36`
- error: ``
- rewritten_queries: `['Phai thuc hien thao tac nap mau vao binh chua va xu ly mau so bo bang hoa chat nhu the nao?']`
- missing_numbers: `[]`
- per_subquery_hits: `[{'query': 'Phai thuc hien thao tac nap mau vao binh chua va xu ly mau so bo bang hoa chat nhu the nao?', 'hits': 100}]`
- R1: score=`0.0008` doc_id=`14/2017/QH14` path=`Dieu 3` type=`ProvisionChunk`
  citation: Luật 14/2017/QH14 - LUẬT Quản lý, sử dụng vũ khí, vật liệu nổ và công cụ hỗ trợ | Dieu 3
- R2: score=`0.0000` doc_id=`28/2004/TT-BTNMT` path=`Document` type=`FallbackChunk`
  citation: Thông tư 28/2004/TT-BTNMT - THÔNG TƯ Về việc hướng dẫn thực hiện thống kê, kiểm kê đất đai và xây dựng bản đồ hiện trạng sử dụng đất
- R3: score=`0.0000` doc_id=`01/2003/TTLT-TANDTC-VKSNDTC-BCA-BTP-BQP` path=`Document` type=`FallbackChunk`
  citation: Thông tư liên tịch 01/2003/TTLT-TANDTC-VKSNDTC-BCA-BTP-BQP - THÔNG TƯ LIÊN TỊCH Hướng dẫn áp dụng một số quy định tại Chương XXIII "Các tội xâm phạm nghĩa vụ, trách nhiệm của quân nhân" của Bộ luật Hình sự năm 1999 Để áp...

### Q3. Trinh tu danh gia chat luong hang nam doi voi Kiem soat vien duoc quy dinh nhu the nao?
- status: `ok`
- retrieval_time_ms: `3940.24`
- rewrite_time_ms: `0.00`
- rewrite_used: `False`
- rewrite_status: `disabled`
- rewrite_query_count: `1`
- retrieval_union_count: `100`
- rerank_time_ms: `1946.01`
- used_reranker: `True`
- generation_time_ms: `93160.42`
- error: ``
- rewritten_queries: `['Trinh tu danh gia chat luong hang nam doi voi Kiem soat vien duoc quy dinh nhu the nao?']`
- missing_numbers: `[]`
- per_subquery_hits: `[{'query': 'Trinh tu danh gia chat luong hang nam doi voi Kiem soat vien duoc quy dinh nhu the nao?', 'hits': 100}]`
- R1: score=`0.0000` doc_id=`28/2004/TT-BTNMT` path=`Document` type=`FallbackChunk`
  citation: Thông tư 28/2004/TT-BTNMT - THÔNG TƯ Về việc hướng dẫn thực hiện thống kê, kiểm kê đất đai và xây dựng bản đồ hiện trạng sử dụng đất
- R2: score=`0.0000` doc_id=`02/2016/QH14` path=`Dieu 6` type=`ProvisionChunk`
  citation: Luật 02/2016/QH14 - LUẬT TÍN NGƯỠNG, TÔN GIÁO | Dieu 6
- R3: score=`0.0000` doc_id=`04/2012/UBTVQH13` path=`Dieu 1` type=`ProvisionChunk`
  citation: Pháp lệnh 04/2012/UBTVQH13 - PHÁP LỆNH Sửa đổi, bổ sung một số điều của P háp lệnh Ư u đãi người có công với cách mạng C ă n c ứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam năm 1992 đã đ ược sửa đ ổi, b | Dieu 1

### Q4. Tu tu chet sau khi tiem thuoc doc ma khong co than nhan nhan xac thi duoc ho tro mai tang nhu the nao?
- status: `ok`
- retrieval_time_ms: `3550.45`
- rewrite_time_ms: `0.00`
- rewrite_used: `False`
- rewrite_status: `disabled`
- rewrite_query_count: `1`
- retrieval_union_count: `100`
- rerank_time_ms: `1815.77`
- used_reranker: `True`
- generation_time_ms: `83987.70`
- error: ``
- rewritten_queries: `['Tu tu chet sau khi tiem thuoc doc ma khong co than nhan nhan xac thi duoc ho tro mai tang nhu the nao?']`
- missing_numbers: `[]`
- per_subquery_hits: `[{'query': 'Tu tu chet sau khi tiem thuoc doc ma khong co than nhan nhan xac thi duoc ho tro mai tang nhu the nao?', 'hits': 100}]`
- R1: score=`0.0001` doc_id=`02/2016/QH14` path=`Dieu 6` type=`ProvisionChunk`
  citation: Luật 02/2016/QH14 - LUẬT TÍN NGƯỠNG, TÔN GIÁO | Dieu 6
- R2: score=`0.0000` doc_id=`25/2025/TT-BYT` path=`Dieu 31` type=`ProvisionChunk`
  citation: Thông tư 25/2025/TT-BYT - THÔNG TƯ QUY ĐỊNH CHI TIẾT THI HÀNH LUẬT BẢO HIỂM XÃ HỘI, LUẬT AN TOÀN, VỆ SINH LAO ĐỘNG VỀ LĨNH VỰC Y TẾ VÀ MỘT SỐ ĐIỀU CỦA LUẬT | Dieu 31
- R3: score=`0.0000` doc_id=`60/2010/QH12` path=`Dieu 2` type=`ProvisionChunk`
  citation: Luật 60/2010/QH12 - LUẬT Khoáng sản C ăn c ứ H i ế n pháp nư ớ c C ộng h oà xã h ội c h ủ n g h ĩ a Việt Na m n ă m 1992 đã đ ược s ử a đổi, b ổ su n g m ột s ố điều theo N g hị quyết s ố 5 1/2001/ Q | Dieu 2

### Q5. Trach nhiem cua Bo Khoa hoc va Cong nghe ve quan ly va phat trien cong nghiep an ninh duoc quy dinh nhu the nao?
- status: `ok`
- retrieval_time_ms: `3838.97`
- rewrite_time_ms: `0.00`
- rewrite_used: `False`
- rewrite_status: `disabled`
- rewrite_query_count: `1`
- retrieval_union_count: `100`
- rerank_time_ms: `1920.40`
- used_reranker: `True`
- generation_time_ms: `58061.83`
- error: ``
- rewritten_queries: `['Trach nhiem cua Bo Khoa hoc va Cong nghe ve quan ly va phat trien cong nghiep an ninh duoc quy dinh nhu the nao?']`
- missing_numbers: `[]`
- per_subquery_hits: `[{'query': 'Trach nhiem cua Bo Khoa hoc va Cong nghe ve quan ly va phat trien cong nghiep an ninh duoc quy dinh nhu the nao?', 'hits': 100}]`
- R1: score=`0.0000` doc_id=`28/2004/TT-BTNMT` path=`Document` type=`FallbackChunk`
  citation: Thông tư 28/2004/TT-BTNMT - THÔNG TƯ Về việc hướng dẫn thực hiện thống kê, kiểm kê đất đai và xây dựng bản đồ hiện trạng sử dụng đất
- R2: score=`0.0000` doc_id=`346/1998/TT-TCĐC` path=`Dieu 4` type=`ProvisionChunk`
  citation: Thông tư 346/1998/TT-TCĐC - THÔNG TƯ Hướng dẫn thủ tục đăng ký đất đai, lập hồ sơ địa chính và cấp giấy chứng nhận quyền sử dụng đất | Dieu 4
- R3: score=`0.0000` doc_id=`101/2015/QH13` path=`Dieu 313` type=`ProvisionChunk`
  citation: Bộ luật 101/2015/QH13 - BỘ LUẬT TỐ TỤNG HÌNH SỰ | Dieu 313

### Q6. Dap pha bien bao khu vuc bien gioi bi phat the nao?
- status: `ok`
- retrieval_time_ms: `3703.49`
- rewrite_time_ms: `0.00`
- rewrite_used: `False`
- rewrite_status: `disabled`
- rewrite_query_count: `1`
- retrieval_union_count: `100`
- rerank_time_ms: `1958.12`
- used_reranker: `True`
- generation_time_ms: `112910.21`
- error: ``
- rewritten_queries: `['Dap pha bien bao khu vuc bien gioi bi phat the nao?']`
- missing_numbers: `[]`
- per_subquery_hits: `[{'query': 'Dap pha bien bao khu vuc bien gioi bi phat the nao?', 'hits': 100}]`
- R1: score=`0.0001` doc_id=`79/2015/QH13` path=`Dieu 3` type=`ProvisionChunk`
  citation: Luật 79/2015/QH13 - LUẬT Thú y _ Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam; Quốc hội ban hành Luật thú y. Chương I NHỮNG QUY ĐỊNH CHUNG | Dieu 3
- R2: score=`0.0000` doc_id=`60/2010/QH12` path=`Dieu 25` type=`ProvisionChunk`
  citation: Luật 60/2010/QH12 - LUẬT Khoáng sản C ăn c ứ H i ế n pháp nư ớ c C ộng h oà xã h ội c h ủ n g h ĩ a Việt Na m n ă m 1992 đã đ ược s ử a đổi, b ổ su n g m ột s ố điều theo N g hị quyết s ố 5 1/2001/ Q | Dieu 25
- R3: score=`0.0000` doc_id=`79/2015/QH13` path=`Dieu 3` type=`ProvisionChunk`
  citation: Luật 79/2015/QH13 - LUẬT Thú y _ Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam; Quốc hội ban hành Luật thú y. Chương I NHỮNG QUY ĐỊNH CHUNG | Dieu 3

### Q7. Muc tieu phat trien khu cong nghiep ho tro duoc quy dinh nhu the nao?
- status: `ok`
- retrieval_time_ms: `3641.35`
- rewrite_time_ms: `0.00`
- rewrite_used: `False`
- rewrite_status: `disabled`
- rewrite_query_count: `1`
- retrieval_union_count: `100`
- rerank_time_ms: `1895.24`
- used_reranker: `True`
- generation_time_ms: `99418.08`
- error: ``
- rewritten_queries: `['Muc tieu phat trien khu cong nghiep ho tro duoc quy dinh nhu the nao?']`
- missing_numbers: `[]`
- per_subquery_hits: `[{'query': 'Muc tieu phat trien khu cong nghiep ho tro duoc quy dinh nhu the nao?', 'hits': 100}]`
- R1: score=`0.0000` doc_id=`64/2010/QH12` path=`Dieu 4` type=`ProvisionChunk`
  citation: Luật 64/2010/QH12 - LUẬT Tố tụng hành chính C ă n c ứ H i ế n phá p nư ớ c C ộn g ho à x ã h ộ i ch ủ n g h ĩ a Việ t Na m n ă m 1992 đ ã đ ược s ửa đ ổi, b ổ su n g m ột s ố điều theo N g hị quyết s | Dieu 4
- R2: score=`0.0000` doc_id=`64/2010/QH12` path=`Dieu 12` type=`ProvisionChunk`
  citation: Luật 64/2010/QH12 - LUẬT Tố tụng hành chính C ă n c ứ H i ế n phá p nư ớ c C ộn g ho à x ã h ộ i ch ủ n g h ĩ a Việ t Na m n ă m 1992 đ ã đ ược s ửa đ ổi, b ổ su n g m ột s ố điều theo N g hị quyết s | Dieu 12
- R3: score=`0.0000` doc_id=`60/2010/QH12` path=`Dieu 13` type=`ProvisionChunk`
  citation: Luật 60/2010/QH12 - LUẬT Khoáng sản C ăn c ứ H i ế n pháp nư ớ c C ộng h oà xã h ội c h ủ n g h ĩ a Việt Na m n ă m 1992 đã đ ược s ử a đổi, b ổ su n g m ột s ố điều theo N g hị quyết s ố 5 1/2001/ Q | Dieu 13

### Q8. Thua phat lai duoc tong dat nhung giay to ho so tai lieu nao?
- status: `ok`
- retrieval_time_ms: `3443.96`
- rewrite_time_ms: `0.00`
- rewrite_used: `False`
- rewrite_status: `disabled`
- rewrite_query_count: `1`
- retrieval_union_count: `100`
- rerank_time_ms: `1672.10`
- used_reranker: `True`
- generation_time_ms: `175614.49`
- error: ``
- rewritten_queries: `['Thua phat lai duoc tong dat nhung giay to ho so tai lieu nao?']`
- missing_numbers: `[]`
- per_subquery_hits: `[{'query': 'Thua phat lai duoc tong dat nhung giay to ho so tai lieu nao?', 'hits': 100}]`
- R1: score=`0.0002` doc_id=`28/2004/TT-BTNMT` path=`Document` type=`FallbackChunk`
  citation: Thông tư 28/2004/TT-BTNMT - THÔNG TƯ Về việc hướng dẫn thực hiện thống kê, kiểm kê đất đai và xây dựng bản đồ hiện trạng sử dụng đất
- R2: score=`0.0001` doc_id=`28/2018/QH14` path=`Dieu 12` type=`ProvisionChunk`
  citation: Luật 28/2018/QH14 - LUẬT SỬA ĐỔI, BỔ SUNG MỘT SỐ ĐIỀU CỦA 11 LUẬT CÓ LIÊN QUAN ĐẾN QUY HOẠCH | Dieu 12
- R3: score=`0.0001` doc_id=`05/2023/TT-BGTVT` path=`Dieu 12` type=`ProvisionChunk`
  citation: Thông tư 05/2023/TT-BGTVT - THÔNG TƯ Sửa đổi, bổ sung một số điều của các thông tư liên quan đến lĩnh vực vận tải đường bộ, dịch vụ hỗ trợ vận tải đường bộ, phương tiện và người lái | Dieu 12

### Q9. Muc phat khi su dung trai phep long duong do thi he pho de rua xe
- status: `ok`
- retrieval_time_ms: `3648.12`
- rewrite_time_ms: `0.00`
- rewrite_used: `False`
- rewrite_status: `disabled`
- rewrite_query_count: `1`
- retrieval_union_count: `100`
- rerank_time_ms: `1865.56`
- used_reranker: `True`
- generation_time_ms: `90983.39`
- error: ``
- rewritten_queries: `['Muc phat khi su dung trai phep long duong do thi he pho de rua xe']`
- missing_numbers: `[]`
- per_subquery_hits: `[{'query': 'Muc phat khi su dung trai phep long duong do thi he pho de rua xe', 'hits': 100}]`
- R1: score=`0.0002` doc_id=`53/2024/QH15` path=`Dieu 2` type=`ProvisionChunk`
  citation: Luật 53/2024/QH15 - LUẬT PHÒNG, CHỐNG MUA BÁN NGƯỜI | Dieu 2
- R2: score=`0.0000` doc_id=`28/2004/TT-BTNMT` path=`Document` type=`FallbackChunk`
  citation: Thông tư 28/2004/TT-BTNMT - THÔNG TƯ Về việc hướng dẫn thực hiện thống kê, kiểm kê đất đai và xây dựng bản đồ hiện trạng sử dụng đất
- R3: score=`0.0000` doc_id=`09/2013/TTLT-BCA-BQP-BTP-VKSNDTC-TANDTC` path=`Dieu 2` type=`ProvisionChunk`
  citation: Thông tư liên tịch 09/2013/TTLT-BCA-BQP-BTP-VKSNDTC-TANDTC - THÔNG TƯ LIÊN TỊCH Hướng dẫn áp dụng quy định tại Chương XIX của Bộ luật Hình sự về các tội xâm phạm trật tự, an toàn giao thông | Dieu 2

### Q10. Hai cong ty con chung cong ty me co duoc tham du goi thau cua nhau khong?
- status: `ok`
- retrieval_time_ms: `3841.25`
- rewrite_time_ms: `0.00`
- rewrite_used: `False`
- rewrite_status: `disabled`
- rewrite_query_count: `1`
- retrieval_union_count: `100`
- rerank_time_ms: `2075.86`
- used_reranker: `True`
- generation_time_ms: `166754.89`
- error: ``
- rewritten_queries: `['Hai cong ty con chung cong ty me co duoc tham du goi thau cua nhau khong?']`
- missing_numbers: `[]`
- per_subquery_hits: `[{'query': 'Hai cong ty con chung cong ty me co duoc tham du goi thau cua nhau khong?', 'hits': 100}]`
- R1: score=`0.0003` doc_id=`59/2020/QH14` path=`Dieu 4` type=`ProvisionChunk`
  citation: Luật 59/2020/QH14 - LUẬT DOANH NGHIỆP | Dieu 4
- R2: score=`0.0001` doc_id=`13/1999/QH10` path=`Dieu 3` type=`ProvisionChunk`
  citation: Luật 13/1999/QH10 - LUẬT Doanh nghiệp Để góp phần phát huy nội lực phục vụ sự nghiệp công nghiệp hóa, hiện đại hóa đất nước | Dieu 3
- R3: score=`0.0000` doc_id=`33/2005/QH11` path=`Dieu 217` type=`ProvisionChunk`
  citation: Bộ luật 33/2005/QH11 - BỘ LUẬT DÂN SỰ | Dieu 217
