# Chainlit Hybrid Smoke (10 queries)

- run_id: `run_20260312_153902`
- created_at: `2026-03-12T15:45:08.871257+00:00`
- model: `truro7/vn-law-embedding`
- collection: `legal_documents__truro7_vn_law_embedding__d768`
- top_k: `5`
- use_reranker: `True`
- reranker_model: `AITeamVN/Vietnamese_Reranker`
- reranker_alpha: `1.0`
- use_graph: `False`
- require_active_only: `True`

## Summary

- query_total: `10`
- status_ok: `10`
- fallback_count: `0`
- hard_fail_count: `0`
- retrieval_p50_ms: `3235.31`
- retrieval_p95_ms: `3635.99`
- generation_p50_ms: `28336.61`
- generation_p95_ms: `38074.63`
- rerank_used_count: `10`
- rerank_p50_ms: `1243.05`
- rerank_p95_ms: `1619.71`

## Gate Checks

- at_least_3_results_each_query: `True`
- top1_citation_non_empty_gte_9: `True`
- llm_non_empty_or_fallback_gte_9: `True`
- reranker_used_when_enabled: `True`

### Q1. Công an xã xử phạt lỗi không mang bằng lái xe có đúng không?
- status: `ok`
- retrieval_time_ms: `8771.90`
- rerank_time_ms: `6079.87`
- used_reranker: `True`
- generation_time_ms: `50919.64`
- error: ``
- R1: score=`0.0006` doc_id=`44/2002/PL-UBTVQH10` path=`Dieu 12` type=`ProvisionChunk`
  citation: Pháp lệnh 44/2002/PL-UBTVQH10 - PHÁP LỆNH Xử lý vi phạm hành chính ____________ Để đấu tranh phòng ngừa và chống vi phạm hành chính, góp phần giữ vững an ninh, trật tự, an toàn xã hội, bảo vệ lợi ích của Nhà nước, quyền,...
- R2: score=`0.0005` doc_id=`44/2002/PL-UBTVQH10` path=`Dieu 12` type=`ProvisionChunk`
  citation: Pháp lệnh 44/2002/PL-UBTVQH10 - PHÁP LỆNH Xử lý vi phạm hành chính ____________ Để đấu tranh phòng ngừa và chống vi phạm hành chính, góp phần giữ vững an ninh, trật tự, an toàn xã hội, bảo vệ lợi ích của Nhà nước, quyền,...
- R3: score=`0.0003` doc_id=`06/2012/TT-BGTVT` path=`Dieu 1 > Khoan 3` type=`ClauseChunk`
  citation: Thông tư 06/2012/TT-BGTVT - THÔNG TƯ Quy định trách nhiệm và xử lý vi phạm trong công tác đào tạo, sát hạch, cấp giấy phép lái xe cơ giới đường bộ _______________________________ Căn cứ Luật Giao thông đường bộ ngày 13 t...

### Q2. Phải thực hiện thao tác nạp mẫu vào bình chứa và xử lý mẫu sơ bộ bằng hóa chất như thế nào?
- status: `ok`
- retrieval_time_ms: `3206.06`
- rerank_time_ms: `1164.40`
- used_reranker: `True`
- generation_time_ms: `25325.33`
- error: ``
- R1: score=`0.0001` doc_id=`69/2025/QH15` path=`Dieu 23 > Khoan 4` type=`ClauseChunk`
  citation: Luật 69/2025/QH15 - LUẬT HÓA CHẤT Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam đã được sửa đổi, bổ sung một số điều theo Nghị quyết số 203/2025/QH15; Quốc hội ban hành Luật Hóa chất. Chương I QUY ĐỊNH CHUNG |...
- R2: score=`0.0001` doc_id=`19/2016/NĐ-CP` path=`Dieu 3 > Khoan 16` type=`ClauseChunk`
  citation: Nghị định 19/2016/NĐ-CP - V ề kinh doanh kh í _____ Căn cứ Luật T ổ chức Chính phủ ngày 19 tháng 6 năm 2015; Căn cứ Luật Thương mại ngày 14 tháng 6 năm 2005; Căn cứ Luật Chất l ượng sản phẩm, hàng hóa ngày 21 tháng 11 nă...
- R3: score=`0.0001` doc_id=`90/2009/NĐ-CP` path=`Dieu 3 > Khoan 4` type=`ClauseChunk`
  citation: Nghị định 90/2009/NĐ-CP - NGHỊ ĐỊNH Quy định về xử phạt vi phạm hành chính trong hoạt động hoá chất _____________________ CHÍNH PHỦ Căn cứ Luật Tổ chức Chính phủ ngày 25 tháng 12 năm 2001; Căn cứ Luật Hóa chất ngày 21 th...

### Q3. Trình tự đánh giá chất lượng hàng năm đối với Kiểm soát viên được quy định như thế nào?
- status: `ok`
- retrieval_time_ms: `3157.16`
- rerank_time_ms: `1181.58`
- used_reranker: `True`
- generation_time_ms: `24846.54`
- error: ``
- R1: score=`0.0799` doc_id=`01/2022/TT-BQP` path=`Dieu 7` type=`ProvisionChunk`
  citation: Thông tư 01/2022/TT-BQP - THÔNG TƯ Quy định về quản lý người giữ chức chức vụ, chức danh và người đại diện phần vốn nhà nước tại doanh nghiệp do Bộ Quốc phòng quản lý Căn cứ Luật Doanh nghiệp ngày 17 tháng 6 năm 2020; Că...
- R2: score=`0.0799` doc_id=`01/2021/TT-BQP` path=`Dieu 7` type=`ProvisionChunk`
  citation: Thông tư 01/2021/TT-BQP - THÔNG TƯ Quy định về quản lý người giữ chức vụ, chức danh và người đại diện phần vốn nhà nước tại doanh nghiệp do Bộ Quốc phòng quản lý 1/01/clip_image001. png" width="114" /> Căn cứ Luật Doanh...
- R3: score=`0.0165` doc_id=`01/2021/TT-BQP` path=`Dieu 10` type=`ProvisionChunk`
  citation: Thông tư 01/2021/TT-BQP - THÔNG TƯ Quy định về quản lý người giữ chức vụ, chức danh và người đại diện phần vốn nhà nước tại doanh nghiệp do Bộ Quốc phòng quản lý 1/01/clip_image001. png" width="114" /> Căn cứ Luật Doanh...

### Q4. Tử tù chết sau khi tiêm thuốc độc mà không có thân nhân nhận xác thì được hỗ trợ mai táng như thế nào?
- status: `ok`
- retrieval_time_ms: `3635.99`
- rerank_time_ms: `1619.71`
- used_reranker: `True`
- generation_time_ms: `25958.12`
- error: ``
- R1: score=`0.0154` doc_id=`159/2025/NĐ-CP` path=`Dieu 14 > Khoan 2` type=`ClauseChunk`
  citation: Nghị định 159/2025/NĐ-CP - NGHỊ ĐỊNH Quy định chi tiết và hướng dẫn thi hành một số điều của Luật Bảo hiểm xã hội về bảo hiểm xã hội tự nguyện ____________ Căn cứ Luật Tổ chức Chính phủ ngày 18 tháng 02 năm 2025; Căn cứ...
- R2: score=`0.0152` doc_id=`41/2024/QH15` path=`Dieu 109 > Khoan 3` type=`ClauseChunk`
  citation: Luật 41/2024/QH15 - LUẬT BẢO HIỂM XÃ HỘI Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam; Quốc hội ban hành Luật Bảo hiểm xã hội. Chương I NHỮNG QUY ĐỊNH CHUNG | Dieu 109 > Khoan 3
- R3: score=`0.0137` doc_id=`04/2012/UBTVQH13` path=`Dieu 4` type=`ProvisionChunk`
  citation: Pháp lệnh 04/2012/UBTVQH13 - PHÁP LỆNH Sửa đổi, bổ sung một số điều của P háp lệnh Ư u đãi người có công với cách mạng _________________________ C ă n c ứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam năm 1992 đã đ ư...

### Q5. Trách nhiệm của Bộ Khoa học và Công nghệ về quản lý và phát triển công nghiệp an ninh được quy định như thế nào?
- status: `ok`
- retrieval_time_ms: `3212.15`
- rerank_time_ms: `1088.56`
- used_reranker: `True`
- generation_time_ms: `30606.53`
- error: ``
- R1: score=`0.9922` doc_id=`25/2014/NĐ-CP` path=`Dieu 24` type=`ProvisionChunk`
  citation: Nghị định 25/2014/NĐ-CP - NGHỊ ĐỊNH Q uy định về phòng, chống tội phạm và vi phạm pháp luật khác có sử dụng công nghệ cao _____________ Căn cứ Luật Tổ chức Chính phủ ngày 25 tháng 12 năm 2001; Căn cứ Bộ luật Hình sự ngày...
- R2: score=`0.9756` doc_id=`38/2024/QH15` path=`Dieu 80 > Khoan 2` type=`ClauseChunk`
  citation: Luật 38/2024/QH15 - LUẬT CÔNG NGHIỆP QUỐC PHÒNG, AN NINH VÀ ĐỘNG VIÊN CÔNG NGHIỆP Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam; Quốc hội ban hành Luật Công nghiệp quốc phòng, an ninh và động viên công nghiệp....
- R3: score=`0.9565` doc_id=`38/2024/QH15` path=`Dieu 80 > Khoan 3` type=`ClauseChunk`
  citation: Luật 38/2024/QH15 - LUẬT CÔNG NGHIỆP QUỐC PHÒNG, AN NINH VÀ ĐỘNG VIÊN CÔNG NGHIỆP Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam; Quốc hội ban hành Luật Công nghiệp quốc phòng, an ninh và động viên công nghiệp....

### Q6. Đập phá biển báo “khu vực biên giới” bị phạt thế nào?
- status: `ok`
- retrieval_time_ms: `3289.10`
- rerank_time_ms: `1271.81`
- used_reranker: `True`
- generation_time_ms: `26066.69`
- error: ``
- R1: score=`0.0574` doc_id=`100/2015/QH13` path=`Dieu 346 > Khoan 000` type=`ClauseChunk`
  citation: Bộ luật 100/2015/QH13 - BỘ LUẬT HÌNH SỰ __ Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam; Quốc hội ban hành Bộ luật hình sự. Phần thứ nhất NHỮNG QUY ĐỊNH CHUNG Chương I ĐIỀU KHOẢN CƠ BẢN | Dieu 346 > Khoan 000
- R2: score=`0.0548` doc_id=`100/2015/QH13` path=`Dieu 346 > Khoan 000` type=`ClauseChunk`
  citation: Bộ luật 100/2015/QH13 - BỘ LUẬT HÌNH SỰ __ Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam; Quốc hội ban hành Bộ luật hình sự. Phần thứ nhất NHỮNG QUY ĐỊNH CHUNG Chương I ĐIỀU KHOẢN CƠ BẢN | Dieu 346 > Khoan 000
- R3: score=`0.0527` doc_id=`100/2015/QH13` path=`Dieu 346 > Khoan 000` type=`ClauseChunk`
  citation: Bộ luật 100/2015/QH13 - BỘ LUẬT HÌNH SỰ __ Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam; Quốc hội ban hành Bộ luật hình sự. Phần thứ nhất NHỮNG QUY ĐỊNH CHUNG Chương I ĐIỀU KHOẢN CƠ BẢN | Dieu 346 > Khoan 000

### Q7. Mục tiêu phát triển khu công nghiệp hỗ trợ được quy định như thế nào?
- status: `ok`
- retrieval_time_ms: `3258.48`
- rerank_time_ms: `1275.43`
- used_reranker: `True`
- generation_time_ms: `24941.15`
- error: ``
- R1: score=`1.0000` doc_id=`82/2018/NĐ-CP` path=`Dieu 34 > Khoan 2` type=`ClauseChunk`
  citation: Nghị định 82/2018/NĐ-CP - NGHỊ ĐỊNH QUY ĐỊNH VỀ QUẢN LÝ KHU CÔNG NGHIỆP VÀ KHU KINH TẾ Căn cứ Luật tổ chức Chính phủ ngày 19 tháng 6 năm 2015; Căn cứ Luật đầu tư ngày 26 tháng 11 năm 2014; Theo đề nghị của Bộ trưởng Bộ K...
- R2: score=`0.9360` doc_id=`82/2018/NĐ-CP` path=`Dieu 33 > Khoan 2` type=`ClauseChunk`
  citation: Nghị định 82/2018/NĐ-CP - NGHỊ ĐỊNH QUY ĐỊNH VỀ QUẢN LÝ KHU CÔNG NGHIỆP VÀ KHU KINH TẾ Căn cứ Luật tổ chức Chính phủ ngày 19 tháng 6 năm 2015; Căn cứ Luật đầu tư ngày 26 tháng 11 năm 2014; Theo đề nghị của Bộ trưởng Bộ K...
- R3: score=`0.6421` doc_id=`82/2018/NĐ-CP` path=`Dieu 36 > Khoan 3` type=`ClauseChunk`
  citation: Nghị định 82/2018/NĐ-CP - NGHỊ ĐỊNH QUY ĐỊNH VỀ QUẢN LÝ KHU CÔNG NGHIỆP VÀ KHU KINH TẾ Căn cứ Luật tổ chức Chính phủ ngày 19 tháng 6 năm 2015; Căn cứ Luật đầu tư ngày 26 tháng 11 năm 2014; Theo đề nghị của Bộ trưởng Bộ K...

### Q8. Thừa phát lại được tống đạt những giấy tờ, hồ sơ, tài liệu nào?
- status: `ok`
- retrieval_time_ms: `3074.23`
- rerank_time_ms: `1163.23`
- used_reranker: `True`
- generation_time_ms: `38074.63`
- error: ``
- R1: score=`0.1373` doc_id=`02/2022/UBTVQH15` path=`Dieu 19 > Khoan 4` type=`ClauseChunk`
  citation: Pháp lệnh 02/2022/UBTVQH15 - Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam; Căn cứ Luật Xử lý vi phạm hành chính số 15/2012/QH13 đã được sửa đổi, bổ sung theo Luật số 67/2020/QH14; Ủy ban Thường vụ Quốc hội ba...
- R2: score=`0.0070` doc_id=`93/2015/QH13` path=`Dieu 101 > Khoan 5` type=`ClauseChunk`
  citation: Luật 93/2015/QH13 - LUẬT TỐ TỤNG HÀNH CHÍNH __________ Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam; Quốc hội ban hành Luật tố tụng hành chính. Chương I NHỮNG QUY ĐỊNH CHUNG | Dieu 101 > Khoan 5
- R3: score=`0.0057` doc_id=`93/2015/QH13` path=`Dieu 104` type=`ProvisionChunk`
  citation: Luật 93/2015/QH13 - LUẬT TỐ TỤNG HÀNH CHÍNH __________ Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam; Quốc hội ban hành Luật tố tụng hành chính. Chương I NHỮNG QUY ĐỊNH CHUNG | Dieu 104

### Q9. Mức phạt khi sử dụng trái phép lòng đường đô thị, hè phố để rửa xe
- status: `ok`
- retrieval_time_ms: `3339.99`
- rerank_time_ms: `1543.22`
- used_reranker: `True`
- generation_time_ms: `37005.02`
- error: ``
- R1: score=`0.1831` doc_id=`100/2019/NĐ-CP` path=`Dieu 20 > Khoan 000` type=`ClauseChunk`
  citation: Nghị định 100/2019/NĐ-CP - NGHỊ ĐỊNH Quy định xử phạt vi phạm hành chính trong lĩnh vực giao thông đường bộ và đường sắt Căn cứ Luật Tổ chức Chính phủ ngày 19 tháng 6 năm 2015; Căn cứ Luật Xử lý vi phạm hành chính ngày 2...
- R2: score=`0.1824` doc_id=`46/2016/NĐ-CP` path=`Dieu 20 > Khoan 000` type=`ClauseChunk`
  citation: Nghị định 46/2016/NĐ-CP - NGHỊ ĐỊNH Q u y định xử phạt v i phạm hành ch í nh trong lĩnh vực giao thông đường bộ và đường sắt ________________________ Căn cứ Luật Tổ chức Chính phủ ngày 19 tháng 6 năm 20 1 5; Căn cứ Luật...
- R3: score=`0.0470` doc_id=`100/2015/QH13` path=`Dieu 261 > Khoan 500` type=`ClauseChunk`
  citation: Bộ luật 100/2015/QH13 - BỘ LUẬT HÌNH SỰ __ Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam; Quốc hội ban hành Bộ luật hình sự. Phần thứ nhất NHỮNG QUY ĐỊNH CHUNG Chương I ĐIỀU KHOẢN CƠ BẢN | Dieu 261 > Khoan 500

### Q10. Hai công ty con chung công ty mẹ có được tham dự gói thầu của nhau không?
- status: `ok`
- retrieval_time_ms: `3119.81`
- rerank_time_ms: `1214.30`
- used_reranker: `True`
- generation_time_ms: `34277.70`
- error: ``
- R1: score=`0.0196` doc_id=`68/2014/QH13` path=`Dieu 189 > Khoan 2` type=`ClauseChunk`
  citation: Luật 68/2014/QH13 - LUẬT Doanh nghiệp ______ Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam; Quốc hội ban hành Luật Doanh nghiệp. Chương I NHỮNG QUY ĐỊNH CHUNG | Dieu 189 > Khoan 2
- R2: score=`0.0107` doc_id=`35/2021/NĐ-CP` path=`Dieu 38` type=`ProvisionChunk`
  citation: Nghị định 35/2021/NĐ-CP - CHÍNH PHỦ - - - - - - - CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM Độc lập - Tự do - Hạnh phúc - - - - - - - - - - - - - - - Số: 35/2021/NĐ - CP Hà Nội, ngày 29 tháng 3 năm 2021 NGHỊ ĐỊNH QUY ĐỊNH CHI T...
- R3: score=`0.0084` doc_id=`35/2021/NĐ-CP` path=`Dieu 30 > Khoan 3` type=`ClauseChunk`
  citation: Nghị định 35/2021/NĐ-CP - CHÍNH PHỦ - - - - - - - CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM Độc lập - Tự do - Hạnh phúc - - - - - - - - - - - - - - - Số: 35/2021/NĐ - CP Hà Nội, ngày 29 tháng 3 năm 2021 NGHỊ ĐỊNH QUY ĐỊNH CHI T...
