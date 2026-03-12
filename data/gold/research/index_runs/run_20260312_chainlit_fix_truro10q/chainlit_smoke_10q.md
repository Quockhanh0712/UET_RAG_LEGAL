# Chainlit Hybrid Smoke (10 queries)

- run_id: `run_20260312_chainlit_fix_truro10q`
- created_at: `2026-03-12T16:32:38.060024+00:00`
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
- retrieval_p50_ms: `3103.82`
- retrieval_p95_ms: `3597.29`
- generation_p50_ms: `69871.05`
- generation_p95_ms: `104961.55`
- rerank_used_count: `10`
- rerank_p50_ms: `1265.77`
- rerank_p95_ms: `1719.36`

## Gate Checks

- at_least_3_results_each_query: `True`
- top1_citation_non_empty_gte_9: `True`
- llm_non_empty_or_fallback_gte_9: `True`
- reranker_used_when_enabled: `True`

### Q1. Cong an xa xu phat loi khong mang bang lai xe co dung khong?
- status: `ok`
- retrieval_time_ms: `11043.96`
- rerank_time_ms: `6008.38`
- used_reranker: `True`
- generation_time_ms: `59176.08`
- error: ``
- R1: score=`0.0002` doc_id=`48/2014/QH13` path=`Dieu 36 > Khoan 5` type=`ClauseChunk`
  citation: Luật 48/2014/QH13 - LUẬT Sửa đổi, bổ sung một số điều của Luật Giao thông đường thủy nội địa ________________ Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam; Quốc hội ban hành Luật Sửa đổi, bổ sung một số điều...
- R2: score=`0.0000` doc_id=`72/2020/QH14` path=`Dieu 3 > Khoan 33` type=`ClauseChunk`
  citation: Luật 72/2020/QH14 - LUẬT BẢO VỆ MÔI TRƯỜNG ________ Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam; Quốc hội ban hành Luật Bảo vệ môi trường. Chương I NHỮNG QUY ĐỊNH CHUNG | Dieu 3 > Khoan 33
- R3: score=`0.0000` doc_id=`19/2017/QH14` path=`Dieu 2` type=`ProvisionChunk`
  citation: Luật 19/2017/QH14 - LUẬT SỬA ĐỔI, BỔ SUNG MỘT SỐ ĐIỀU CỦA LUẬT CƠ QUAN ĐẠI DIỆN NƯỚC CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM Ở NƯỚC NGOÀI Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam; Quốc hội ban hành Luật sửa đổ...

### Q2. Phai thuc hien thao tac nap mau vao binh chua va xu ly mau so bo bang hoa chat nhu the nao?
- status: `ok`
- retrieval_time_ms: `3135.08`
- rerank_time_ms: `1280.12`
- used_reranker: `True`
- generation_time_ms: `70111.48`
- error: ``
- R1: score=`0.0000` doc_id=`16/2011/UBTVQH12` path=`Dieu 3 > Khoan 5` type=`ClauseChunk`
  citation: Pháp lệnh 16/2011/UBTVQH12 - PHÁP LỆNH Quản lý, sử dụng vũ khí, vật liệu nổ và công cụ hỗ trợ ___________________ Căn cứ Hiến pháp nước Cộng hoà xã hội chủ nghĩa Việt Nam năm 1992 đã được sửa đổi, bổ sung một số điều the...
- R2: score=`0.0000` doc_id=`54/1998/NĐ-CP` path=`Dieu 6` type=`ProvisionChunk`
  citation: Nghị định 54/1998/NĐ-CP - NGHỊ ĐỊNH CỦA CHÍNH PHỦ S ửa đổi, bổ sung một số điều của Nghị định số 16/CP ngày 20 tháng 3 năm 1996 của Chính phủ về xử phạt vi phạm hành chính trong lĩnh vực quản lý Nhà nước về hải quan CHÍN...
- R3: score=`0.0000` doc_id=`04/2012/UBTVQH13` path=`Dieu 10 > Khoan 2` type=`ClauseChunk`
  citation: Pháp lệnh 04/2012/UBTVQH13 - PHÁP LỆNH Sửa đổi, bổ sung một số điều của P háp lệnh Ư u đãi người có công với cách mạng _________________________ C ă n c ứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam năm 1992 đã đ ư...

### Q3. Trinh tu danh gia chat luong hang nam doi voi Kiem soat vien duoc quy dinh nhu the nao?
- status: `ok`
- retrieval_time_ms: `2953.47`
- rerank_time_ms: `1153.71`
- used_reranker: `True`
- generation_time_ms: `61416.77`
- error: ``
- R1: score=`0.0000` doc_id=`59/2022/NĐ-CP` path=`Dieu 3 > Khoan 13` type=`ClauseChunk`
  citation: Nghị định 59/2022/NĐ-CP - NGHỊ ĐỊNH Quy định về định danh và xác thực điện tử Căn cứ Luật T ổ chức Chính phủ ngày 19 tháng 6 năm 2015; Luật sửa đổi, bổ sung một số điều của Luật T ổ c hức Chính phủ và Luật T ổ chức chính...
- R2: score=`0.0000` doc_id=`83/2025/QH15` path=`Dieu 39` type=`ProvisionChunk`
  citation: Luật 83/2025/QH15 - LUẬT Sửa đổi, bổ sung một số điều của Luật Bầu cử đại biểu Quốc hội và đại biểu Hội đồng nhân dân Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam đã được sửa đổi, bổ sung một số điều theo Ngh...
- R3: score=`0.0000` doc_id=`24/2023/QH15` path=`Dieu 3 > Khoan 17` type=`ClauseChunk`
  citation: Luật 24/2023/QH15 - QUỐC HỘI - - - - - - - CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM Độc lập - Tự do - Hạnh phúc - - - - - - - - - - - - - - - Luật số: 24/2023/QH15 Hà Nội, ngày 24 tháng 11 năm 2023 LUẬT VIỄN THÔNG Căn cứ Hiến...

### Q4. Tu tu chet sau khi tiem thuoc doc ma khong co than nhan nhan xac thi duoc ho tro mai tang nhu the nao?
- status: `ok`
- retrieval_time_ms: `3048.09`
- rerank_time_ms: `1213.71`
- used_reranker: `True`
- generation_time_ms: `85971.28`
- error: ``
- R1: score=`0.0001` doc_id=`15/2023/QH15` path=`Dieu 119` type=`ProvisionChunk`
  citation: Luật 15/2023/QH15 - LUẬT KHÁM BỆNH, CHỮA BỆNH Căn cứ Hiến pháp nước Cộng hoà xã hội chủ nghĩa Việt Nam; Quốc hội ban hành Luật Khám bệnh, chữa bệnh. Chương I NHỮNG QUY ĐỊNH CHUNG | Dieu 119
- R2: score=`0.0000` doc_id=`02/2020/TTLT-BCA-BQP-BYT-BNG-TANDTC-VKSNDTC` path=`Dieu 14` type=`ProvisionChunk`
  citation: Thông tư liên tịch 02/2020/TTLT-BCA-BQP-BYT-BNG-TANDTC-VKSNDTC - THÔNG TƯ LIÊN TỊCH Quy định về việc phối hợp tổ chức thi hành án tử hình bằng hình thức tiêm thuốc độc Căn cứ Bộ luật Tố tụng hình sự ngày 27 tháng 11 năm...
- R3: score=`0.0000` doc_id=`09/2021/TT-BTNMT` path=`Dieu 1 > Khoan 9` type=`ClauseChunk`
  citation: Thông tư 09/2021/TT-BTNMT - THÔNG TƯ Sửa đổi, bổ sung một số điều của các Thông tư quy định chi tiết và hướng dẫn thi hành Luật Đất đai Căn cứ Luật Đất đai ngày 29 tháng 11 năm 2013; Căn cứ Luật sửa đổi, bổ sung một số đ...

### Q5. Trach nhiem cua Bo Khoa hoc va Cong nghe ve quan ly va phat trien cong nghiep an ninh duoc quy dinh nhu the nao?
- status: `ok`
- retrieval_time_ms: `3072.56`
- rerank_time_ms: `1251.42`
- used_reranker: `True`
- generation_time_ms: `69630.62`
- error: ``
- R1: score=`0.0004` doc_id=`69/2009/NĐ-CP` path=`Dieu 43` type=`ProvisionChunk`
  citation: Nghị định 69/2009/NĐ-CP - NGHỊ ĐỊNH Quy định bổ sung về quy hoạch sử dụng đất, giá đất, thu hồi đất, bồi thường, hỗ trợ và tái định cư __________________________________________ CHÍNH PHỦ Căn cứ Luật Tổ chức Chính phủ ng...
- R2: score=`0.0002` doc_id=`34/2018/TT-BCA` path=`Dieu 260` type=`ProvisionChunk`
  citation: Thông tư 34/2018/TT-BCA - THÔNG TƯ Quy định trình tự điều tra, giải quyết tai nạn giao thông đường sắt của lực lượng Cảnh sát giao thông Căn cứ Bộ luật Hình sự năm 2015 (sửa đổi, bổ sung năm 2017); Căn cứ Bộ luật Tố tụng...
- R3: score=`0.0000` doc_id=`85/2025/QH15` path=`Dieu 504 > Khoan 33` type=`ClauseChunk`
  citation: Luật 85/2025/QH15 - QUỐC HỘI –––––––––– Luật số: 85/2025/QH15 CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM Độc lập - Tự do - Hạnh phúc –––––––––––––––––––––––––––––––––––– LUẬT SỬA ĐỔI, BỔ SUNG MỘT SỐ ĐIỀU CỦA BỘ LUẬT TỐ TỤNG DÂN...

### Q6. Dap pha bien bao khu vuc bien gioi bi phat the nao?
- status: `ok`
- retrieval_time_ms: `3597.29`
- rerank_time_ms: `1513.86`
- used_reranker: `True`
- generation_time_ms: `64968.44`
- error: ``
- R1: score=`0.0001` doc_id=`79/2015/QH13` path=`Dieu 3 > Khoan 12` type=`ClauseChunk`
  citation: Luật 79/2015/QH13 - LUẬT Thú y _ Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam; Quốc hội ban hành Luật thú y. Chương I NHỮNG QUY ĐỊNH CHUNG | Dieu 3 > Khoan 12
- R2: score=`0.0001` doc_id=`79/2015/QH13` path=`Dieu 3 > Khoan 3` type=`ClauseChunk`
  citation: Luật 79/2015/QH13 - LUẬT Thú y _ Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam; Quốc hội ban hành Luật thú y. Chương I NHỮNG QUY ĐỊNH CHUNG | Dieu 3 > Khoan 3
- R3: score=`0.0000` doc_id=`09/2012/QH13` path=`Dieu 24` type=`ProvisionChunk`
  citation: Luật 09/2012/QH13 - LUẬT Phòng, chống tác hại của thuốc lá _______________ Căn cứ Hiến pháp nước Cộng hoà xã hội chủ nghĩa Việt Nam năm 1992 đã được sửa đổi, bổ sung một số điều theo Nghị quyết số 51/2001/QH10; Quốc hội...

### Q7. Muc tieu phat trien khu cong nghiep ho tro duoc quy dinh nhu the nao?
- status: `ok`
- retrieval_time_ms: `3580.38`
- rerank_time_ms: `1719.36`
- used_reranker: `True`
- generation_time_ms: `81583.27`
- error: ``
- R1: score=`0.0001` doc_id=`06/2017/QH14` path=`Dieu 3 > Khoan 19` type=`ClauseChunk`
  citation: Luật 06/2017/QH14 - Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam; Quốc hội ban hành Luật Đường sắt. Chương I NHỮNG QUY ĐỊNH CHUNG | Dieu 3 > Khoan 19
- R2: score=`0.0000` doc_id=`48/2014/QH13` path=`Dieu 36 > Khoan 5` type=`ClauseChunk`
  citation: Luật 48/2014/QH13 - LUẬT Sửa đổi, bổ sung một số điều của Luật Giao thông đường thủy nội địa ________________ Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam; Quốc hội ban hành Luật Sửa đổi, bổ sung một số điều...
- R3: score=`0.0000` doc_id=`64/2010/QH12` path=`Dieu 41 > Khoan 8` type=`ClauseChunk`
  citation: Luật 64/2010/QH12 - LUẬT Tố tụng hành chính ___________ C ă n c ứ H i ế n phá p nư ớ c C ộn g ho à x ã h ộ i ch ủ n g h ĩ a Việ t Na m n ă m 1992 đ ã đ ược s ửa đ ổi, b ổ su n g m ột s ố điều theo N g hị quyết s ố 5 1/20...

### Q8. Thua phat lai duoc tong dat nhung giay to ho so tai lieu nao?
- status: `ok`
- retrieval_time_ms: `2944.70`
- rerank_time_ms: `1068.48`
- used_reranker: `True`
- generation_time_ms: `64566.14`
- error: ``
- R1: score=`0.0017` doc_id=`15/1999/QH10` path=`Dieu 248` type=`ProvisionChunk`
  citation: Bộ luật 15/1999/QH10 - BỘ LUẬT HÌNH SỰ CỦA NƯỚC CỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM ______________________ LỜI NÓI ĐẦU Pháp luật hình sự là một trong những công cụ sắc bén, hữu hiệu để đấu tranh phòng ngừa và chống tội ph...
- R2: score=`0.0011` doc_id=`23/2018/TT-BGTVT` path=`Dieu 38` type=`ProvisionChunk`
  citation: Thông tư 23/2018/TT-BGTVT - THÔNG TƯ Quy định về giải quyết sự cố, tai nạn giao thông đường sắt và phân tích, thống kê báo cáo về sự cố, tai nạn giao thông đường sắt ________________________ Căn cứ Bộ luật Dân sự ngày 24...
- R3: score=`0.0001` doc_id=`31/2018/QH14` path=`Dieu 29` type=`ProvisionChunk`
  citation: Luật 31/2018/QH14 - LUẬT TRỒNG TRỌT Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam; Quốc hội ban hành Luật Trồng trọt. Chương I NHỮNG QUY ĐỊNH CHUNG | Dieu 29

### Q9. Muc phat khi su dung trai phep long duong do thi he pho de rua xe
- status: `ok`
- retrieval_time_ms: `3151.00`
- rerank_time_ms: `1314.62`
- used_reranker: `True`
- generation_time_ms: `105484.16`
- error: ``
- R1: score=`0.0001` doc_id=`17/2007/TTLT-BCA-VKSNDTC-TANDTC-BTP` path=`Dieu 197 > Khoan 7` type=`ClauseChunk`
  citation: Thông tư liên tịch 17/2007/TTLT-BCA-VKSNDTC-TANDTC-BTP - THÔNG TƯ LIÊN TỊCH Hướng dẫn áp dụng một số quy định tại Chương XVIII “Các tội phạm về ma túy” của Bộ luật Hình sự năm 1999 ____________________________ Để áp dụng...
- R2: score=`0.0001` doc_id=`17/2007/TTLT-BCA-VKSNDTC-TANDTC-BTP` path=`Dieu 197 > Khoan 7` type=`ClauseChunk`
  citation: Thông tư liên tịch 17/2007/TTLT-BCA-VKSNDTC-TANDTC-BTP - THÔNG TƯ LIÊN TỊCH Hướng dẫn áp dụng một số quy định tại Chương XVIII “Các tội phạm về ma túy” của Bộ luật Hình sự năm 1999 ____________________________ Để áp dụng...
- R3: score=`0.0001` doc_id=`13/2022/QH15` path=`Dieu 3 > Khoan 7` type=`ClauseChunk`
  citation: Luật 13/2022/QH15 - LUẬT PHÒNG, CHỐNG BẠO LỰC GIA ĐÌNH Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam; Quốc hội ban hành Luật Phòng, chống bạo lực gia đình. Chương I NHỮNG QUY ĐỊNH CHUNG | Dieu 3 > Khoan 7

### Q10. Hai cong ty con chung cong ty me co duoc tham du goi thau cua nhau khong?
- status: `ok`
- retrieval_time_ms: `3010.04`
- rerank_time_ms: `1141.07`
- used_reranker: `True`
- generation_time_ms: `104961.55`
- error: ``
- R1: score=`0.0004` doc_id=`139/2007/NĐ-CP` path=`Dieu 4 > Khoan 4` type=`ClauseChunk`
  citation: Nghị định 139/2007/NĐ-CP - NGHỊ ĐỊNH Hướng dẫn chi tiết thi hành một số điều của Luật doanh nghiệp ______________________________ CHÍNH PHỦ Căn cứ Luật Tổ chức Chính phủ ngày 25 tháng 12 năm 2001; Căn cứ Luật Doanh nghiệ...
- R2: score=`0.0003` doc_id=`102/2010/NĐ-CP` path=`Dieu 4 > Khoan 4` type=`ClauseChunk`
  citation: Nghị định 102/2010/NĐ-CP - NGHỊ ĐỊNH Hướng dẫn chi tiết thi hành một số điều của Luật Doanh nghiệp ______________________________ CHÍNH PHỦ Căn cứ Luật Tổ chức Chính phủ ngày 25 tháng 12 năm 2001; Căn cứ Luật Doanh nghiệ...
- R3: score=`0.0001` doc_id=`52/2014/QH13` path=`Dieu 14` type=`ProvisionChunk`
  citation: Luật 52/2014/QH13 - LUẬT Hôn nhân và gia đình _______ Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam; Quốc hội ban hành Luật Hôn nhân và gia đình. CHƯƠNG I NHỮNG QUY ĐỊNH CHUNG | Dieu 14
