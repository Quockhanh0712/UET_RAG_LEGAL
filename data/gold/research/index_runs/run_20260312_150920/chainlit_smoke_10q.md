# Chainlit Hybrid Smoke (10 queries)

- run_id: `run_20260312_150920`
- created_at: `2026-03-12T15:13:55.099437+00:00`
- model: `truro7/vn-law-embedding`
- collection: `legal_documents__truro7_vn_law_embedding__d768`
- top_k: `5`
- use_reranker: `True`
- use_graph: `False`
- require_active_only: `True`

## Summary

- query_total: `10`
- status_ok: `10`
- fallback_count: `0`
- hard_fail_count: `0`
- retrieval_p50_ms: `3389.72`
- retrieval_p95_ms: `3738.75`
- generation_p50_ms: `18919.63`
- generation_p95_ms: `29184.62`

## Gate Checks

- at_least_3_results_each_query: `True`
- top1_citation_non_empty_gte_9: `True`
- llm_non_empty_or_fallback_gte_9: `True`

### Q1. Cong an xa xu phat loi khong mang bang lai xe co dung khong?
- status: `ok`
- retrieval_time_ms: `9843.49`
- generation_time_ms: `20126.57`
- error: ``
- R1: score=`0.2001` doc_id=`64/2010/QH12` path=`Dieu 28 > Khoan 2` type=`ClauseChunk`
  citation: Luật 64/2010/QH12 - LUẬT Tố tụng hành chính ___________ C ă n c ứ H i ế n phá p nư ớ c C ộn g ho à x ã h ộ i ch ủ n g h ĩ a Việ t Na m n ă m 1992 đ ã đ ược s ửa đ ổi, b ổ su n g m ột s ố điều theo N g hị quyết s ố 5 1/20...
- R2: score=`0.2000` doc_id=`07-TT/LBTL-ĐC` path=`Dieu 23 > Khoan 1` type=`ClauseChunk`
  citation: Thông tư liên tịch 07-TT/LBTL-ĐC - THÔNG TƯ LIÊN BỘ THỦY LỢI - TỔNG CỤC ĐỊA CHÍNH Hướng dẫn thủ tục lập hồ sơ giao đất để xây dựng công trình thuỷ lợi Căn cứ Luật đất đai ngày 14 - 7 - 1993. Căn cứ Nghị định số 177/CP ng...
- R3: score=`0.1336` doc_id=`62/2010/QH12` path=`Dieu 3` type=`ProvisionChunk`
  citation: Luật 62/2010/QH12 - LUẬT Sửa đổi, bổ sung một số điều của Luật Chứng khoán __________________________ C ăn c ứ H i ế n pháp nư ớ c C ộng h oà xã h ội c h ủ n g h ĩ a Việt Na m n ă m 1992 đã đ ược s ử a đổi, b ổ su n g m...

### Q2. Phai thuc hien thao tac nap mau vao binh chua va xu ly mau so bo bang hoa chat nhu the nao?
- status: `ok`
- retrieval_time_ms: `3377.54`
- generation_time_ms: `16920.49`
- error: ``
- R1: score=`0.2000` doc_id=`57/2020/QH14` path=`Dieu 28 > Khoan 4` type=`ClauseChunk`
  citation: Luật 57/2020/QH14 - LUẬT THANH NIÊN Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam; Quốc hội ban hành Luật Thanh niên. Chương I NHỮNG QUY ĐỊNH CHUNG | Dieu 28 > Khoan 4
- R2: score=`0.2000` doc_id=`07-TT/LBTL-ĐC` path=`Dieu 23 > Khoan 1` type=`ClauseChunk`
  citation: Thông tư liên tịch 07-TT/LBTL-ĐC - THÔNG TƯ LIÊN BỘ THỦY LỢI - TỔNG CỤC ĐỊA CHÍNH Hướng dẫn thủ tục lập hồ sơ giao đất để xây dựng công trình thuỷ lợi Căn cứ Luật đất đai ngày 14 - 7 - 1993. Căn cứ Nghị định số 177/CP ng...
- R3: score=`0.1334` doc_id=`22/2018/QH14` path=`Dieu 21 > Khoan 7` type=`ClauseChunk`
  citation: Luật 22/2018/QH14 - LUẬT QUỐC PHÒNG ____ Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam; Quốc hội ban hành Luật Quốc phòng. Chương I NHỮNG QUY ĐỊNH CHUNG | Dieu 21 > Khoan 7

### Q3. Trinh tu danh gia chat luong hang nam doi voi Kiem soat vien duoc quy dinh nhu the nao?
- status: `ok`
- retrieval_time_ms: `3200.18`
- generation_time_ms: `14195.54`
- error: ``
- R1: score=`0.2000` doc_id=`07-TT/LBTL-ĐC` path=`Dieu 23 > Khoan 1` type=`ClauseChunk`
  citation: Thông tư liên tịch 07-TT/LBTL-ĐC - THÔNG TƯ LIÊN BỘ THỦY LỢI - TỔNG CỤC ĐỊA CHÍNH Hướng dẫn thủ tục lập hồ sơ giao đất để xây dựng công trình thuỷ lợi Căn cứ Luật đất đai ngày 14 - 7 - 1993. Căn cứ Nghị định số 177/CP ng...
- R2: score=`0.2000` doc_id=`19/2025/TT-NHNN` path=`Dieu 6 > Khoan 11` type=`ClauseChunk`
  citation: Thông tư 19/2025/TT-NHNN - THÔNG TƯ Quy định về mạng lưới hoạt động của tổ chức tài chính vi mô Căn cứ Luật Ngân hàng Nhà nước Việt Nam số 46/2010/QH12; Căn cứ Luật Các tổ chức tín dụng số 32/2024/QH15; Căn cứ Luật Doanh...
- R3: score=`0.1334` doc_id=`59/2022/NĐ-CP` path=`Dieu 9 > Khoan 5` type=`ClauseChunk`
  citation: Nghị định 59/2022/NĐ-CP - NGHỊ ĐỊNH Quy định về định danh và xác thực điện tử Căn cứ Luật T ổ chức Chính phủ ngày 19 tháng 6 năm 2015; Luật sửa đổi, bổ sung một số điều của Luật T ổ c hức Chính phủ và Luật T ổ chức chính...

### Q4. Tu tu chet sau khi tiem thuoc doc ma khong co than nhan nhan xac thi duoc ho tro mai tang nhu the nao?
- status: `ok`
- retrieval_time_ms: `3494.51`
- generation_time_ms: `24437.81`
- error: ``
- R1: score=`0.2001` doc_id=`02/2016/QH14` path=`Dieu 6 > Khoan 3` type=`ClauseChunk`
  citation: Luật 02/2016/QH14 - LUẬT TÍN NGƯỠNG, TÔN GIÁO Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Vi ệt Nam; Quốc hội ban hành Luật tín ngưỡng, tôn giáo. Chương I NHỮNG QUY ĐỊNH CHUNG | Dieu 6 > Khoan 3
- R2: score=`0.2000` doc_id=`43/2016/TT-BCA` path=`Dieu 2` type=`ProvisionChunk`
  citation: Thông tư 43/2016/TT-BCA - THÔNG TƯ Quy định biện pháp thực hiện bảo hiểm y tế đối với sĩ quan, hạ sĩ quan, chiến sĩ Công an nhân dân Căn cứ Luật Bảo hiểm y tế ngày 14 tháng 11 năm 2008 và Luật sửa đổi, bổ sung một số điề...
- R3: score=`0.1360` doc_id=`64/2010/QH12` path=`Dieu 3 > Khoan 1` type=`ClauseChunk`
  citation: Luật 64/2010/QH12 - LUẬT Tố tụng hành chính ___________ C ă n c ứ H i ế n phá p nư ớ c C ộn g ho à x ã h ộ i ch ủ n g h ĩ a Việ t Na m n ă m 1992 đ ã đ ược s ửa đ ổi, b ổ su n g m ột s ố điều theo N g hị quyết s ố 5 1/20...

### Q5. Trach nhiem cua Bo Khoa hoc va Cong nghe ve quan ly va phat trien cong nghiep an ninh duoc quy dinh nhu the nao?
- status: `ok`
- retrieval_time_ms: `3401.90`
- generation_time_ms: `17712.70`
- error: ``
- R1: score=`0.2045` doc_id=`64/2010/QH12` path=`Dieu 29 > Khoan 3` type=`ClauseChunk`
  citation: Luật 64/2010/QH12 - LUẬT Tố tụng hành chính ___________ C ă n c ứ H i ế n phá p nư ớ c C ộn g ho à x ã h ộ i ch ủ n g h ĩ a Việ t Na m n ă m 1992 đ ã đ ược s ửa đ ổi, b ổ su n g m ột s ố điều theo N g hị quyết s ố 5 1/20...
- R2: score=`0.2000` doc_id=`38/2016/TT-BGTVT` path=`Dieu 3 > Khoan 1` type=`ClauseChunk`
  citation: Thông tư 38/2016/TT-BGTVT - THÔNG TƯ Q uy định về giải quyết sự cố, tai nạn giao thông đường sắ t _____________ Căn cứ Bộ luật Dân sự ngày 14 tháng 6 năm 2005; Căn cứ Luật Đường sắt ngày 14 tháng 6 năm 2005; Căn cứ Nghị...
- R3: score=`0.1334` doc_id=`64/2010/QH12` path=`Dieu 28 > Khoan 2` type=`ClauseChunk`
  citation: Luật 64/2010/QH12 - LUẬT Tố tụng hành chính ___________ C ă n c ứ H i ế n phá p nư ớ c C ộn g ho à x ã h ộ i ch ủ n g h ĩ a Việ t Na m n ă m 1992 đ ã đ ược s ửa đ ổi, b ổ su n g m ột s ố điều theo N g hị quyết s ố 5 1/20...

### Q6. Dap pha bien bao khu vuc bien gioi bi phat the nao?
- status: `ok`
- retrieval_time_ms: `3490.94`
- generation_time_ms: `17410.20`
- error: ``
- R1: score=`0.2001` doc_id=`32/2024/TT-NHNN` path=`Dieu 3 > Khoan 11` type=`ClauseChunk`
  citation: Thông tư 32/2024/TT-NHNN - THÔNG TƯ Quy định về mạng lưới hoạt động của ngân hàng thương mại Căn cứ Luật Ngân hàng Nhà nước Việt Nam ngày 16 tháng 6 năm 2010; Căn cứ Luật C ác tổ chức tín dụng ngày 18 tháng 01 năm 20 24;...
- R2: score=`0.2000` doc_id=`07-TT/LBTL-ĐC` path=`Dieu 23 > Khoan 1` type=`ClauseChunk`
  citation: Thông tư liên tịch 07-TT/LBTL-ĐC - THÔNG TƯ LIÊN BỘ THỦY LỢI - TỔNG CỤC ĐỊA CHÍNH Hướng dẫn thủ tục lập hồ sơ giao đất để xây dựng công trình thuỷ lợi Căn cứ Luật đất đai ngày 14 - 7 - 1993. Căn cứ Nghị định số 177/CP ng...
- R3: score=`0.1334` doc_id=`83/2016/TT-BTC` path=`Dieu 3 > Khoan 2` type=`ClauseChunk`
  citation: Thông tư 83/2016/TT-BTC - THÔNG TƯ HƯỚNG DẪN THỰC HIỆN ƯU ĐÃI ĐẦU TƯ THEO QUY ĐỊNH CỦA LUẬT ĐẦU TƯ VÀ NGHỊ ĐỊNH SỐ 118/2015/N%C4%90 - CP&area=2&type=0&match=False&vc=True&lan=1" target="_blank"> 118/2015/NĐ - CP NGÀY 12/...

### Q7. Muc tieu phat trien khu cong nghiep ho tro duoc quy dinh nhu the nao?
- status: `ok`
- retrieval_time_ms: `3738.75`
- generation_time_ms: `16486.15`
- error: ``
- R1: score=`0.2002` doc_id=`07-TT/LBTL-ĐC` path=`Dieu 23 > Khoan 1` type=`ClauseChunk`
  citation: Thông tư liên tịch 07-TT/LBTL-ĐC - THÔNG TƯ LIÊN BỘ THỦY LỢI - TỔNG CỤC ĐỊA CHÍNH Hướng dẫn thủ tục lập hồ sơ giao đất để xây dựng công trình thuỷ lợi Căn cứ Luật đất đai ngày 14 - 7 - 1993. Căn cứ Nghị định số 177/CP ng...
- R2: score=`0.2000` doc_id=`32/2024/TT-NHNN` path=`Dieu 3 > Khoan 11` type=`ClauseChunk`
  citation: Thông tư 32/2024/TT-NHNN - THÔNG TƯ Quy định về mạng lưới hoạt động của ngân hàng thương mại Căn cứ Luật Ngân hàng Nhà nước Việt Nam ngày 16 tháng 6 năm 2010; Căn cứ Luật C ác tổ chức tín dụng ngày 18 tháng 01 năm 20 24;...
- R3: score=`0.1335` doc_id=`60/2010/QH12` path=`Dieu 14 > Khoan 2` type=`ClauseChunk`
  citation: Luật 60/2010/QH12 - LUẬT Khoáng sản ______ C ăn c ứ H i ế n pháp nư ớ c C ộng h oà xã h ội c h ủ n g h ĩ a Việt Na m n ă m 1992 đã đ ược s ử a đổi, b ổ su n g m ột s ố điều theo N g hị quyết s ố 5 1/2001/ Q H 10; Quốc hộ...

### Q8. Thua phat lai duoc tong dat nhung giay to ho so tai lieu nao?
- status: `ok`
- retrieval_time_ms: `3262.55`
- generation_time_ms: `35620.24`
- error: ``
- R1: score=`0.2006` doc_id=`05/2024/TT-BGTVT` path=`Dieu 12 > Khoan 7` type=`ClauseChunk`
  citation: Thông tư 05/2024/TT-BGTVT - THÔNG TƯ Sửa đổi, bổ sung một số điều của các Thông tư liên quan đến lĩnh vực vận tải đường bộ, dịch vụ hỗ trợ vận tải đường bộ, phương tiện và người lái _________________ Căn cứ Luật Giao thô...
- R2: score=`0.2000` doc_id=`09/2022/QH15` path=`Dieu 40` type=`ProvisionChunk`
  citation: Luật 09/2022/QH15 - LUẬT SỬA ĐỔI, BỔ SUNG MỘT SỐ ĐIỀU CỦA LUẬT TẦN SỐ VÔ TUYẾN ĐIỆN\ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam; Quốc hội ba...
- R3: score=`0.1343` doc_id=`05/2024/TT-BGTVT` path=`Dieu 1 > Khoan 8` type=`ClauseChunk`
  citation: Thông tư 05/2024/TT-BGTVT - THÔNG TƯ Sửa đổi, bổ sung một số điều của các Thông tư liên quan đến lĩnh vực vận tải đường bộ, dịch vụ hỗ trợ vận tải đường bộ, phương tiện và người lái _________________ Căn cứ Luật Giao thô...

### Q9. Muc phat khi su dung trai phep long duong do thi he pho de rua xe
- status: `ok`
- retrieval_time_ms: `3088.12`
- generation_time_ms: `29153.30`
- error: ``
- R1: score=`0.2005` doc_id=`48/2014/QH13` path=`Dieu 47` type=`ProvisionChunk`
  citation: Luật 48/2014/QH13 - LUẬT Sửa đổi, bổ sung một số điều của Luật Giao thông đường thủy nội địa ________________ Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam; Quốc hội ban hành Luật Sửa đổi, bổ sung một số điều...
- R2: score=`0.2000` doc_id=`65/2013/NĐ-CP` path=`Dieu 5 > Khoan 2` type=`ClauseChunk`
  citation: Nghị định 65/2013/NĐ-CP - NGHỊ ĐỊNH Quy định chi tiết một số điều của Luật Thuế thu nhập cá nhân và Luật Sửa đổi, bổ sung một số điều của Luật Thuế thu nhập cá nhâ n ____________________ Căn cứ Luật Tổ chức Chính phủ ngà...
- R3: score=`0.1339` doc_id=`26/2001/QH10` path=`Dieu 33 > Khoan 3` type=`ClauseChunk`
  citation: Luật 26/2001/QH10 - LUẬT GIAO THÔNG ĐƯỜNG BỘ Để tăng cường hiệu lực quản lý Nhà nước, đề cao ý thức trách nhiệm của cơ quan, tổ chức, cá nhân nhằm bảo đảm giao thông đường bộ thông suốt, trật tự, an toàn, thuận lợi, phục...

### Q10. Hai cong ty con chung cong ty me co duoc tham du goi thau cua nhau khong?
- status: `ok`
- retrieval_time_ms: `3298.23`
- generation_time_ms: `29184.62`
- error: ``
- R1: score=`0.2931` doc_id=`47/2021/NĐ-CP` path=`Dieu 13 > Khoan 3` type=`ClauseChunk`
  citation: Nghị định 47/2021/NĐ-CP - CHÍNH PHỦ 47/2021/NĐ - CP CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM Độc Lập - Tự Do - Hạnh Phúc. . . . . . . . . . . . Hà Nội, ngày 01 tháng 04 năm 20247/2021/NĐ - CP NGHỊ ĐỊNH QUY ĐỊNH CHI TIẾT MỘT SỐ...
- R2: score=`0.2780` doc_id=`68/2014/QH13` path=`Dieu 189 > Khoan 2` type=`ClauseChunk`
  citation: Luật 68/2014/QH13 - LUẬT Doanh nghiệp ______ Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam; Quốc hội ban hành Luật Doanh nghiệp. Chương I NHỮNG QUY ĐỊNH CHUNG | Dieu 189 > Khoan 2
- R3: score=`0.2594` doc_id=`69/2014/NĐ-CP` path=`Dieu 28 > Khoan 2` type=`ClauseChunk`
  citation: Nghị định 69/2014/NĐ-CP - NGHỊ ĐỊNH Về tập đoàn kinh tế nhà nước và tổng công ty nhà nướ c ________________ Căn cứ Luật Tổ chức Chính phủ ngày 25 tháng 12 năm 2001; Căn cứ Luật Doanh nghiệp ngày 29 tháng 11 năm 2005; The...
