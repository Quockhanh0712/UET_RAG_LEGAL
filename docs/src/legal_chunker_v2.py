"""
Legal Chunker v2 - Chunking theo cấu trúc pháp lý Việt Nam

Nguyên tắc:
- 1 Điều = 1 chunk gốc (nếu ngắn)
- Nếu Điều dài: chia theo Khoản/Điểm
- Mỗi chunk giữ đầy đủ metadata pháp lý
- Target: 300-700 tokens
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Token estimation: 1 Vietnamese word ≈ 1.5 tokens
CHARS_PER_TOKEN = 4  # Average for Vietnamese
TARGET_MIN_TOKENS = 300
TARGET_MAX_TOKENS = 700
TARGET_MIN_CHARS = TARGET_MIN_TOKENS * CHARS_PER_TOKEN  # 1200
TARGET_MAX_CHARS = TARGET_MAX_TOKENS * CHARS_PER_TOKEN  # 2800


@dataclass
class LegalChunk:
    """Represents a legal document chunk with full metadata"""
    chunk_id: str
    content: str
    
    # Legal metadata
    van_ban_id: str
    ten_van_ban: str
    loai_van_ban: str
    co_quan: str
    chuong: str
    ten_chuong: str
    dieu_so: str
    tieu_de_dieu: str
    
    # Chunk metadata
    parent_dieu: str  # Reference to parent article
    khoan_so: Optional[str] = None  # If chunked by Khoản
    diem_so: Optional[str] = None  # If chunked by Điểm
    chunk_index: int = 0  # Index within the article
    total_chunks: int = 1  # Total chunks for this article
    
    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "van_ban_id": self.van_ban_id,
            "ten_van_ban": self.ten_van_ban,
            "loai_van_ban": self.loai_van_ban,
            "co_quan": self.co_quan,
            "chuong": self.chuong,
            "ten_chuong": self.ten_chuong,
            "dieu_so": self.dieu_so,
            "tieu_de_dieu": self.tieu_de_dieu,
            "parent_dieu": self.parent_dieu,
            "khoan_so": self.khoan_so or "",
            "diem_so": self.diem_so or "",
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks
        }


class LegalChunkerV2:
    """
    Legal Document Chunker following Vietnamese legal structure:
    Điều → Khoản → Điểm
    """
    
    # Patterns for legal structure
    KHOAN_PATTERN = re.compile(
        r'^(\d+)\.\s+(.+?)(?=(?:^\d+\.\s+|\Z))',
        re.MULTILINE | re.DOTALL
    )
    DIEM_PATTERN = re.compile(
        r'^([a-zđ])\)\s+(.+?)(?=(?:^[a-zđ]\)\s+|\Z))',
        re.MULTILINE | re.DOTALL
    )
    
    def __init__(
        self,
        min_chars: int = TARGET_MIN_CHARS,
        max_chars: int = TARGET_MAX_CHARS,
        overlap_chars: int = 100
    ):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
    
    def normalize_text(self, text: str) -> str:
        """Normalize Vietnamese text"""
        if not text:
            return ""
        # Unicode normalization (NFC for Vietnamese)
        text = unicodedata.normalize('NFC', text)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for Vietnamese text"""
        return len(text) // CHARS_PER_TOKEN
    
    def parse_khoan(self, text: str) -> List[Tuple[str, str]]:
        """Parse Khoản (numbered paragraphs) from text"""
        khoans = []
        
        # Split by Khoản pattern: "1. ", "2. ", etc.
        parts = re.split(r'(?=^\d+\.\s+)', text, flags=re.MULTILINE)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            match = re.match(r'^(\d+)\.\s+(.+)$', part, re.DOTALL)
            if match:
                khoan_so = match.group(1)
                content = match.group(2).strip()
                khoans.append((khoan_so, content))
            elif khoans:
                # Append to previous Khoản if no number
                prev_khoan, prev_content = khoans[-1]
                khoans[-1] = (prev_khoan, prev_content + " " + part)
            else:
                # Content before first Khoản
                khoans.append(("0", part))
        
        return khoans
    
    def parse_diem(self, text: str) -> List[Tuple[str, str]]:
        """Parse Điểm (lettered items) from text"""
        diems = []
        
        # Split by Điểm pattern: "a) ", "b) ", etc.
        parts = re.split(r'(?=^[a-zđ]\)\s+)', text, flags=re.MULTILINE)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            match = re.match(r'^([a-zđ])\)\s+(.+)$', part, re.DOTALL)
            if match:
                diem_so = match.group(1)
                content = match.group(2).strip()
                diems.append((diem_so, content))
            elif diems:
                # Append to previous Điểm
                prev_diem, prev_content = diems[-1]
                diems[-1] = (prev_diem, prev_content + " " + part)
            else:
                # Content before first Điểm
                diems.append(("", part))
        
        return diems
    
    def chunk_article(
        self,
        row: Dict,
        row_index: int
    ) -> List[LegalChunk]:
        """
        Chunk a single article (Điều) into legal chunks.
        
        Strategy:
        1. If article is short enough → 1 chunk
        2. If article has Khoản → chunk by Khoản
        3. If Khoản is still too long → chunk by Điểm
        4. If still too long → split with overlap
        """
        chunks = []
        
        # Get article content
        content = self.normalize_text(row.get('clean_text', '') or row.get('noi_dung', ''))
        if not content:
            return chunks
        
        # Build metadata
        van_ban_id = str(row.get('van_ban_id', ''))
        dieu_so = str(row.get('dieu_so', ''))
        parent_dieu = f"{van_ban_id}_Điều_{dieu_so}"
        
        base_metadata = {
            'van_ban_id': van_ban_id,
            'ten_van_ban': str(row.get('ten_van_ban', '')),
            'loai_van_ban': str(row.get('loai_van_ban', '')),
            'co_quan': str(row.get('co_quan', '')),
            'chuong': str(row.get('chuong', '')),
            'ten_chuong': str(row.get('ten_chuong', '')),
            'dieu_so': dieu_so,
            'tieu_de_dieu': str(row.get('tieu_de_dieu', '')),
            'parent_dieu': parent_dieu
        }
        
        # Strategy 1: Article is short enough → single chunk
        if len(content) <= self.max_chars:
            chunk_id = f"{van_ban_id}_D{dieu_so}_{row_index}_0"
            chunks.append(LegalChunk(
                chunk_id=chunk_id,
                content=self._build_chunk_content(base_metadata, content),
                chunk_index=0,
                total_chunks=1,
                **base_metadata
            ))
            return chunks
        
        # Strategy 2: Chunk by Khoản
        khoans = self.parse_khoan(content)
        
        if len(khoans) > 1:
            # Multiple Khoản found
            chunk_index = 0
            
            for khoan_so, khoan_content in khoans:
                if len(khoan_content) <= self.max_chars:
                    # Khoản fits in one chunk
                    chunk_id = f"{van_ban_id}_D{dieu_so}_K{khoan_so}_{row_index}_{chunk_index}"
                    context = f"Điều {dieu_so}, Khoản {khoan_so}" if khoan_so != "0" else f"Điều {dieu_so}"
                    
                    chunks.append(LegalChunk(
                        chunk_id=chunk_id,
                        content=self._build_chunk_content(base_metadata, khoan_content, context),
                        khoan_so=khoan_so if khoan_so != "0" else None,
                        chunk_index=chunk_index,
                        **base_metadata
                    ))
                    chunk_index += 1
                else:
                    # Khoản too long → try to split by Điểm or hard split
                    sub_chunks = self._split_khoan(
                        khoan_content, khoan_so, base_metadata, 
                        van_ban_id, dieu_so, row_index, chunk_index
                    )
                    chunks.extend(sub_chunks)
                    chunk_index += len(sub_chunks)
        else:
            # No clear Khoản structure → hard split with overlap
            sub_chunks = self._hard_split(
                content, base_metadata, van_ban_id, dieu_so, row_index
            )
            chunks.extend(sub_chunks)
        
        # Update total_chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total
        
        return chunks
    
    def _split_khoan(
        self,
        khoan_content: str,
        khoan_so: str,
        base_metadata: Dict,
        van_ban_id: str,
        dieu_so: str,
        row_index: int,
        start_index: int
    ) -> List[LegalChunk]:
        """Split a Khoản that is too long"""
        chunks = []
        
        # Try to split by Điểm first
        diems = self.parse_diem(khoan_content)
        
        if len(diems) > 1:
            chunk_index = start_index
            current_content = ""
            current_diems = []
            
            for diem_so, diem_content in diems:
                if len(current_content) + len(diem_content) <= self.max_chars:
                    current_content += f" {diem_so}) {diem_content}" if diem_so else f" {diem_content}"
                    current_diems.append(diem_so)
                else:
                    # Save current chunk
                    if current_content:
                        chunk_id = f"{van_ban_id}_D{dieu_so}_K{khoan_so}_{''.join(current_diems)}_{row_index}_{chunk_index}"
                        context = f"Điều {dieu_so}, Khoản {khoan_so}"
                        
                        chunks.append(LegalChunk(
                            chunk_id=chunk_id,
                            content=self._build_chunk_content(base_metadata, current_content.strip(), context),
                            khoan_so=khoan_so,
                            diem_so=','.join(current_diems),
                            chunk_index=chunk_index,
                            **base_metadata
                        ))
                        chunk_index += 1
                    
                    current_content = f"{diem_so}) {diem_content}" if diem_so else diem_content
                    current_diems = [diem_so] if diem_so else []
            
            # Save last chunk
            if current_content:
                chunk_id = f"{van_ban_id}_D{dieu_so}_K{khoan_so}_{''.join(current_diems)}_{row_index}_{chunk_index}"
                context = f"Điều {dieu_so}, Khoản {khoan_so}"
                
                chunks.append(LegalChunk(
                    chunk_id=chunk_id,
                    content=self._build_chunk_content(base_metadata, current_content.strip(), context),
                    khoan_so=khoan_so,
                    diem_so=','.join(current_diems) if current_diems else None,
                    chunk_index=chunk_index,
                    **base_metadata
                ))
        else:
            # No Điểm structure → hard split
            sub_chunks = self._hard_split(
                khoan_content, base_metadata, van_ban_id, dieu_so, row_index,
                prefix=f"K{khoan_so}", khoan_so=khoan_so
            )
            chunks.extend(sub_chunks)
        
        return chunks
    
    def _hard_split(
        self,
        content: str,
        base_metadata: Dict,
        van_ban_id: str,
        dieu_so: str,
        row_index: int,
        prefix: str = "",
        khoan_so: str = None
    ) -> List[LegalChunk]:
        """Hard split content with overlap when no legal structure found"""
        chunks = []
        
        # Split by sentences
        sentences = re.split(r'(?<=[.;])\s+', content)
        
        chunk_index = 0
        current_content = ""
        
        for sentence in sentences:
            if len(current_content) + len(sentence) <= self.max_chars:
                current_content += " " + sentence if current_content else sentence
            else:
                # Save current chunk
                if current_content:
                    chunk_id = f"{van_ban_id}_D{dieu_so}_{prefix}_{row_index}_{chunk_index}"
                    context = f"Điều {dieu_so}" + (f", Khoản {khoan_so}" if khoan_so else "")
                    
                    chunks.append(LegalChunk(
                        chunk_id=chunk_id,
                        content=self._build_chunk_content(base_metadata, current_content.strip(), context),
                        khoan_so=khoan_so,
                        chunk_index=chunk_index,
                        **base_metadata
                    ))
                    chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = current_content[-self.overlap_chars:] if len(current_content) > self.overlap_chars else ""
                current_content = overlap_text + " " + sentence if overlap_text else sentence
        
        # Save last chunk
        if current_content:
            chunk_id = f"{van_ban_id}_D{dieu_so}_{prefix}_{row_index}_{chunk_index}"
            context = f"Điều {dieu_so}" + (f", Khoản {khoan_so}" if khoan_so else "")
            
            chunks.append(LegalChunk(
                chunk_id=chunk_id,
                content=self._build_chunk_content(base_metadata, current_content.strip(), context),
                khoan_so=khoan_so,
                chunk_index=chunk_index,
                **base_metadata
            ))
        
        return chunks
    
    def _build_chunk_content(
        self,
        metadata: Dict,
        content: str,
        context: str = None
    ) -> str:
        """Build chunk content with legal context prefix"""
        # Add legal reference prefix for better retrieval
        prefix_parts = []
        
        if metadata.get('ten_van_ban'):
            prefix_parts.append(metadata['ten_van_ban'])
        
        if context:
            prefix_parts.append(context)
        elif metadata.get('dieu_so'):
            prefix_parts.append(f"Điều {metadata['dieu_so']}")
        
        if metadata.get('tieu_de_dieu'):
            prefix_parts.append(metadata['tieu_de_dieu'])
        
        prefix = " - ".join(prefix_parts) if prefix_parts else ""
        
        if prefix:
            return f"[{prefix}]\n{content}"
        return content
    
    def process_dataframe(self, df) -> List[LegalChunk]:
        """Process entire DataFrame and return all chunks"""
        all_chunks = []
        
        for idx, row in df.iterrows():
            try:
                chunks = self.chunk_article(row.to_dict(), idx)
                all_chunks.extend(chunks)
                
                if (idx + 1) % 1000 == 0:
                    logger.info(f"[CHUNKER] Processed {idx + 1} articles, {len(all_chunks)} chunks")
            except Exception as e:
                logger.warning(f"[CHUNKER] Error processing row {idx}: {e}")
                continue
        
        logger.info(f"[CHUNKER] Total: {len(all_chunks)} chunks from {len(df)} articles")
        return all_chunks


# Convenience function
def chunk_legal_csv(csv_path: str) -> List[LegalChunk]:
    """Load CSV and chunk all articles"""
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    chunker = LegalChunkerV2()
    return chunker.process_dataframe(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample
    sample = {
        'van_ban_id': '100/2019/NĐ-CP',
        'ten_van_ban': 'Nghị định về xử phạt vi phạm hành chính trong lĩnh vực giao thông',
        'loai_van_ban': 'Nghị định',
        'co_quan': 'Chính phủ',
        'chuong': 'II',
        'ten_chuong': 'Xử phạt vi phạm',
        'dieu_so': '6',
        'tieu_de_dieu': 'Xử phạt người điều khiển xe mô tô',
        'clean_text': """1. Phạt tiền từ 100.000 đồng đến 200.000 đồng đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây:
a) Không chấp hành hiệu lệnh của đèn tín hiệu giao thông;
b) Không đội mũ bảo hiểm hoặc đội mũ bảo hiểm không đúng quy cách;
c) Chở người ngồi trên xe không đội mũ bảo hiểm.
2. Phạt tiền từ 200.000 đồng đến 300.000 đồng đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây:
a) Điều khiển xe lạng lách, đánh võng;
b) Điều khiển xe chạy quá tốc độ quy định từ 10 km/h đến 20 km/h."""
    }
    
    chunker = LegalChunkerV2()
    chunks = chunker.chunk_article(sample, 0)
    
    print(f"Generated {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"\n--- {chunk.chunk_id} ---")
        print(f"Khoản: {chunk.khoan_so}, Điểm: {chunk.diem_so}")
        print(f"Content ({len(chunk.content)} chars):")
        print(chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content)
