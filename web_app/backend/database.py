"""
SQLite Database for Legal RAG System

Stores:
- Chat sessions and messages
- Answer quality metrics (BERTScore, grades)
- Uploaded documents metadata
- Source tracking

Usage:
    from database import get_db, ChatSession, Message, AnswerMetrics
    
    with get_db() as db:
        session = ChatSession(user_id="user123")
        db.add(session)
        db.commit()
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from contextlib import contextmanager

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Text, DateTime, ForeignKey, Index, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.pool import StaticPool

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Database path
DB_DIR = PROJECT_ROOT / "data" / "database"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "legal_rag.db"

# SQLAlchemy setup
Base = declarative_base()


# ==============================================================================
# Models
# ==============================================================================

class ChatSession(Base):
    """Chat session with user"""
    __tablename__ = "chat_sessions"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    title = Column(String, nullable=True)  # First user message preview
    message_count = Column(Integer, default=0)
    
    # Relationships
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "title": self.title,
            "message_count": self.message_count
        }


class Message(Base):
    """Individual message in chat"""
    __tablename__ = "messages"
    
    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False, index=True)
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    
    # Request metadata
    search_mode = Column(String, nullable=True)  # 'legal', 'user', 'hybrid'
    top_k = Column(Integer, nullable=True)
    reranker_used = Column(Boolean, default=False)
    
    # Timing metrics
    search_time = Column(Float, nullable=True)
    generation_time = Column(Float, nullable=True)
    rerank_time = Column(Float, nullable=True)
    total_time = Column(Float, nullable=True)
    
    # Relationships
    session = relationship("ChatSession", back_populates="messages")
    metrics = relationship("AnswerMetrics", back_populates="message", uselist=False, cascade="all, delete-orphan")
    sources = relationship("MessageSource", back_populates="message", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_session_created', 'session_id', 'created_at'),
        Index('idx_role', 'role'),
    )
    
    def to_dict(self) -> Dict:
        result = {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "search_mode": self.search_mode,
            "top_k": self.top_k,
            "reranker_used": self.reranker_used,
            "search_time": self.search_time,
            "generation_time": self.generation_time,
            "rerank_time": self.rerank_time,
            "total_time": self.total_time
        }
        
        # Include metrics if available
        if self.metrics:
            result["metrics"] = self.metrics.to_dict()
        
        # Include sources
        if self.sources:
            result["sources"] = [s.to_dict() for s in self.sources]
        
        return result


class AnswerMetrics(Base):
    """Quality metrics for assistant answers"""
    __tablename__ = "answer_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(String, ForeignKey("messages.id"), nullable=False, unique=True, index=True)
    
    # Overall scores
    overall_score = Column(Float, nullable=True)  # 0-1 range
    grade = Column(String, nullable=True)  # A/B/C/D/F
    feedback = Column(Text, nullable=True)
    
    # Quality Monitor metrics (0-1 range)
    bert_score = Column(Float, nullable=True)  # Semantic similarity
    hallucination_score = Column(Float, nullable=True)  # 1.0 = no hallucination
    factuality_score = Column(Float, nullable=True)  # Factual accuracy
    context_relevance = Column(Float, nullable=True)  # Context quality
    
    # Legacy fields (for backward compatibility)
    query_answer_score = Column(Float, nullable=True)  # BERTScore: query vs answer
    answer_context_score = Column(Float, nullable=True)  # BERTScore: answer vs contexts
    extractive_score = Column(Float, nullable=True)  # How much is directly extracted
    hallucination_risk = Column(Float, nullable=True)  # 0-1, higher = more risk
    
    # Context info
    top_k_contexts = Column(Integer, nullable=True)
    contexts_used = Column(Integer, nullable=True)
    reranker_used = Column(Boolean, default=False)
    
    # Query and answer for reference
    query = Column(Text, nullable=True)
    answer = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    message = relationship("Message", back_populates="metrics")
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "message_id": self.message_id,
            "overall_score": self.overall_score,
            "grade": self.grade,
            "feedback": self.feedback,
            "bert_score": self.bert_score,
            "hallucination_score": self.hallucination_score,
            "factuality_score": self.factuality_score,
            "context_relevance": self.context_relevance,
            "query_answer_score": self.query_answer_score,
            "answer_context_score": self.answer_context_score,
            "extractive_score": self.extractive_score,
            "hallucination_risk": self.hallucination_risk,
            "top_k_contexts": self.top_k_contexts,
            "contexts_used": self.contexts_used,
            "reranker_used": self.reranker_used,
            "query": self.query,
            "answer": self.answer,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class MessageSource(Base):
    """Source documents for a message"""
    __tablename__ = "message_sources"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(String, ForeignKey("messages.id"), nullable=False, index=True)
    
    source_type = Column(String, nullable=True)  # 'legal' or 'user'
    label = Column(String, nullable=True)  # e.g., "Điều 123"
    detail = Column(String, nullable=True)  # e.g., "Luật Hình sự 2015"
    content_preview = Column(Text, nullable=True)  # First 500 chars
    score = Column(Float, nullable=True)  # Relevance score
    rank = Column(Integer, nullable=True)  # Position in results (1-based)
    
    # Relationships
    message = relationship("Message", back_populates="sources")
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "message_id": self.message_id,
            "source_type": self.source_type,
            "label": self.label,
            "detail": self.detail,
            "content_preview": self.content_preview,
            "score": self.score,
            "rank": self.rank
        }


class UploadedDocument(Base):
    """Metadata for user-uploaded documents"""
    __tablename__ = "uploaded_documents"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    session_id = Column(String, nullable=True, index=True)
    
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=True)  # 'pdf', 'docx', 'txt'
    file_size = Column(Integer, nullable=True)  # bytes
    chunk_count = Column(Integer, nullable=True)
    job_id = Column(String, nullable=True, index=True)
    ingest_mode = Column(String, nullable=True)
    page_count = Column(Integer, nullable=True)
    text_chars = Column(Integer, nullable=True)
    parser_used = Column(String, nullable=True)
    ocr_provider = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)

    upload_time = Column(DateTime, default=datetime.now)
    indexed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    status = Column(String, default='queued')  # queued, running, needs_ocr, ready, failed, deleted

    # Indexes
    __table_args__ = (
        Index('idx_user_status', 'user_id', 'status'),
        Index('idx_session', 'session_id'),
    )
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "chunk_count": self.chunk_count,
            "job_id": self.job_id,
            "ingest_mode": self.ingest_mode,
            "page_count": self.page_count,
            "text_chars": self.text_chars,
            "parser_used": self.parser_used,
            "ocr_provider": self.ocr_provider,
            "error_message": self.error_message,
            "upload_time": self.upload_time.isoformat() if self.upload_time else None,
            "indexed_at": self.indexed_at.isoformat() if self.indexed_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "status": self.status
        }


class DocumentIngestJob(Base):
    """Background ingestion job state for uploaded documents."""

    __tablename__ = "document_ingest_jobs"

    id = Column(String, primary_key=True)
    doc_id = Column(String, ForeignKey("uploaded_documents.id"), nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    session_id = Column(String, nullable=True, index=True)
    filename = Column(String, nullable=False)
    content_mode = Column(String, nullable=False)
    ocr_provider = Column(String, nullable=True)
    parser_used = Column(String, nullable=True)
    status = Column(String, nullable=False, default="queued")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    def to_dict(self) -> Dict:
        return {
            "job_id": self.id,
            "doc_id": self.doc_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "filename": self.filename,
            "content_mode": self.content_mode,
            "ocr_provider": self.ocr_provider,
            "parser_used": self.parser_used,
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


# ==============================================================================
# Database Engine & Session
# ==============================================================================

# Create engine
engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False, "timeout": 30},
    poolclass=StaticPool,
    echo=False  # Set to True for SQL debugging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db():
    """Context manager for database sessions"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


def init_db():
    """Initialize database (create all tables)"""
    print(f"📦 Initializing database at {DB_PATH}")
    ensure_db_schema()
    print("✅ Database initialized successfully")


def ensure_db_schema():
    """Create tables and add backward-compatible columns for SQLite deployments."""
    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    if not inspector.has_table("uploaded_documents"):
        return

    existing_columns = {column["name"] for column in inspector.get_columns("uploaded_documents")}
    missing_column_ddls = {
        "job_id": "ALTER TABLE uploaded_documents ADD COLUMN job_id VARCHAR",
        "ingest_mode": "ALTER TABLE uploaded_documents ADD COLUMN ingest_mode VARCHAR",
        "page_count": "ALTER TABLE uploaded_documents ADD COLUMN page_count INTEGER",
        "text_chars": "ALTER TABLE uploaded_documents ADD COLUMN text_chars INTEGER",
        "parser_used": "ALTER TABLE uploaded_documents ADD COLUMN parser_used VARCHAR",
        "ocr_provider": "ALTER TABLE uploaded_documents ADD COLUMN ocr_provider VARCHAR",
        "error_message": "ALTER TABLE uploaded_documents ADD COLUMN error_message TEXT",
        "indexed_at": "ALTER TABLE uploaded_documents ADD COLUMN indexed_at DATETIME",
        "updated_at": "ALTER TABLE uploaded_documents ADD COLUMN updated_at DATETIME",
    }

    with engine.begin() as conn:
        for column_name, ddl in missing_column_ddls.items():
            if column_name not in existing_columns:
                conn.exec_driver_sql(ddl)

        # Ensure updated_at is initialized for older rows.
        conn.exec_driver_sql(
            "UPDATE uploaded_documents SET updated_at = COALESCE(updated_at, upload_time) "
            "WHERE updated_at IS NULL"
        )


def reset_db():
    """Reset database (drop all tables and recreate)"""
    print("⚠️  Resetting database...")
    Base.metadata.drop_all(bind=engine)
    ensure_db_schema()
    print("✅ Database reset complete")


# ==============================================================================
# Helper Functions
# ==============================================================================

def save_chat_session(session_id: str, user_id: str, title: Optional[str] = None) -> ChatSession:
    """Save or update chat session"""
    with get_db() as db:
        session = db.query(ChatSession).filter_by(id=session_id).first()
        
        if not session:
            session = ChatSession(
                id=session_id,
                user_id=user_id,
                title=title
            )
            db.add(session)
        else:
            session.updated_at = datetime.now()
            if title:
                session.title = title
        
        db.commit()
        db.refresh(session)
        return session


def save_message(
    message_id: str,
    session_id: str,
    role: str,
    content: str,
    **kwargs
) -> Message:
    """Save message to database"""
    with get_db() as db:
        message = Message(
            id=message_id,
            session_id=session_id,
            role=role,
            content=content,
            **kwargs
        )
        db.add(message)
        
        # Update session message count
        session = db.query(ChatSession).filter_by(id=session_id).first()
        if session:
            session.message_count += 1
            session.updated_at = datetime.now()
        
        db.commit()
        db.refresh(message)
        return message


def save_answer_metrics(
    message_id: str,
    overall_score: float,
    grade: str,
    feedback: str,
    **kwargs
) -> AnswerMetrics:
    """Save answer quality metrics"""
    with get_db() as db:
        metrics = AnswerMetrics(
            message_id=message_id,
            overall_score=overall_score,
            grade=grade,
            feedback=feedback,
            **kwargs
        )
        db.add(metrics)
        db.commit()
        db.refresh(metrics)
        return metrics


def save_message_sources(message_id: str, sources: List[Dict]) -> List[MessageSource]:
    """Save sources for a message"""
    with get_db() as db:
        source_objects = []
        for i, source in enumerate(sources, 1):
            source_obj = MessageSource(
                message_id=message_id,
                source_type=source.get("type"),
                label=source.get("label"),
                detail=source.get("detail"),
                content_preview=source.get("content_preview", "")[:500],
                score=source.get("score"),
                rank=i
            )
            db.add(source_obj)
            source_objects.append(source_obj)
        
        db.commit()
        for obj in source_objects:
            db.refresh(obj)
        
        return source_objects


def save_uploaded_document(
    *,
    doc_id: str,
    user_id: str,
    session_id: Optional[str],
    filename: str,
    file_type: str,
    file_size: int,
    chunk_count: int = 0,
    job_id: Optional[str] = None,
    ingest_mode: Optional[str] = None,
    page_count: Optional[int] = None,
    text_chars: Optional[int] = None,
    parser_used: Optional[str] = None,
    ocr_provider: Optional[str] = None,
    error_message: Optional[str] = None,
    status: str = "queued",
    indexed_at: Optional[datetime] = None,
) -> UploadedDocument:
    """Create or update uploaded document metadata."""
    with get_db() as db:
        document = db.query(UploadedDocument).filter_by(id=doc_id).first()
        if document is None:
            document = UploadedDocument(
                id=doc_id,
                user_id=user_id,
                session_id=session_id,
                filename=filename,
                file_type=file_type,
                file_size=file_size,
                upload_time=datetime.now(),
            )
            db.add(document)

        document.user_id = user_id
        document.session_id = session_id
        document.filename = filename
        document.file_type = file_type
        document.file_size = file_size
        document.chunk_count = chunk_count
        document.job_id = job_id
        document.ingest_mode = ingest_mode
        document.page_count = page_count
        document.text_chars = text_chars
        document.parser_used = parser_used
        document.ocr_provider = ocr_provider
        document.error_message = error_message
        document.status = status
        document.indexed_at = indexed_at
        document.updated_at = datetime.now()

        db.commit()
        db.refresh(document)
        return document


def get_uploaded_document(doc_id: str, user_id: Optional[str] = None) -> Optional[UploadedDocument]:
    with get_db() as db:
        query = db.query(UploadedDocument).filter_by(id=doc_id)
        if user_id is not None:
            query = query.filter_by(user_id=user_id)
        document = query.first()
        if document is None:
            return None
        db.expunge(document)
        return document


def list_uploaded_documents(
    user_id: str,
    session_id: Optional[str] = None,
    include_deleted: bool = False,
) -> List[UploadedDocument]:
    with get_db() as db:
        query = db.query(UploadedDocument).filter_by(user_id=user_id)
        if session_id is not None:
            query = query.filter_by(session_id=session_id)
        if not include_deleted:
            query = query.filter(UploadedDocument.status != "deleted")
        documents = query.order_by(UploadedDocument.upload_time.desc()).all()
        for document in documents:
            db.expunge(document)
        return documents


def mark_uploaded_document_deleted(doc_id: str, user_id: str) -> bool:
    with get_db() as db:
        document = db.query(UploadedDocument).filter_by(id=doc_id, user_id=user_id).first()
        if document is None:
            return False
        document.status = "deleted"
        document.updated_at = datetime.now()
        db.commit()
        return True


def delete_uploaded_document(doc_id: str, user_id: str) -> bool:
    with get_db() as db:
        document = db.query(UploadedDocument).filter_by(id=doc_id, user_id=user_id).first()
        if document is None:
            return False
        db.delete(document)
        db.commit()
        return True


def save_document_ingest_job(
    *,
    job_id: str,
    doc_id: str,
    user_id: str,
    session_id: Optional[str],
    filename: str,
    content_mode: str,
    status: str,
    ocr_provider: Optional[str] = None,
    parser_used: Optional[str] = None,
    error_message: Optional[str] = None,
    started_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
) -> DocumentIngestJob:
    """Create or update ingestion job metadata."""
    with get_db() as db:
        job = db.query(DocumentIngestJob).filter_by(id=job_id).first()
        if job is None:
            job = DocumentIngestJob(
                id=job_id,
                doc_id=doc_id,
                user_id=user_id,
                session_id=session_id,
                filename=filename,
                content_mode=content_mode,
                created_at=datetime.now(),
            )
            db.add(job)

        job.doc_id = doc_id
        job.user_id = user_id
        job.session_id = session_id
        job.filename = filename
        job.content_mode = content_mode
        job.status = status
        job.ocr_provider = ocr_provider
        job.parser_used = parser_used
        job.error_message = error_message
        if started_at is not None:
            job.started_at = started_at
        if completed_at is not None:
            job.completed_at = completed_at
        job.updated_at = datetime.now()

        db.commit()
        db.refresh(job)
        return job


def get_document_ingest_job(job_id: str) -> Optional[DocumentIngestJob]:
    with get_db() as db:
        job = db.query(DocumentIngestJob).filter_by(id=job_id).first()
        if job is None:
            return None
        db.expunge(job)
        return job


def get_document_ingest_job_for_doc(doc_id: str) -> Optional[DocumentIngestJob]:
    with get_db() as db:
        job = (
            db.query(DocumentIngestJob)
            .filter_by(doc_id=doc_id)
            .order_by(DocumentIngestJob.created_at.desc())
            .first()
        )
        if job is None:
            return None
        db.expunge(job)
        return job


def delete_document_ingest_jobs(doc_id: str) -> int:
    with get_db() as db:
        jobs = db.query(DocumentIngestJob).filter_by(doc_id=doc_id).all()
        deleted = len(jobs)
        for job in jobs:
            db.delete(job)
        db.commit()
        return deleted


def get_session_messages(session_id: str) -> List[Message]:
    """Get all messages for a session"""
    with get_db() as db:
        messages = db.query(Message).filter_by(session_id=session_id).order_by(Message.created_at).all()
        return [msg.to_dict() for msg in messages]


def get_user_sessions(user_id: str, limit: int = 50) -> List[ChatSession]:
    """Get user's chat sessions"""
    with get_db() as db:
        sessions = db.query(ChatSession)\
            .filter_by(user_id=user_id)\
            .order_by(ChatSession.updated_at.desc())\
            .limit(limit)\
            .all()
        return [s.to_dict() for s in sessions]


def get_metrics_stats(user_id: Optional[str] = None, days: int = 7) -> Dict:
    """Get aggregated metrics statistics"""
    with get_db() as db:
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=days)
        
        query = db.query(AnswerMetrics).join(Message).join(ChatSession)
        
        if user_id:
            query = query.filter(ChatSession.user_id == user_id)
        
        query = query.filter(AnswerMetrics.created_at >= cutoff)
        
        metrics = query.all()
        
        if not metrics:
            return {
                "total_answers": 0,
                "avg_overall_score": 0,
                "grade_distribution": {},
                "avg_hallucination_risk": 0,
                "reranker_usage_rate": 0
            }
        
        # Calculate stats
        total = len(metrics)
        avg_score = sum(m.overall_score or 0 for m in metrics) / total
        
        # Calculate average for each metric
        bert_scores = [m.bert_score for m in metrics if m.bert_score is not None]
        avg_bert = sum(bert_scores) / len(bert_scores) if bert_scores else 0
        
        factuality_scores = [m.factuality_score for m in metrics if m.factuality_score is not None]
        avg_factuality = sum(factuality_scores) / len(factuality_scores) if factuality_scores else 0
        
        context_relevance = [m.context_relevance for m in metrics if m.context_relevance is not None]
        avg_context = sum(context_relevance) / len(context_relevance) if context_relevance else 0
        
        grades = {}
        for m in metrics:
            if m.grade:
                grades[m.grade] = grades.get(m.grade, 0) + 1
        
        avg_hallucination = sum(m.hallucination_risk or 0 for m in metrics) / total
        reranker_used = sum(1 for m in metrics if m.reranker_used)
        
        return {
            "total_answers": total,
            "avg_overall_score": round(avg_score, 2),
            "avg_bert_score": round(avg_bert, 3) if avg_bert else None,
            "avg_factuality": round(avg_factuality, 3) if avg_factuality else None,
            "avg_context_relevance": round(avg_context, 3) if avg_context else None,
            "grade_distribution": grades,
            "avg_hallucination_risk": round(avg_hallucination, 3),
            "reranker_usage_rate": round(reranker_used / total, 2) if total > 0 else 0
        }


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Database management for Legal RAG")
    parser.add_argument("--init", action="store_true", help="Initialize database")
    parser.add_argument("--reset", action="store_true", help="Reset database (DANGER!)")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    
    args = parser.parse_args()
    
    if args.init:
        init_db()
    elif args.reset:
        confirm = input("⚠️  This will DELETE ALL DATA. Type 'RESET' to confirm: ")
        if confirm == "RESET":
            reset_db()
        else:
            print("❌ Reset cancelled")
    elif args.stats:
        init_db()  # Ensure tables exist
        stats = get_metrics_stats()
        print("\n📊 Database Statistics:")
        print(f"  Total answers: {stats['total_answers']}")
        print(f"  Avg score: {stats['avg_overall_score']}")
        print(f"  Grade distribution: {stats['grade_distribution']}")
        print(f"  Avg hallucination risk: {stats['avg_hallucination_risk']}")
        print(f"  Reranker usage: {stats['reranker_usage_rate']:.0%}")
    else:
        print("Usage: python database.py [--init|--reset|--stats]")
