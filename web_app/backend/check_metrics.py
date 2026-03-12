"""
View Quality Metrics from Database
==================================
Script to view quality metrics saved in database

Usage:
    python check_metrics.py
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from database import get_db, ChatSession, Message, AnswerMetrics
from sqlalchemy import func, desc

def show_recent_metrics(limit=10):
    """Show recent quality metrics"""
    print("\n" + "="*100)
    print(f"📊 RECENT QUALITY METRICS (Last {limit} messages)")
    print("="*100)
    
    with get_db() as db:
        # Get recent metrics
        metrics = db.query(AnswerMetrics)\
            .join(Message)\
            .order_by(desc(AnswerMetrics.created_at))\
            .limit(limit)\
            .all()
        
        if not metrics:
            print("\n❌ No metrics found. Send some chat messages first!")
            return
        
        for i, m in enumerate(metrics, 1):
            print(f"\n{i}. Message ID: {m.message_id[:8]}...")
            overall = f"{m.overall_score:.3f}" if m.overall_score else "0.000"
            print(f"   Grade: {m.grade or 'N/A'} | Overall Score: {overall}")
            bert = f"{m.bert_score:.3f}" if m.bert_score else "0.000"
            print(f"   📈 BERTScore: {bert}")
            hall = f"{m.hallucination_score:.3f}" if m.hallucination_score else "0.000"
            print(f"   🔍 Hallucination: {hall}")
            fact = f"{m.factuality_score:.3f}" if m.factuality_score else "0.000"
            print(f"   ✅ Factuality: {fact}")
            rel = f"{m.context_relevance:.3f}" if m.context_relevance else "0.000"
            print(f"   🎯 Context Rel: {rel}")
            if m.feedback:
                print(f"   💬 Feedback: {m.feedback[:80]}...")
            if m.query:
                print(f"   ❓ Query: {m.query[:60]}...")
            if m.answer:
                print(f"   💡 Answer: {m.answer[:60]}...")
            print(f"   🗂️  Contexts: {m.contexts_used or 0}/{m.top_k_contexts or 0}")
            print(f"   🔄 Reranker: {'Yes' if m.reranker_used else 'No'}")
            print(f"   📅 Time: {m.created_at}")

def show_statistics():
    """Show aggregated statistics"""
    print("\n" + "="*100)
    print("📈 QUALITY STATISTICS")
    print("="*100)
    
    with get_db() as db:
        # Total metrics
        total = db.query(func.count(AnswerMetrics.id)).scalar()
        
        if total == 0:
            print("\n❌ No statistics available yet.")
            return
        
        # Average scores
        avg_overall = db.query(func.avg(AnswerMetrics.overall_score)).scalar()
        avg_bert = db.query(func.avg(AnswerMetrics.bert_score)).scalar()
        avg_hallucination = db.query(func.avg(AnswerMetrics.hallucination_score)).scalar()
        avg_factuality = db.query(func.avg(AnswerMetrics.factuality_score)).scalar()
        avg_relevance = db.query(func.avg(AnswerMetrics.context_relevance)).scalar()
        
        print(f"\n📊 Total Evaluated Messages: {total}")
        print(f"\n🎯 Average Scores:")
        print(f"   Overall:       {avg_overall:.3f}" if avg_overall else "   Overall:       0.000")
        print(f"   BERTScore:     {avg_bert:.3f}" if avg_bert else "   BERTScore:     0.000")
        print(f"   Hallucination: {avg_hallucination:.3f}" if avg_hallucination else "   Hallucination: 0.000")
        print(f"   Factuality:    {avg_factuality:.3f}" if avg_factuality else "   Factuality:    0.000")
        print(f"   Relevance:     {avg_relevance:.3f}" if avg_relevance else "   Relevance:     0.000")
        
        # Grade distribution
        grades = db.query(AnswerMetrics.grade, func.count(AnswerMetrics.id))\
            .filter(AnswerMetrics.grade.isnot(None))\
            .group_by(AnswerMetrics.grade)\
            .all()
        
        if grades:
            print(f"\n📊 Grade Distribution:")
            for grade, count in sorted(grades, key=lambda x: x[0]):
                pct = (count / total) * 100
                bar = "█" * int(pct / 2)
                print(f"   {grade}: {count:3d} ({pct:5.1f}%) {bar}")
        
        # Reranker usage
        reranker_count = db.query(func.count(AnswerMetrics.id))\
            .filter(AnswerMetrics.reranker_used == True)\
            .scalar()
        
        if total > 0:
            reranker_pct = (reranker_count / total) * 100
            print(f"\n🔄 Reranker Usage: {reranker_count}/{total} ({reranker_pct:.1f}%)")
        
        # Best and worst
        best = db.query(AnswerMetrics)\
            .filter(AnswerMetrics.overall_score.isnot(None))\
            .order_by(desc(AnswerMetrics.overall_score))\
            .first()
        
        worst = db.query(AnswerMetrics)\
            .filter(AnswerMetrics.overall_score.isnot(None))\
            .order_by(AnswerMetrics.overall_score)\
            .first()
        
        if best:
            print(f"\n🏆 Best Answer: Score={best.overall_score:.3f}, Grade={best.grade}")
            if best.query:
                print(f"   Query: {best.query[:80]}...")
        
        if worst:
            print(f"\n⚠️  Worst Answer: Score={worst.overall_score:.3f}, Grade={worst.grade}")
            if worst.query:
                print(f"   Query: {worst.query[:80]}...")

def show_sessions():
    """Show chat sessions"""
    print("\n" + "="*100)
    print("💬 CHAT SESSIONS")
    print("="*100)
    
    with get_db() as db:
        sessions = db.query(ChatSession)\
            .order_by(desc(ChatSession.updated_at))\
            .limit(5)\
            .all()
        
        if not sessions:
            print("\n❌ No sessions found.")
            return
        
        for i, s in enumerate(sessions, 1):
            print(f"\n{i}. Session: {s.id}")
            print(f"   User: {s.user_id}")
            print(f"   Title: {s.title or 'Untitled'}")
            print(f"   Messages: {s.message_count}")
            print(f"   Created: {s.created_at}")
            print(f"   Updated: {s.updated_at}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="View quality metrics from database")
    parser.add_argument("--limit", type=int, default=5, help="Number of recent metrics to show")
    parser.add_argument("--stats", action="store_true", help="Show statistics only")
    parser.add_argument("--sessions", action="store_true", help="Show sessions only")
    
    args = parser.parse_args()
    
    try:
        if args.stats:
            show_statistics()
        elif args.sessions:
            show_sessions()
        else:
            show_recent_metrics(limit=args.limit)
            show_statistics()
            show_sessions()
        
        print("\n" + "="*100)
        print("✅ Done!")
        print("="*100 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
