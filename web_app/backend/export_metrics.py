"""
Export All Quality Metrics
===========================
Script to export all quality metrics to file (JSON, CSV, or TXT)

Usage:
    python export_metrics.py [--format json|csv|txt] [--output filename]
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from database import get_db, AnswerMetrics, Message, ChatSession, MessageSource
from sqlalchemy import desc

def export_all_metrics(format='json', output=None):
    """Export all metrics to file"""
    
    print(f"\n📊 Exporting all quality metrics to {format.upper()}...")
    
    with get_db() as db:
        # Get all metrics with related message and session info
        metrics = db.query(AnswerMetrics)\
            .join(Message)\
            .join(ChatSession)\
            .order_by(desc(AnswerMetrics.created_at))\
            .all()
        
        if not metrics:
            print("❌ No metrics found in database.")
            return
        
        print(f"✅ Found {len(metrics)} quality assessments")
        
        # Prepare data
        data_list = []
        for m in metrics:
            # Get message and sources
            message = db.query(Message).filter_by(id=m.message_id).first()
            sources = db.query(MessageSource).filter_by(message_id=m.message_id).all() if message else []
            session = db.query(ChatSession).filter_by(id=message.session_id).first() if message else None
            
            data_list.append({
                'id': m.id,
                'timestamp': m.created_at.isoformat() if m.created_at else None,
                'session_id': session.id if session else None,
                'session_title': session.title if session else None,
                'user_id': session.user_id if session else None,
                'query': m.query,
                'answer': m.answer,
                'quality_metrics': {
                    'overall_score': m.overall_score,
                    'grade': m.grade,
                    'feedback': m.feedback,
                    'bert_score': m.bert_score,
                    'hallucination_score': m.hallucination_score,
                    'factuality_score': m.factuality_score,
                    'context_relevance': m.context_relevance,
                },
                'legacy_metrics': {
                    'query_answer_score': m.query_answer_score,
                    'answer_context_score': m.answer_context_score,
                    'extractive_score': m.extractive_score,
                    'hallucination_risk': m.hallucination_risk,
                },
                'context_info': {
                    'top_k_contexts': m.top_k_contexts,
                    'contexts_used': m.contexts_used,
                    'reranker_used': m.reranker_used,
                },
                'sources': [
                    {
                        'type': s.source_type,
                        'label': s.label,
                        'detail': s.detail,
                        'score': s.score,
                        'content_preview': s.content_preview,
                        'rank': s.rank,
                    }
                    for s in sources
                ],
                'message_content': message.content if message else None,
            })
        
        # Generate filename if not provided
        if not output:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output = f"quality_metrics_export_{timestamp}.{format}"
        
        # Export based on format
        if format == 'json':
            export_json(data_list, output)
        elif format == 'csv':
            export_csv(data_list, output)
        elif format == 'txt':
            export_txt(data_list, output)
        else:
            print(f"❌ Unsupported format: {format}")
            return
        
        print(f"✅ Exported to: {output}")
        print(f"📝 Total records: {len(data_list)}")

def export_json(data_list, filename):
    """Export as JSON"""
    export_data = {
        'export_time': datetime.now().isoformat(),
        'total_records': len(data_list),
        'metrics': data_list,
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

def export_csv(data_list, filename):
    """Export as CSV"""
    import csv
    
    with open(filename, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'ID', 'Timestamp', 'Session ID', 'User ID', 'Query', 'Answer',
            'Overall Score', 'Grade', 'Feedback',
            'BERTScore', 'Hallucination Score', 'Factuality Score', 'Context Relevance',
            'Top K', 'Contexts Used', 'Reranker Used', 'Sources Count'
        ])
        
        # Data rows
        for item in data_list:
            writer.writerow([
                item['id'],
                item['timestamp'],
                item['session_id'],
                item['user_id'],
                item['query'],
                item['answer'][:100] + '...' if item['answer'] and len(item['answer']) > 100 else item['answer'],
                item['quality_metrics']['overall_score'],
                item['quality_metrics']['grade'],
                item['quality_metrics']['feedback'],
                item['quality_metrics']['bert_score'],
                item['quality_metrics']['hallucination_score'],
                item['quality_metrics']['factuality_score'],
                item['quality_metrics']['context_relevance'],
                item['context_info']['top_k_contexts'],
                item['context_info']['contexts_used'],
                item['context_info']['reranker_used'],
                len(item['sources']),
            ])

def export_txt(data_list, filename):
    """Export as formatted text"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("LEGAL RAG - QUALITY METRICS EXPORT\n")
        f.write("="*100 + "\n")
        f.write(f"Export Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Records: {len(data_list)}\n")
        f.write("="*100 + "\n\n")
        
        for i, item in enumerate(data_list, 1):
            f.write(f"\n{'='*100}\n")
            f.write(f"RECORD #{i}\n")
            f.write(f"{'='*100}\n")
            f.write(f"ID: {item['id']}\n")
            f.write(f"Timestamp: {item['timestamp']}\n")
            f.write(f"Session: {item['session_id']} ({item['session_title']})\n")
            f.write(f"User: {item['user_id']}\n")
            f.write(f"\n--- QUERY ---\n{item['query']}\n")
            f.write(f"\n--- ANSWER ---\n{item['answer']}\n")
            
            metrics = item['quality_metrics']
            f.write(f"\n--- QUALITY METRICS ---\n")
            f.write(f"Overall Score: {metrics['overall_score']:.3f} ({(metrics['overall_score']*100):.1f}%)\n")
            f.write(f"Grade: {metrics['grade']}\n")
            f.write(f"Feedback: {metrics['feedback']}\n")
            f.write(f"\nDetailed Scores:\n")
            f.write(f"  - BERTScore:           {metrics['bert_score']:.3f} ({(metrics['bert_score']*100):.1f}%)\n")
            f.write(f"  - Hallucination Score: {metrics['hallucination_score']:.3f} ({(metrics['hallucination_score']*100):.1f}%)\n")
            f.write(f"  - Factuality Score:    {metrics['factuality_score']:.3f} ({(metrics['factuality_score']*100):.1f}%)\n")
            f.write(f"  - Context Relevance:   {metrics['context_relevance']:.3f} ({(metrics['context_relevance']*100):.1f}%)\n")
            
            context = item['context_info']
            f.write(f"\n--- CONTEXT INFO ---\n")
            f.write(f"Top K: {context['top_k_contexts']}\n")
            f.write(f"Contexts Used: {context['contexts_used']}\n")
            f.write(f"Reranker Used: {'Yes' if context['reranker_used'] else 'No'}\n")
            
            if item['sources']:
                f.write(f"\n--- SOURCES ({len(item['sources'])}) ---\n")
                for j, source in enumerate(item['sources'], 1):
                    f.write(f"{j}. {source['label']}")
                    if source['detail']:
                        f.write(f" - {source['detail']}")
                    if source['score']:
                        f.write(f" (score: {source['score']:.2f})")
                    f.write("\n")
            
            f.write("\n")

def print_statistics(format='txt'):
    """Print statistics about metrics"""
    with get_db() as db:
        from sqlalchemy import func
        
        total = db.query(func.count(AnswerMetrics.id)).scalar()
        avg_overall = db.query(func.avg(AnswerMetrics.overall_score)).scalar()
        avg_bert = db.query(func.avg(AnswerMetrics.bert_score)).scalar()
        
        grades = db.query(AnswerMetrics.grade, func.count(AnswerMetrics.id))\
            .filter(AnswerMetrics.grade.isnot(None))\
            .group_by(AnswerMetrics.grade)\
            .all()
        
        print("\n" + "="*100)
        print("📊 QUALITY METRICS STATISTICS")
        print("="*100)
        print(f"\nTotal Assessments: {total}")
        print(f"Average Overall Score: {avg_overall:.3f} ({(avg_overall*100):.1f}%)" if avg_overall else "N/A")
        print(f"Average BERTScore: {avg_bert:.3f} ({(avg_bert*100):.1f}%)" if avg_bert else "N/A")
        
        if grades:
            print(f"\nGrade Distribution:")
            for grade, count in sorted(grades):
                pct = (count / total) * 100
                bar = "█" * int(pct / 2)
                print(f"  {grade}: {count:3d} ({pct:5.1f}%) {bar}")
        
        print("="*100 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export quality metrics to file")
    parser.add_argument('--format', choices=['json', 'csv', 'txt'], default='json',
                       help='Export format (default: json)')
    parser.add_argument('--output', help='Output filename (optional)')
    parser.add_argument('--stats', action='store_true',
                       help='Show statistics only (no export)')
    
    args = parser.parse_args()
    
    try:
        if args.stats:
            print_statistics()
        else:
            export_all_metrics(format=args.format, output=args.output)
            print_statistics()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
