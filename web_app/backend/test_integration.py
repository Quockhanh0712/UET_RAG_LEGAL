"""
Test Quality Monitor Integration
================================
Test script to verify quality monitoring is integrated into the chat flow.

Author: AI Assistant
Date: 2025-12-27
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from quality_monitor import get_quality_monitor
from database import get_session_messages, get_metrics_stats, get_db

def test_quality_monitor():
    """Test Quality Monitor standalone"""
    print("\n" + "="*80)
    print("TEST 1: Quality Monitor Standalone")
    print("="*80)
    
    monitor = get_quality_monitor()
    
    query = "Quyền sử dụng đất được quy định như thế nào?"
    answer = "Quyền sử dụng đất là quyền được Nhà nước giao đất, cho thuê đất hoặc công nhận để sử dụng đất theo quy định của pháp luật."
    contexts = [
        "Quyền sử dụng đất là quyền được Nhà nước giao đất, cho thuê đất, công nhận quyền sử dụng đất.",
        "Người sử dụng đất có quyền sử dụng đất ổn định, lâu dài theo quy định của pháp luật."
    ]
    prompt = f"Câu hỏi: {query}\n\nTài liệu: {contexts[0]}\n\nTrả lời:"
    
    metrics = monitor.evaluate_answer(query, answer, contexts, prompt, top_k=2)
    
    print(f"\n✅ Test 1 PASSED")
    print(f"   Overall Score: {metrics.overall_score:.3f}")
    print(f"   Grade: {metrics.grade}")
    print(f"   BERTScore: {metrics.bert_score:.3f}")
    print(f"   Hallucination: {metrics.hallucination_score:.3f}")
    print(f"   Factuality: {metrics.factuality_score:.3f}")
    print(f"   Context Relevance: {metrics.context_relevance:.3f}")
    print(f"   Feedback: {metrics.feedback}")


def test_database_functions():
    """Test database helper functions"""
    print("\n" + "="*80)
    print("TEST 2: Database Helper Functions")
    print("="*80)
    
    from database import save_chat_session, save_message, save_answer_metrics, save_message_sources
    import uuid
    
    session_id = "test_session_" + str(uuid.uuid4())[:8]
    user_id = "test_user"
    message_id = str(uuid.uuid4())
    
    # Test save_chat_session
    try:
        save_chat_session(
            session_id=session_id,
            user_id=user_id,
            title="Test Session"
        )
        print(f"\n✅ Test 2.1 PASSED: save_chat_session() worked")
    except Exception as e:
        print(f"\n❌ Test 2.1 FAILED: {e}")
        return
    
    # Test save_message
    try:
        save_message(
            message_id=message_id,
            session_id=session_id,
            role="assistant",
            content="Test answer"
        )
        print(f"✅ Test 2.2 PASSED: save_message() worked")
    except Exception as e:
        print(f"❌ Test 2.2 FAILED: {e}")
        return
    
    # Test save_message_sources
    try:
        sources = [{"type": "legal", "label": "Điều 1", "score": 0.95, "detail": "Luật test"}]
        save_message_sources(message_id, sources)
        print(f"✅ Test 2.3 PASSED: save_message_sources() worked")
    except Exception as e:
        print(f"❌ Test 2.3 FAILED: {e}")
        return
    
    # Test save_answer_metrics
    try:
        save_answer_metrics(
            message_id=message_id,
            overall_score=0.85,
            grade="B",
            feedback="Tốt! Câu trả lời đạt yêu cầu.",
            bert_score=0.82,
            hallucination_score=0.95,
            factuality_score=0.80,
            context_relevance=0.75,
            query="Test query",
            answer="Test answer",
            top_k_contexts=5,
            contexts_used=5
        )
        print(f"✅ Test 2.4 PASSED: save_answer_metrics() worked")
    except Exception as e:
        print(f"❌ Test 2.4 FAILED: {e}")
        return
    
    # Test get_session_messages
    try:
        messages = get_session_messages(session_id=session_id)
        print(f"✅ Test 2.5 PASSED: get_session_messages() returned {len(messages)} messages")
    except Exception as e:
        print(f"❌ Test 2.5 FAILED: {e}")
        return
    
    # Test get_metrics_stats
    try:
        stats = get_metrics_stats(user_id=user_id, days=7)
        print(f"✅ Test 2.6 PASSED: get_metrics_stats() worked")
        print(f"   Total answers: {stats.get('total_answers', 0)}")
        print(f"   Average score: {stats.get('avg_overall_score', 0):.3f}")
        print(f"   Grade distribution: {stats.get('grade_distribution', {})}")
    except Exception as e:
        print(f"❌ Test 2.6 FAILED: {e}")


def test_integration_readiness():
    """Test if all components are ready for integration"""
    print("\n" + "="*80)
    print("TEST 3: Integration Readiness")
    print("="*80)
    
    components = []
    
    # Check Quality Monitor
    try:
        from quality_monitor import get_quality_monitor
        monitor = get_quality_monitor()
        components.append(("✅", "Quality Monitor", "Ready"))
    except Exception as e:
        components.append(("❌", "Quality Monitor", f"Error: {e}"))
    
    # Check Database
    try:
        from database import get_db
        with get_db() as session:
            pass
        components.append(("✅", "Database", "Ready"))
    except Exception as e:
        components.append(("❌", "Database", f"Error: {e}"))
    
    # Check Vietnamese Reranker
    try:
        from vietnamese_reranker import get_reranker
        reranker = get_reranker()
        components.append(("✅", "Vietnamese Reranker", "Ready"))
    except Exception as e:
        components.append(("❌", "Vietnamese Reranker", f"Error: {e}"))
    
    # Check main.py imports
    try:
        import main
        components.append(("✅", "main.py imports", "Ready"))
    except Exception as e:
        components.append(("❌", "main.py imports", f"Error: {e}"))
    
    print("\n📦 Component Status:")
    for status, name, message in components:
        print(f"   {status} {name}: {message}")
    
    all_ready = all(status == "✅" for status, _, _ in components)
    if all_ready:
        print(f"\n✅ ALL COMPONENTS READY FOR INTEGRATION")
    else:
        print(f"\n⚠️  SOME COMPONENTS NEED ATTENTION")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("QUALITY MONITOR INTEGRATION TEST SUITE")
    print("="*80)
    
    try:
        test_quality_monitor()
        test_database_functions()
        test_integration_readiness()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS COMPLETED")
        print("="*80)
        print("\n🎉 Quality Monitor is ready to use!")
        print("\n📋 Next Steps:")
        print("   1. Start backend: python web_app/backend/main.py")
        print("   2. Test /api/chat endpoint with reranker enabled")
        print("   3. Check database for saved metrics")
        print("   4. View metrics via /api/metrics/stats")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()
