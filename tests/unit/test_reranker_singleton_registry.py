from src import reranker as reranker_module


class _FakeLegalReranker:
    def __init__(self, model_name, use_gpu, max_length, batch_size):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.max_length = max_length
        self.batch_size = batch_size


def test_get_legal_reranker_reuses_same_instance_for_same_config(monkeypatch):
    monkeypatch.setattr(reranker_module, "LegalReranker", _FakeLegalReranker)
    monkeypatch.setattr(reranker_module, "_legal_reranker_registry", {})

    first = reranker_module.get_legal_reranker(
        model_name="AITeamVN/Vietnamese_Reranker",
        use_gpu=True,
        max_length=2304,
        batch_size=8,
    )
    second = reranker_module.get_legal_reranker(
        model_name="AITeamVN/Vietnamese_Reranker",
        use_gpu=True,
        max_length=2304,
        batch_size=8,
    )

    assert first is second


def test_get_legal_reranker_separates_instances_by_config(monkeypatch):
    monkeypatch.setattr(reranker_module, "LegalReranker", _FakeLegalReranker)
    monkeypatch.setattr(reranker_module, "_legal_reranker_registry", {})

    gpu_instance = reranker_module.get_legal_reranker(
        model_name="AITeamVN/Vietnamese_Reranker",
        use_gpu=True,
        max_length=2304,
        batch_size=8,
    )
    cpu_instance = reranker_module.get_legal_reranker(
        model_name="AITeamVN/Vietnamese_Reranker",
        use_gpu=False,
        max_length=2304,
        batch_size=8,
    )

    assert gpu_instance is not cpu_instance


def test_hybrid_reranker_uses_shared_legal_reranker(monkeypatch):
    sentinel = object()

    def _fake_get_legal_reranker(**kwargs):
        return sentinel

    monkeypatch.setattr(reranker_module, "get_legal_reranker", _fake_get_legal_reranker)

    hybrid = reranker_module.HybridReranker(
        reranker_model="AITeamVN/Vietnamese_Reranker",
        use_reranker=True,
    )

    assert hybrid.reranker is sentinel
