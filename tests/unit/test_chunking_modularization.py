from types import SimpleNamespace

from src.legal_rag.chunking.metadata_builder import build_provision_id, generate_chunk_id
from src.legal_rag.chunking.policy import build_chunking_policy
from src.legal_rag.chunking.provision_chunker import build_unit_windows


def test_build_unit_windows_preserves_order_and_overlap():
    units = [SimpleNamespace(token_len=v) for v in (40, 40, 40, 40, 40)]

    windows = build_unit_windows(
        units,
        token_getter=lambda item: item.token_len,
        target_chunk_tokens=90,
        hard_cap_tokens=100,
        overlap_tokens=40,
    )

    assert len(windows) >= 2
    assert windows[0][0] is units[0]
    assert windows[-1][-1] is units[-1]


def test_metadata_builder_generates_stable_ids():
    provision_id = build_provision_id("100/2015/QH13", dieu_so="77")
    chunk_id = generate_chunk_id(
        doc_id="100/2015/QH13",
        dieu_so="77",
        chunk_index=3,
        text="Dieu 77 noi dung kiem thu",
        overflow_index=2,
    )

    assert provision_id == "100/2015/QH13_D77"
    assert chunk_id.startswith("100/2015/QH13_D77_W2_C3_")


def test_chunking_policy_builder_keeps_expected_values():
    policy = build_chunking_policy(
        min_chars=1200,
        max_chars=2800,
        overlap_chars=120,
        target_chunk_tokens=420,
        hard_cap_tokens=448,
        overlap_tokens=64,
    )

    assert policy.min_chars == 1200
    assert policy.max_chars == 2800
    assert policy.target_chunk_tokens == 420
    assert policy.hard_cap_tokens == 448

