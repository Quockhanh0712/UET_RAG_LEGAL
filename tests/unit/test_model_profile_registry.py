from src.services.model_profile_registry import get_default_model_profile_id, get_model_profile, list_model_profiles


def test_model_profiles_expose_two_presets():
    profiles = list_model_profiles()
    ids = {item.id for item in profiles}

    assert 'legal_tgi_remote' in ids
    assert 'qwen3_8b_ollama_remote' in ids


def test_get_model_profile_falls_back_to_default():
    default_id = get_default_model_profile_id()
    profile = get_model_profile('missing_profile')

    assert profile.id == default_id
