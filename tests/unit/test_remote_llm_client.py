from src.services.remote_llm_client import RemoteLLMClient


class _StubResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_remote_llm_client_tgi(monkeypatch):
    calls = {}

    def _fake_post(url, json, timeout):
        calls['url'] = url
        calls['json'] = json
        calls['timeout'] = timeout
        return _StubResponse({'choices': [{'text': 'rewrite output'}]})

    monkeypatch.setattr('requests.post', _fake_post)
    client = RemoteLLMClient(provider_type='tgi_completions', base_url='http://tgi/v1/completions', model='tgi', timeout_s=12)

    result = client.generate_prompt('hello')

    assert result == 'rewrite output'
    assert calls['url'] == 'http://tgi/v1/completions'
    assert calls['json']['model'] == 'tgi'
    assert calls['timeout'] == 12


def test_remote_llm_client_ollama(monkeypatch):
    calls = {}

    def _fake_post(url, json, timeout):
        calls['url'] = url
        calls['json'] = json
        calls['timeout'] = timeout
        return _StubResponse({'response': 'answer output'})

    monkeypatch.setattr('requests.post', _fake_post)
    client = RemoteLLMClient(provider_type='ollama_generate', base_url='http://ollama:11434', model='qwen3:8b', timeout_s=30)

    result = client.generate('Cau hoi?', 'Van ban')

    assert result == 'answer output'
    assert calls['url'] == 'http://ollama:11434/api/generate'
    assert calls['json']['model'] == 'qwen3:8b'
    assert calls['timeout'] == 30
