from edb.settings import Settings


def test_derived_endpoints():
    s = Settings(_env_file=None, azure_ai_resource="my-res", azure_ai_project="p1")
    assert s.openai_base_url == "https://my-res.openai.azure.com/openai/v1/"
    assert s.ai_endpoint == "https://my-res.services.ai.azure.com"
    assert s.project_endpoint == "https://my-res.services.ai.azure.com/api/projects/p1"
    assert s.decision1_url == "https://my-res.services.ai.azure.com/providers/microsoft/v1/systemone"


def test_defaults_point_at_kz_exp_foundry(monkeypatch):
    monkeypatch.delenv("AZURE_AI_RESOURCE", raising=False)
    monkeypatch.delenv("AZURE_AI_PROJECT", raising=False)
    s = Settings(_env_file=None)
    assert s.azure_ai_resource == "kz-exp-foundry"
    assert s.azure_ai_project == "proj-default"


def test_endpoint_overrides():
    s = Settings(
        _env_file=None,
        azure_openai_base_url="https://x.example/openai/v1",
        azure_ai_endpoint="https://y.example/",
    )
    assert s.openai_base_url == "https://x.example/openai/v1/"
    assert s.decision1_url == "https://y.example/providers/microsoft/v1/systemone"
