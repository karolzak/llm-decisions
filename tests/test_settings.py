import pytest

from edb.settings import Settings


def test_derived_endpoints():
    s = Settings(_env_file=None, azure_ai_resource="my-res", azure_ai_project="p1")
    assert s.openai_base_url == "https://my-res.openai.azure.com/openai/v1/"
    assert s.ai_endpoint == "https://my-res.services.ai.azure.com"
    assert s.project_endpoint == "https://my-res.services.ai.azure.com/api/projects/p1"
    assert s.decision1_url == "https://my-res.services.ai.azure.com/providers/microsoft/v1/systemone"


def test_no_environment_specific_defaults(monkeypatch):
    for var in ("AZURE_AI_RESOURCE", "AZURE_AI_PROJECT", "AZURE_RESOURCE_GROUP",
                "AZURE_SUBSCRIPTION_ID", "AML_WORKSPACE", "AML_COMPUTE"):
        monkeypatch.delenv(var, raising=False)
    s = Settings(_env_file=None)
    assert s.azure_ai_resource is None and s.aml_workspace is None and s.aml_compute is None
    with pytest.raises(RuntimeError, match="AZURE_AI_RESOURCE"):
        _ = s.openai_base_url
    with pytest.raises(RuntimeError, match="AZURE_AI_PROJECT"):
        _ = Settings(_env_file=None, azure_ai_resource="r").project_endpoint


def test_endpoint_overrides():
    s = Settings(
        _env_file=None,
        azure_openai_base_url="https://x.example/openai/v1",
        azure_ai_endpoint="https://y.example/",
    )
    assert s.openai_base_url == "https://x.example/openai/v1/"
    assert s.decision1_url == "https://y.example/providers/microsoft/v1/systemone"
