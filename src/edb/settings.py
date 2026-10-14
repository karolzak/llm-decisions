"""Runtime settings loaded from environment variables / `.env`.

Only environment-specific values live here (endpoints, credentials, paths).
Everything experiment-specific (deployments, prompts, labels, datasets) lives in `configs/`.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    azure_subscription_id: str | None = None
    azure_tenant_id: str | None = None
    azure_resource_group: str = "kz-exp-rg"
    # Foundry (AIServices) resource + project hosting the model deployments.
    azure_ai_resource: str = "kz-exp-foundry"
    azure_ai_project: str = "proj-default"

    # Entra ID scope for Azure OpenAI and Foundry model endpoints.
    azure_token_scope: str = "https://cognitiveservices.azure.com/.default"

    # Optional overrides; derived from `azure_ai_resource` when unset.
    azure_openai_base_url: str | None = None
    azure_ai_endpoint: str | None = None

    # Direct OpenAI platform access (OpenAI Decisions API is not served by Azure yet).
    openai_api_key: str | None = Field(default=None, repr=False)
    openai_base_url_direct: str = "https://api.openai.com/v1/"

    hf_token: str | None = Field(default=None, repr=False)

    # Azure ML workspace + GPU cluster for local-model (C-series) jobs.
    aml_workspace: str = "kz-exp-aml"
    aml_compute: str = "cc-nc24ads-a100"

    configs_dir: Path = REPO_ROOT / "configs"
    data_dir: Path = REPO_ROOT / "data"
    artifacts_dir: Path = REPO_ROOT / "artifacts"
    # Large reusable caches (embeddings, fitted heads, hidden-state features); defaults to
    # <artifacts_dir>/cache. Remote jobs point it at local disk to avoid syncing it.
    cache_dir: Path | None = None

    @model_validator(mode="after")
    def _default_cache_dir(self) -> "Settings":
        if self.cache_dir is None:
            self.cache_dir = self.artifacts_dir / "cache"
        return self

    @property
    def openai_base_url(self) -> str:
        if self.azure_openai_base_url:
            return self.azure_openai_base_url.rstrip("/") + "/"
        return f"https://{self.azure_ai_resource}.openai.azure.com/openai/v1/"

    @property
    def ai_endpoint(self) -> str:
        if self.azure_ai_endpoint:
            return self.azure_ai_endpoint.rstrip("/")
        return f"https://{self.azure_ai_resource}.services.ai.azure.com"

    @property
    def project_endpoint(self) -> str:
        return f"{self.ai_endpoint}/api/projects/{self.azure_ai_project}"

    @property
    def decision1_url(self) -> str:
        return f"{self.ai_endpoint}/providers/microsoft/v1/systemone"


@lru_cache
def get_settings() -> Settings:
    return Settings()
