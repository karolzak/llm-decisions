"""Runtime settings loaded from environment variables / `.env`."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Azure subscription / Foundry resource hosting the model deployments.
    azure_subscription_id: str | None = None
    azure_tenant_id: str | None = None
    azure_resource_group: str = "kz-exp-rg"
    azure_ai_resource: str = "llm-decisions-resource"

    # Entra ID scope for Azure OpenAI and Foundry model endpoints.
    azure_token_scope: str = "https://cognitiveservices.azure.com/.default"

    # Deployment names (the `model` field sent to the APIs).
    gpt6_luna_deployment: str = "gpt-6-luna"
    decision1_deployment: str = "Decision-1"

    # Optional overrides; derived from `azure_ai_resource` when unset.
    azure_openai_base_url: str | None = None
    azure_ai_endpoint: str | None = None

    # Azure ML workspace used for GPU runs of open-weight models.
    aml_workspace: str = "kz-exp-aml"
    aml_location: str = "swedencentral"
    aml_compute_name: str = "kz-gpu-ci"
    aml_compute_size: str = "Standard_NC24ads_A100_v4"

    # Open-weight model.
    oss_model_id: str = "openai/gpt-oss-20b"
    hf_token: str | None = Field(default=None, repr=False)

    # Local GPU price used for cost accounting (USD / hour).
    gpu_hourly_price_usd: float | None = None

    data_dir: Path = REPO_ROOT / "data"
    artifacts_dir: Path = REPO_ROOT / "artifacts"

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
    def decision1_url(self) -> str:
        return f"{self.ai_endpoint}/providers/microsoft/v1/systemone"


@lru_cache
def get_settings() -> Settings:
    return Settings()
