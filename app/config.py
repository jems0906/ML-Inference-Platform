from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    model_name_v1: str = "distilbert-base-uncased-finetuned-sst-2-english"
    model_name_v2: str = "distilbert-base-uncased-finetuned-sst-2-english"
    ab_test_percentage: float = 0.5  # fraction of requests routed to v2

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()