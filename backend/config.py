from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    openai_api_key: str | None = None
    elevenlabs_api_key: str | None = None
    pyannote_token: str | None = None
    asr_model_size: str = "small"  # e.g., "tiny", "small", "medium"
    aasist_checkpoint_path: str | None = None  # e.g., "backend/models/aasist_scripted.pt"

    model_config = SettingsConfigDict(env_file="../.env", env_file_encoding="utf-8")

settings = Settings()