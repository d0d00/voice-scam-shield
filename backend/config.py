from pydantic import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str | None = None
    elevenlabs_api_key: str | None = None
    pyannote_token: str | None = None

    class Config:
        env_file = "../.env"

settings = Settings()