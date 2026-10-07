"""Settings read from environment variables (or a .env file next to backend/)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]          # project root
load_dotenv(ROOT / "backend" / ".env")


class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    vision_model: str = os.getenv("VISION_MODEL", "gpt-4.1-mini")
    search_model: str = os.getenv("SEARCH_MODEL", "gpt-4o-mini-search-preview")
    web_research: bool = os.getenv("WEB_RESEARCH", "on").lower() in ("1", "on", "true", "yes")
    admin_key: str = os.getenv("ADMIN_KEY", "")
    learned_db: Path = Path(os.getenv("LEARNED_DB", ROOT / "data" / "learned.db"))
    data_path: Path = Path(os.getenv("DATA_PATH", ROOT / "data" / "crop_master.xlsx"))
    frontend_dir: Path = Path(os.getenv("FRONTEND_DIR", ROOT / "frontend"))
    allowed_origins: list[str] = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
    max_tool_rounds: int = int(os.getenv("MAX_TOOL_ROUNDS", "2"))
    max_history_turns: int = int(os.getenv("MAX_HISTORY_TURNS", "10"))
    max_image_mb: float = float(os.getenv("MAX_IMAGE_MB", "5"))
    jwt_secret: str = os.getenv("JWT_SECRET", "")
    mobile_pepper: str = os.getenv("MOBILE_PEPPER", "")
    otp_provider: str = os.getenv("OTP_PROVIDER", "console").lower()
    msg91_auth_key: str = os.getenv("MSG91_AUTH_KEY", "")
    website_jwt_secret: str = os.getenv("WEBSITE_JWT_SECRET", "")
    farmers_db: Path = Path(os.getenv("FARMERS_DB", ROOT / "data" / "farmers.db"))


settings = Settings()
