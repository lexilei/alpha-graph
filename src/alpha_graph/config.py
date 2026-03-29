"""Central configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FILINGS_DIR = DATA_DIR / "filings"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
CACHE_DIR = DATA_DIR / "cache"


@dataclass
class Config:
    sec_user_agent: str = field(
        default_factory=lambda: os.getenv("SEC_EDGAR_USER_AGENT", "")
    )
    finnhub_api_key: str = field(
        default_factory=lambda: os.getenv("FINNHUB_API_KEY", "")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    alpaca_api_key: str = field(
        default_factory=lambda: os.getenv("ALPACA_API_KEY", "")
    )
    alpaca_secret_key: str = field(
        default_factory=lambda: os.getenv("ALPACA_SECRET_KEY", "")
    )

    # Universe defaults
    universe_index: str = "S&P 500"
    max_tickers: int = 100  # start with top 100 for speed

    # Filing settings
    filing_types: list[str] = field(default_factory=lambda: ["10-K", "10-Q"])
    filing_years_back: int = 5

    # LLM settings
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0

    def validate(self) -> list[str]:
        """Return list of missing required config keys."""
        missing = []
        if not self.sec_user_agent:
            missing.append("SEC_EDGAR_USER_AGENT")
        if not self.finnhub_api_key:
            missing.append("FINNHUB_API_KEY")
        return missing


cfg = Config()
