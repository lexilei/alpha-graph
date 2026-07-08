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
CACHE_DIR = DATA_DIR / "cache"

# Single source of truth for any random seed in the project.
SEED = 42


def set_global_seeds(seed: int = SEED) -> None:
    """Seed Python, NumPy, and (if installed) PyTorch."""
    import os as _os
    import random as _random

    _random.seed(seed)
    _os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch as _torch  # noqa
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@dataclass
class Config:
    sec_user_agent: str = field(
        default_factory=lambda: os.getenv("SEC_EDGAR_USER_AGENT", "")
    )

    # LLM for relationship extraction (any OpenAI-compatible endpoint).
    llm_api_key: str = field(
        default_factory=lambda: os.getenv(
            "LLM_API_KEY", os.getenv("TOGETHER_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        )
    )
    llm_base_url: str = field(
        default_factory=lambda: os.getenv(
            "LLM_BASE_URL",
            "https://api.together.xyz/v1" if os.getenv("TOGETHER_API_KEY") else "",
        )
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V4-Pro")
    )

    universe_index: str = "S&P 500"
    max_tickers: int | None = None  # None = full universe

    filing_types: list[str] = field(default_factory=lambda: ["10-K", "10-Q", "8-K"])
    filing_years_back: int = 15

    def validate(self) -> list[str]:
        return [] if self.sec_user_agent else ["SEC_EDGAR_USER_AGENT"]


cfg = Config()
