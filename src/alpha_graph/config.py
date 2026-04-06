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

# Single source of truth for any random seed in the project. Every model,
# train/test split, sampler, and shuffle should use this — never a literal.
SEED = 42


def set_global_seeds(seed: int = SEED) -> None:
    """Seed Python, NumPy, and (if installed) PyTorch for reproducibility.

    Call this at the top of every entry-point script. Model classes that
    accept their own seed (LightGBM, sklearn) still need it passed
    explicitly — this only covers process-level RNG state.
    """
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
    finnhub_api_key: str = field(
        default_factory=lambda: os.getenv("FINNHUB_API_KEY", "")
    )
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
    filing_types: list[str] = field(default_factory=lambda: ["10-K", "10-Q", "8-K"])
    filing_years_back: int = 5

    # LLM settings (works with any OpenAI-compatible API: OpenAI, Together, Groq, etc.)
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V3")
    )
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
