"""
Configuration manager for litmus_v2.
Loads settings from YAML, validates model family independence, and provides defaults.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import os
import yaml
from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    family: str
    model: str
    provider: str
    temperature: float = 0.0
    fallback_model: Optional[str] = None
    fallback_provider: Optional[str] = None


class PathsConfig(BaseModel):
    docs_dir: str = "docs"
    data_dir: str = "litmus_v2/data"
    results_dir: str = "litmus_v2/results"


class ChunkingConfig(BaseModel):
    strategy: str = "hierarchical_paragraph"
    max_tokens: int = 300
    overlap_tokens: int = 50


class RetrievalConfig(BaseModel):
    top_k: int = 5
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rrf_k: int = 60


class ActiveSamplingConfig(BaseModel):
    target_sample_size: int = 100
    categories: Dict[str, int] = Field(
        default_factory=lambda: {
            "single_fact": 25,
            "numeric": 25,
            "yes_no": 20,
            "multi_hop": 15,
            "unanswerable": 15,
        }
    )


class LitmusConfig(BaseModel):
    seed: int = 42
    paths: PathsConfig = Field(default_factory=PathsConfig)
    model_families: Dict[str, ModelSpec]
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    active_sampling: ActiveSamplingConfig = Field(default_factory=ActiveSamplingConfig)

    def validate_family_independence(self) -> Dict[str, Any]:
        """
        Validates that Generator (Family A), Answering (Family B), and Judge (Family C)
        originate from distinct model families to prevent evaluation circularity.
        """
        gen = self.model_families.get("generator")
        ans = self.model_families.get("answering")
        judge = self.model_families.get("judge")

        f_gen = gen.family.lower() if gen else "unknown"
        f_ans = ans.family.lower() if ans else "unknown"
        f_judge = judge.family.lower() if judge else "unknown"

        overlaps = []
        if f_gen == f_ans:
            overlaps.append(f"Generator and Answering share family: {f_gen}")
        if f_ans == f_judge:
            overlaps.append(f"Answering and Judge share family: {f_ans}")
        if f_gen == f_judge:
            overlaps.append(f"Generator and Judge share family: {f_gen}")

        is_independent = len(overlaps) == 0
        return {
            "is_independent": is_independent,
            "generator_family": f_gen,
            "answering_family": f_ans,
            "judge_family": f_judge,
            "overlaps": overlaps,
        }


def load_config(config_path: Optional[str] = None) -> LitmusConfig:
    """Load configuration from YAML file or return defaults."""
    if config_path is None:
        # Default relative to litmus_v2 root or workspace root
        candidates = [
            Path("litmus_v2/config/config.yaml"),
            Path("config/config.yaml"),
            Path(__file__).parent / "config.yaml",
        ]
        for c in candidates:
            if c.exists():
                config_path = str(c)
                break

    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
            return LitmusConfig(**raw)

    # Fallback default configuration
    return LitmusConfig(
        model_families={
            "generator": ModelSpec(family="qwen", model="qwen/qwen3.8-27b", provider="groq"),
            "answering": ModelSpec(family="meta-llama", model="llama3.2:1b", provider="ollama"),
            "judge": ModelSpec(family="mistral", model="mixtral-8x7b-32768", provider="groq"),
        }
    )
