from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    solver_model: str
    parser_model: str
    judge_model: str
    supports_vision: bool = False
    temperature: float = 0.0
    max_tokens: int = 1200


@dataclass
class APIConfig:
    api_key_env: str
    base_url: str
    api_version: str = "2024-12-01-preview"
    request_timeout_sec: int = 300
    max_retries: int = 3


@dataclass
class LocalConfig:
    device: str = "auto"
    dtype: str = "auto"
    trust_remote_code: bool = False
    local_files_only: bool = True
    hf_token_env: str | None = None


@dataclass
class SweepConfig:
    penalties: list[float]


@dataclass
class RunConfig:
    data_path: Path
    out_dir: Path
    prompt_strategy: int
    max_examples: int | None
    random_seed: int
    save_llm_traces: bool = False


@dataclass
class Config:
    provider: str
    solver_provider: str
    parser_provider: str
    judge_provider: str
    api: APIConfig | None
    local: LocalConfig
    models: ModelConfig
    sweep: SweepConfig
    run: RunConfig


def _expand(path_str: str) -> Path:
    return Path(os.path.expandvars(path_str)).expanduser().resolve()


def load_config(path: str | Path) -> Config:
    p = Path(path)
    data = tomllib.loads(p.read_text(encoding="utf-8"))

    provider = str(data.get("llm", {}).get("provider", "api")).strip().lower()
    if provider not in {"api", "local"}:
        raise ValueError("llm.provider must be 'api' or 'local'")
    solver_provider = str(data.get("llm", {}).get("solver_provider", provider)).strip().lower()
    parser_provider = str(data.get("llm", {}).get("parser_provider", provider)).strip().lower()
    judge_provider = str(data.get("llm", {}).get("judge_provider", provider)).strip().lower()
    for role_name, role_provider in (
        ("solver", solver_provider),
        ("parser", parser_provider),
        ("judge", judge_provider),
    ):
        if role_provider not in {"api", "local"}:
            raise ValueError(f"llm.{role_name}_provider must be 'api' or 'local'")

    api_data = data.get("api")
    api: APIConfig | None = None
    if api_data:
        api = APIConfig(
            api_key_env=api_data["api_key_env"],
            base_url=api_data["base_url"],
            api_version=api_data.get("api_version", "2024-12-01-preview"),
            request_timeout_sec=int(api_data.get("request_timeout_sec", 300)),
            max_retries=int(api_data.get("max_retries", 3)),
        )

    if any(x == "api" for x in (provider, solver_provider, parser_provider, judge_provider)) and api is None:
        raise ValueError("API provider selected for at least one role but [api] section is missing")

    local_data = data.get("local", {})
    local = LocalConfig(
        device=str(local_data.get("device", "auto")),
        dtype=str(local_data.get("dtype", "auto")),
        trust_remote_code=bool(local_data.get("trust_remote_code", False)),
        local_files_only=bool(local_data.get("local_files_only", True)),
        hf_token_env=(
            str(local_data.get("hf_token_env")).strip()
            if local_data.get("hf_token_env")
            else None
        ),
    )

    models = ModelConfig(
        solver_model=data["models"]["solver_model"],
        parser_model=data["models"].get("parser_model", data["models"]["solver_model"]),
        judge_model=data["models"]["judge_model"],
        supports_vision=bool(data["models"].get("supports_vision", False)),
        temperature=float(data["models"].get("temperature", 0.0)),
        max_tokens=int(data["models"].get("max_tokens", 1200)),
    )

    sweep = SweepConfig(
        penalties=[float(x) for x in data["sweep"]["penalties"]],
    )

    run = RunConfig(
        data_path=_expand(data["run"]["data_path"]),
        out_dir=_expand(data["run"]["out_dir"]),
        prompt_strategy=int(data["run"].get("prompt_strategy", 1)),
        max_examples=(int(data["run"]["max_examples"]) if data["run"].get("max_examples") else None),
        random_seed=int(data["run"].get("random_seed", 42)),
        save_llm_traces=bool(data["run"].get("save_llm_traces", False)),
    )

    return Config(
        provider=provider,
        solver_provider=solver_provider,
        parser_provider=parser_provider,
        judge_provider=judge_provider,
        api=api,
        local=local,
        models=models,
        sweep=sweep,
        run=run,
    )


def resolve_api_key(api_key_env: str) -> str:
    key = os.getenv(api_key_env, "").strip()
    if not key:
        raise RuntimeError(
            f"Missing API key. Set environment variable {api_key_env} first."
        )
    return key
