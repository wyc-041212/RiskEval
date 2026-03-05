from __future__ import annotations

import http.client
import json
import socket
import time
from dataclasses import dataclass
from typing import Protocol
from urllib import error, parse, request

from .config import Config, resolve_api_key


class ChatClient(Protocol):
    def complete(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        image_url: str | None = None,
    ) -> str: ...


@dataclass
class _ModelBundle:
    tokenizer: object
    model: object
    device: object


class APILLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        api_version: str,
        model: str,
        temperature: float,
        max_tokens: int,
        request_timeout_sec: int,
        max_retries: int,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.api_version = api_version
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout_sec = request_timeout_sec
        self.max_retries = max_retries

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        image_url: str | None = None,
    ) -> str:
        use_model = model or self.model
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if image_url:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            )
        else:
            messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "max_tokens": self.max_tokens,
            "top_p": 1,
            "stream": False,
        }
        if self.temperature != 1.0:
            payload["temperature"] = self.temperature

        errors: list[str] = []
        for path_prefix in ("", "/openai"):
            url = self._build_url(path_prefix, use_model)
            try:
                data = self._post_json(url, payload)
                return self._extract_chat_text(data)
            except RuntimeError as exc:
                errors.append(f"{url}: {exc}")

        raise RuntimeError(
            "No supported HKBU chat completion endpoint succeeded.\n" + "\n".join(errors)
        )

    def _build_url(self, path_prefix: str, model: str) -> str:
        quoted_model = parse.quote(model, safe="")
        query = parse.urlencode({"api-version": self.api_version})
        return (
            f"{self.base_url}{path_prefix}/deployments/{quoted_model}/chat/completions?{query}"
        )

    def _post_json(self, url: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(url=url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        # Some gateways expect Bearer auth, others Azure-style api-key.
        req.add_header("Authorization", f"Bearer {self.api_key}")
        req.add_header("api-key", self.api_key)

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with request.urlopen(req, timeout=self.request_timeout_sec) as resp:
                    raw = resp.read().decode("utf-8")
                break
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code in {429, 500, 502, 503, 504}:
                    last_error = exc
                    if attempt == self.max_retries:
                        raise RuntimeError(
                            f"HTTP {exc.code} after {self.max_retries} attempts: {detail}"
                        ) from exc
                    print(
                        f"[retry {attempt}/{self.max_retries}] transient HTTP {exc.code}, retrying in {attempt}s",
                        flush=True,
                    )
                    time.sleep(attempt)
                    continue
                raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
            except (
                TimeoutError,
                socket.timeout,
                error.URLError,
                http.client.RemoteDisconnected,
                http.client.IncompleteRead,
            ) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    reason = getattr(exc, "reason", str(exc))
                    raise RuntimeError(
                        f"Network timeout/error after {self.max_retries} attempts: {reason}"
                    ) from exc
                print(
                    f"[retry {attempt}/{self.max_retries}] request failed, retrying in {attempt}s: {exc}",
                    flush=True,
                )
                time.sleep(attempt)
        else:
            raise RuntimeError(f"Request failed: {last_error}")

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON response: {raw[:400]}") from exc

    @staticmethod
    def _extract_chat_text(data: dict) -> str:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"Missing choices in response: {json.dumps(data)[:400]}")

        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            if parts:
                return "\n".join(parts).strip()

        raise RuntimeError(f"Missing assistant content in response: {json.dumps(data)[:400]}")


class LocalHFClient:
    def __init__(
        self,
        default_model: str,
        temperature: float,
        max_tokens: int,
        device: str,
        dtype: str,
        trust_remote_code: bool,
        local_files_only: bool,
        hf_token: str | None,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local backend requires transformers and torch. "
                "Install with: pip install 'transformers>=4.41.0' 'torch>=2.2.0'"
            ) from exc

        self._torch = torch
        self._AutoModelForCausalLM = AutoModelForCausalLM
        self._AutoTokenizer = AutoTokenizer
        self.default_model = default_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device_pref = device.strip().lower()
        self.dtype_pref = dtype.strip().lower()
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self.hf_token = hf_token
        self._bundles: dict[str, _ModelBundle] = {}

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        image_url: str | None = None,
    ) -> str:
        if image_url:
            raise RuntimeError(
                "Local backend currently supports text-only generation. "
                "Set supports_vision=false or switch to API vision model."
            )

        model_name = model or self.default_model
        bundle = self._get_bundle(model_name)
        rendered = self._render_prompt(bundle.tokenizer, prompt=prompt, system=system)
        inputs = bundle.tokenizer(rendered, return_tensors="pt").to(bundle.device)

        generate_kwargs = {
            "max_new_tokens": self.max_tokens,
            "pad_token_id": bundle.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = 1.0
        else:
            generate_kwargs["do_sample"] = False

        output_ids = bundle.model.generate(**inputs, **generate_kwargs)
        new_tokens = output_ids[0, inputs.input_ids.shape[-1] :]
        return bundle.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _resolve_device(self) -> str:
        if self.device_pref != "auto":
            return self.device_pref
        if self._torch.backends.mps.is_available():
            return "mps"
        if self._torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _resolve_dtype(self, device: str):
        if self.dtype_pref == "auto":
            if device in {"mps", "cuda"}:
                return self._torch.float16
            return self._torch.float32

        mapping = {
            "float16": self._torch.float16,
            "fp16": self._torch.float16,
            "bfloat16": self._torch.bfloat16,
            "bf16": self._torch.bfloat16,
            "float32": self._torch.float32,
            "fp32": self._torch.float32,
        }
        if self.dtype_pref not in mapping:
            raise ValueError(
                "local.dtype must be one of auto/float16/bfloat16/float32 (or fp16/bf16/fp32)"
            )
        return mapping[self.dtype_pref]

    def _get_bundle(self, model_name: str) -> _ModelBundle:
        if model_name in self._bundles:
            return self._bundles[model_name]

        device = self._resolve_device()
        dtype = self._resolve_dtype(device)

        tokenizer = self._AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
            token=self.hf_token,
        )
        model = self._AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
            token=self.hf_token,
            torch_dtype=dtype,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model.to(device)
        model.eval()
        bundle = _ModelBundle(tokenizer=tokenizer, model=model, device=device)
        self._bundles[model_name] = bundle
        return bundle

    @staticmethod
    def _render_prompt(tokenizer, prompt: str, system: str | None) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            return apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        if system:
            return f"System:\n{system}\n\nUser:\n{prompt}\n\nAssistant:\n"
        return f"User:\n{prompt}\n\nAssistant:\n"


def build_client(cfg: Config) -> ChatClient:
    if cfg.provider == "api":
        if cfg.api is None:
            raise RuntimeError("Missing [api] config for API provider")
        return APILLMClient(
            api_key=resolve_api_key(cfg.api.api_key_env),
            base_url=cfg.api.base_url,
            api_version=cfg.api.api_version,
            model=cfg.models.solver_model,
            temperature=cfg.models.temperature,
            max_tokens=cfg.models.max_tokens,
            request_timeout_sec=cfg.api.request_timeout_sec,
            max_retries=cfg.api.max_retries,
        )

    hf_token = None
    if cfg.local.hf_token_env:
        hf_token = resolve_api_key(cfg.local.hf_token_env)

    return LocalHFClient(
        default_model=cfg.models.solver_model,
        temperature=cfg.models.temperature,
        max_tokens=cfg.models.max_tokens,
        device=cfg.local.device,
        dtype=cfg.local.dtype,
        trust_remote_code=cfg.local.trust_remote_code,
        local_files_only=cfg.local.local_files_only,
        hf_token=hf_token,
    )


def build_client_for_provider(cfg: Config, provider: str) -> ChatClient:
    provider = provider.strip().lower()
    if provider == "api":
        if cfg.api is None:
            raise RuntimeError("Missing [api] config for API provider")
        return APILLMClient(
            api_key=resolve_api_key(cfg.api.api_key_env),
            base_url=cfg.api.base_url,
            api_version=cfg.api.api_version,
            model=cfg.models.solver_model,
            temperature=cfg.models.temperature,
            max_tokens=cfg.models.max_tokens,
            request_timeout_sec=cfg.api.request_timeout_sec,
            max_retries=cfg.api.max_retries,
        )
    if provider == "local":
        hf_token = None
        if cfg.local.hf_token_env:
            hf_token = resolve_api_key(cfg.local.hf_token_env)
        return LocalHFClient(
            default_model=cfg.models.solver_model,
            temperature=cfg.models.temperature,
            max_tokens=cfg.models.max_tokens,
            device=cfg.local.device,
            dtype=cfg.local.dtype,
            trust_remote_code=cfg.local.trust_remote_code,
            local_files_only=cfg.local.local_files_only,
            hf_token=hf_token,
        )
    raise ValueError("provider must be 'api' or 'local'")
