"""LLM providers used by notebooks 02 and 03.

`APIProvider`     — remote APIs (Anthropic + OpenAI-compatible: DeepSeek, Kimi, OpenAI).
`LocalHFProvider` — local HuggingFace causal LM (Colab GPU / DGX).

Both expose the same `.complete(system, user, ...) -> str` and
`.step(state_text, ...) -> (actions, raw, fallback)` interface so
`agent.make_policy_llm()` works unchanged for either backend.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import time

from src.agent import ACTION_RE, make_minimal_prompt, parse_actions


_logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  Remote API provider
# ──────────────────────────────────────────────────────────────────────────────

# Models that use the OpenAI "reasoning" parameter conventions
# (max_completion_tokens instead of max_tokens, default temperature only).
_OPENAI_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _is_openai_reasoning_model(model: str) -> bool:
    m = model.lower()
    return any(m.startswith(p) for p in _OPENAI_REASONING_PREFIXES)


class APIProvider:
    """Remote LLM provider for Anthropic and any OpenAI-compatible endpoint.

    Per-model API quirks handled here so the rest of the notebook stays clean:
      * OpenAI reasoning / GPT-5 family (gpt-5*, o1*, o3*, o4*) only supports
        `max_completion_tokens` (not `max_tokens`) and `temperature=1` (default,
        so we just omit the override).
      * Kimi `kimi-k2.5` requires `temperature=1` (set via constructor arg).
      * Anthropic and DeepSeek follow the standard convention (max_tokens,
        temperature=0 for determinism).

    Timeout is enforced via a background thread so a hung API call never
    freezes the rollout.

    Args:
        name:        Friendly name used in logs / result labels (e.g. 'anthropic').
        model:       Model ID (e.g. 'claude-haiku-4-5', 'gpt-5.4-nano').
        key_env:     Name of the env var holding the API key.
        base_url:    Override endpoint for OpenAI-compat providers.
        temperature: Force a specific sampling temperature. If None we pick:
                     0.0 for deterministic providers, and skip the override
                     entirely for OpenAI reasoning models (they only allow 1).
    """

    def __init__(
        self,
        name: str,
        model: str,
        key_env: str,
        base_url: str | None = None,
        temperature: float | None = None,
    ):
        self.name        = name
        self.model       = model
        self.label       = f"{name}:{model}"
        self.temperature = temperature

        api_key = os.environ.get(key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"Missing API key — set env var {key_env!r}")

        if name == "anthropic":
            try:
                from anthropic import Anthropic
            except ImportError as e:
                raise ImportError("pip install anthropic") from e
            self.client = Anthropic(api_key=api_key)
            self._kind  = "anthropic"
        else:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError("pip install openai") from e
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self._kind  = "openai_compat"

        self._is_openai_reasoning = (
            name == "openai" and _is_openai_reasoning_model(model)
        )

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 512,
        timeout_s: float | None = 45.0,
        **kwargs,
    ) -> str:
        """Call the API and return assistant text only."""

        def _call() -> str:
            if self._kind == "anthropic":
                temp = 0.0 if self.temperature is None else self.temperature
                resp = self.client.messages.create(
                    model=self.model,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    max_tokens=max_tokens,
                    temperature=temp,
                )
                return "".join(
                    b.text for b in resp.content if getattr(b, "type", None) == "text"
                )

            kw: dict = dict(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            if self._is_openai_reasoning:
                kw["max_completion_tokens"] = max_tokens
            else:
                kw["max_tokens"] = max_tokens
                kw["temperature"] = 0.0 if self.temperature is None else self.temperature

            resp = self.client.chat.completions.create(**kw)
            return resp.choices[0].message.content

        if timeout_s is None:
            return _call()

        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="api_call"
        )
        future = executor.submit(_call)
        try:
            result = future.result(timeout=timeout_s)
            executor.shutdown(wait=False)
            return result
        except concurrent.futures.TimeoutError:
            future.cancel()
            executor.shutdown(wait=False)
            raise TimeoutError(f"{self.label} did not respond within {timeout_s:.0f}s")
        except Exception:
            executor.shutdown(wait=False)
            raise

    def step(
        self,
        state_text: str,
        system: str | None = None,
        n_buildings: int = 6,
        max_retries: int = 2,
        timeout_s: float = 45.0,
        max_tokens: int = 250,
        **kwargs,
    ) -> tuple[list[float], str, bool]:
        """Query the API for one environment step.

        Returns (actions, raw_response, used_fallback).
        On timeout the step returns fallback zeros immediately without retrying.
        """
        _system  = system or make_minimal_prompt(n_buildings)
        last_raw = ""

        for attempt in range(max_retries):
            try:
                last_raw = self.complete(
                    _system,
                    f"STATE:\n{state_text}",
                    max_tokens=max_tokens,
                    timeout_s=timeout_s,
                )
                if ACTION_RE.search(last_raw):
                    return parse_actions(last_raw, n_buildings), last_raw, False
                _logger.debug("No action tags (attempt %d): %s", attempt + 1, last_raw[:80])
            except TimeoutError as exc:
                last_raw = f"TIMEOUT: {exc}"
                _logger.warning("API timeout provider=%s", self.name)
                break
            except Exception as exc:
                last_raw = f"ERROR (attempt {attempt + 1}): {exc}"
                if attempt < max_retries - 1:
                    time.sleep(1.0)

        _logger.warning("API fallback provider=%s — returning zeros", self.name)
        return [0.0] * n_buildings, last_raw, True


# ──────────────────────────────────────────────────────────────────────────────
#  Local HuggingFace provider
# ──────────────────────────────────────────────────────────────────────────────

class LocalHFProvider:
    """HuggingFace local model provider. Drop-in for `APIProvider`.

    Runs inference on the local GPU (Colab T4, DGX Spark, etc.). No API keys,
    no rate limits. Supports any HF causal LM, including Gemma (system-role
    workaround), Qwen3 (thinking-mode disabled), and 4-bit quantized models.

    Args:
        model_id:       HuggingFace model ID.
        load_in_4bit:   Use 4-bit NF4 quantization. Required for 8B models on T4.
        max_new_tokens: Max tokens generated per call (default for `.complete`).
    """

    def __init__(
        self,
        model_id: str,
        load_in_4bit: bool = False,
        max_new_tokens: int = 250,
    ):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.model_id       = model_id
        self.name           = "local_hf"
        self.label          = f"local:{model_id.split('/')[-1]}"
        self.max_new_tokens = max_new_tokens
        self._device        = "cuda" if torch.cuda.is_available() else "cpu"
        self._is_qwen3      = "qwen3" in model_id.lower()
        self._is_gemma      = "gemma" in model_id.lower()

        print(f"Loading {model_id} on {self._device} …")
        load_kw: dict = {"device_map": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                # float16 (not bfloat16) for T4 compatibility
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            load_kw["torch_dtype"] = (
                torch.float16 if self._device == "cuda" else torch.float32
            )

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kw)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Fallback chat template if the tokenizer doesn't ship one
        if getattr(self.tokenizer, "chat_template", None) is None:
            self.tokenizer.chat_template = (
                "{% for message in messages %}"
                "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
            )

        n_params = sum(p.numel() for p in self.model.parameters()) / 1e9
        mem_gb   = torch.cuda.memory_allocated() / 1e9 if self._device == "cuda" else 0.0
        print(f"  ✓ {n_params:.2f}B params | GPU mem: {mem_gb:.1f} GB")
        if self._is_qwen3:
            print("  Qwen3 detected — thinking mode disabled (enable_thinking=False)")
        if self._is_gemma:
            print("  Gemma detected — system prompt merged into user message")

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int | None = None,
        **kwargs,
    ) -> str:
        """Generate a response. Returns newly generated text only."""
        import torch

        max_new = max_tokens or self.max_new_tokens

        if self._is_gemma:
            messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ]

        chat_kw = {"enable_thinking": False} if self._is_qwen3 else {}

        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            **chat_kw,
        )

        if isinstance(encoded, torch.Tensor):
            input_ids      = encoded.to(self._device)
            attention_mask = torch.ones_like(input_ids)
        else:
            encoded        = encoded.to(self._device)
            input_ids      = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def step(
        self,
        state_text: str,
        system: str | None = None,
        n_buildings: int = 6,
        max_retries: int = 2,
        max_tokens: int | None = None,
        **kwargs,
    ) -> tuple[list[float], str, bool]:
        """Query the model for one environment step.

        Returns (actions, raw_response, used_fallback).
        """
        _system  = system or make_minimal_prompt(n_buildings)
        last_raw = ""

        for attempt in range(max_retries):
            last_raw = self.complete(_system, f"STATE:\n{state_text}", max_tokens=max_tokens)
            if ACTION_RE.search(last_raw):
                return parse_actions(last_raw, n_buildings), last_raw, False
            _logger.debug("No action tags (attempt %d): %s", attempt + 1, last_raw[:80])

        _logger.warning("LocalHF fallback model=%s — returning zeros", self.model_id)
        return [0.0] * n_buildings, last_raw, True
