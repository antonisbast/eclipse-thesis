"""LLM providers used by notebooks 02, 03, 05, and 06.

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
from contextlib import nullcontext
from typing import Callable

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

        # Background-thread the call so a hung endpoint can't freeze the rollout.
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="api_call"
        )
        future = executor.submit(_call)
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError(f"{self.label} did not respond within {timeout_s:.0f}s")
        finally:
            executor.shutdown(wait=False)

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

    Used by:
      • nb 03 — zero-shot SLM (loads from HF Hub via `model_id`).
      • nb 05 — post-SFT eval (wraps an Unsloth `FastModel` already in memory).
      • nb 06 — base-vs-SFT generalisation (one PEFT model in memory, two
        providers — one with `disable_adapter=True` to give the pure base).

    Two construction modes:

    1. Auto-load from HF Hub:
           LocalHFProvider(model_id="Qwen/Qwen3-4B-Instruct-2507", load_in_4bit=True)
       Loads the model via `AutoModelForCausalLM.from_pretrained` with optional
       4-bit NF4 quantization.

    2. Wrap an already-loaded model + tokenizer:
           LocalHFProvider(model=unsloth_model, tokenizer=tok,
                           model_id="unsloth/gemma-4-E4B-it",
                           prompt_builder=make_sft_prompt,
                           label="sft:gemma-4-E4B-it")
       Skips the loading path. Use when the model was loaded by something else
       (Unsloth, peft.PeftModel.from_pretrained, …). `model_id` is still used
       for the gemma / qwen3 family detection.

    Args:
        model_id:        HuggingFace model ID. Required in mode 1; in mode 2
                         it's used only for the gemma/qwen3 family flags and
                         the default `label`.
        model:           Pre-loaded HF (or PEFT) causal LM. Pass together with
                         `tokenizer` to skip auto-loading.
        tokenizer:       Pre-loaded HF tokenizer. Required if `model` is given.
        load_in_4bit:    Use 4-bit NF4 quantization (auto-load path only).
        max_new_tokens:  Max tokens generated per call (default for `.complete`).
        prompt_builder:  Callable `n_buildings -> str` returning the system
                         prompt. Defaults to `src.agent.make_minimal_prompt`
                         (CoT, canonical zero-shot prompt). Pass
                         `src.sft.make_sft_prompt` (no-CoT) when evaluating a
                         model fine-tuned on that prompt — mixing the two at
                         eval is OOD and degrades sharply.
        disable_adapter: If True and `model` is a PEFT model with an attached
                         adapter, run `with model.disable_adapter()` during
                         every `complete()` call → base-model behaviour with
                         the LoRA bypassed. Used by nb 06 to evaluate the
                         "pure" Gemma without freeing GPU memory and reloading.
        label:           Override the auto-generated label used in result
                         tables. Default: `local:<model_basename>` (or with
                         a ` (base)` suffix when `disable_adapter=True`).
    """

    def __init__(
        self,
        model_id: str | None = None,
        *,
        model=None,
        tokenizer=None,
        load_in_4bit: bool = False,
        max_new_tokens: int = 250,
        prompt_builder: Callable[[int], str] | None = None,
        disable_adapter: bool = False,
        label: str | None = None,
    ):
        import torch

        if (model is None) != (tokenizer is None):
            raise ValueError("Pass model and tokenizer together, or neither.")
        if model is None and model_id is None:
            raise ValueError("Provide model_id (auto-load) or (model, tokenizer).")

        self.max_new_tokens  = max_new_tokens
        self.prompt_builder  = prompt_builder or make_minimal_prompt
        self.disable_adapter = disable_adapter
        self.name            = "local_hf"

        if model is not None:
            # Mode 2: wrap a pre-loaded model.
            self.model     = model
            self.tokenizer = tokenizer
            self.model_id  = model_id or getattr(model, "name_or_path", "loaded_model")
            self._device   = next(model.parameters()).device
        else:
            # Mode 1: auto-load from HF Hub.
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.model_id = model_id
            self._device  = "cuda" if torch.cuda.is_available() else "cpu"

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
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            n_params = sum(p.numel() for p in self.model.parameters()) / 1e9
            mem_gb   = torch.cuda.memory_allocated() / 1e9 if self._device == "cuda" else 0.0
            print(f"  ✓ {n_params:.2f}B params | GPU mem: {mem_gb:.1f} GB")

        self.model.eval()
        self._is_qwen3 = "qwen3" in str(self.model_id).lower()
        self._is_gemma = "gemma"  in str(self.model_id).lower()

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

        # Default label — caller can override (useful in nb 06 where two
        # providers share the same underlying model).
        suffix = str(self.model_id).split('/')[-1]
        adapter_tag = " (base)" if disable_adapter else ""
        self.label = label or f"local:{suffix}{adapter_tag}"

        if self._is_qwen3:
            print("  Qwen3 detected — thinking mode disabled (enable_thinking=False)")
        if self._is_gemma:
            print("  Gemma detected — system prompt merged into user message")
        if disable_adapter:
            print("  disable_adapter=True — every forward bypasses the LoRA")

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

        # PEFT context: temporarily disable the LoRA adapter for this
        # forward when the caller asked for base-only behaviour.
        adapter_ctx = (
            self.model.disable_adapter()
            if self.disable_adapter and hasattr(self.model, "disable_adapter")
            else nullcontext()
        )
        with torch.no_grad(), adapter_ctx:
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
        _system  = system or self.prompt_builder(n_buildings)
        last_raw = ""

        for attempt in range(max_retries):
            last_raw = self.complete(_system, f"STATE:\n{state_text}", max_tokens=max_tokens)
            if ACTION_RE.search(last_raw):
                return parse_actions(last_raw, n_buildings), last_raw, False
            _logger.debug("No action tags (attempt %d): %s", attempt + 1, last_raw[:80])

        _logger.warning("LocalHF fallback model=%s — returning zeros", self.model_id)
        return [0.0] * n_buildings, last_raw, True
