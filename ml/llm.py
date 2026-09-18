"""
ml/llm.py
=========
Provider-agnostic LLM client for the two places DotPool uses one:

  * ml/event_ingest.py — reads messy public sources and proposes structured events
  * ml/chat_agent.py   — answers merchant questions by CALLING the forecast model

Both are *language* jobs. Neither is allowed to produce a demand number: the
LightGBM models do that, and the LLM only routes, extracts and narrates.

Three providers, no code changes needed to switch
-------------------------------------------------
    OLLAMA     — free, local, no API key, nothing leaves your machine
    OPENAI     — GPT-4o / GPT-4 Turbo, or any OpenAI-compatible endpoint
    ANTHROPIC  — Claude

Auto-detection order: whichever is configured, checking Anthropic key → OpenAI
key → a reachable Ollama server. Override with DOTPOOL_LLM_PROVIDER.

Ollama (no key, runs on your machine)
-------------------------------------
    curl -fsSL https://ollama.com/install.sh | sh     # or ollama.com/download
    ollama pull qwen2.5:7b                            # tool-calling capable
    ollama serve

    export DOTPOOL_LLM_PROVIDER=ollama
    export DOTPOOL_LLM_MODEL=qwen2.5:7b

Pick a model that supports tool calling — qwen2.5, llama3.1, llama3.2, mistral-nemo,
firefunction-v2. Models without tool support can still do event ingestion (plain
JSON), but the chat assistant needs tools to reach the forecaster.

NOTE ON STREAMLIT CLOUD: a deployed app cannot reach an Ollama server running on
your laptop — different machine. Ollama is for local runs, or point
DOTPOOL_OLLAMA_URL at a host the app can actually reach.

OpenAI / GPT-4
--------------
    export OPENAI_API_KEY=sk-...
    export DOTPOOL_LLM_PROVIDER=openai
    export DOTPOOL_LLM_MODEL=gpt-4o          # or gpt-4-turbo, gpt-4.1, gpt-4o-mini

Any OpenAI-compatible gateway (LM Studio, vLLM, Groq, Together, OpenRouter) works
by also setting OPENAI_BASE_URL.

Keys are read from the environment or Streamlit secrets — never from the repo.
On Streamlit Cloud: app → Settings → Secrets.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from ml.config import LLM_MAX_TOKENS, LLM_MODEL, LLM_PROVIDER

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4o",
    "ollama": "qwen2.5:7b",
}

# Local Ollama models known to support tool calling.
OLLAMA_TOOL_MODELS = ("qwen2.5", "qwen3", "llama3.1", "llama3.2", "llama3.3",
                      "mistral-nemo", "firefunction", "command-r", "hermes3")

_probe_cache = {"at": 0.0, "ok": False}
PROBE_TTL = 30.0


# ── Config discovery ──────────────────────────────────────────────────────────
def _secret(name: str) -> Optional[str]:
    v = os.environ.get(name)
    if v:
        return v
    try:                                        # Streamlit secrets, if running in it
        import streamlit as st
        return st.secrets.get(name)             # type: ignore[attr-defined]
    except Exception:
        return None


def ollama_url() -> str:
    return (_secret("DOTPOOL_OLLAMA_URL") or _secret("OLLAMA_HOST")
            or "http://localhost:11434").rstrip("/")


def ollama_up(timeout: float = 1.5) -> bool:
    """Is a local/remote Ollama server reachable? Cached briefly — this is called
    on every page render and a dead host must not stall the dashboard."""
    now = time.time()
    if now - _probe_cache["at"] < PROBE_TTL:
        return _probe_cache["ok"]
    ok = False
    try:
        import requests
        r = requests.get(f"{ollama_url()}/api/tags", timeout=timeout)
        ok = r.status_code == 200
    except Exception:
        ok = False
    _probe_cache.update(at=now, ok=ok)
    return ok


def ollama_models() -> list:
    try:
        import requests
        r = requests.get(f"{ollama_url()}/api/tags", timeout=2)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def provider() -> str:
    """Configured provider, or the first one that is actually usable."""
    p = (LLM_PROVIDER or "").lower().strip()
    explicit = bool(os.environ.get("DOTPOOL_LLM_PROVIDER") or
                    _secret("DOTPOOL_LLM_PROVIDER"))
    if explicit and p in DEFAULT_MODELS:
        return p
    if _secret("ANTHROPIC_API_KEY"):
        return "anthropic"
    if _secret("OPENAI_API_KEY"):
        return "openai"
    if ollama_up():
        return "ollama"
    return p if p in DEFAULT_MODELS else "anthropic"


def model_name(p: str = None) -> str:
    p = p or provider()
    return (_secret("DOTPOOL_LLM_MODEL") or LLM_MODEL or DEFAULT_MODELS[p])


def available() -> bool:
    p = provider()
    if p == "ollama":
        return ollama_up()
    return bool(_secret("ANTHROPIC_API_KEY" if p == "anthropic" else "OPENAI_API_KEY"))


def supports_tools() -> bool:
    """Ollama models vary; the chat assistant needs tool calling to be useful."""
    if provider() != "ollama":
        return True
    m = model_name().lower()
    return any(k in m for k in OLLAMA_TOOL_MODELS)


def status() -> dict:
    p = provider()
    out = {"available": available(), "provider": p, "model": model_name(p),
           "supports_tools": supports_tools(), "hint": None}
    if out["available"]:
        if p == "ollama":
            out["endpoint"] = ollama_url()
            out["installed_models"] = ollama_models()
            if not out["supports_tools"]:
                out["hint"] = (f"{out['model']} may not support tool calling — the chat "
                               f"assistant needs it. Try `ollama pull qwen2.5:7b`.")
        return out

    if p == "ollama":
        out["hint"] = (f"No Ollama server at {ollama_url()}. Install from ollama.com, "
                       f"then `ollama pull qwen2.5:7b` and `ollama serve`.")
    else:
        key = "ANTHROPIC_API_KEY" if p == "anthropic" else "OPENAI_API_KEY"
        out["hint"] = (f"Set {key} in the environment or Streamlit secrets — or run "
                       f"Ollama locally for a free, keyless option "
                       f"(DOTPOOL_LLM_PROVIDER=ollama).")
    out["options"] = {
        "ollama": "free, local, no key — ollama.com, then pull qwen2.5:7b",
        "openai": "export OPENAI_API_KEY and DOTPOOL_LLM_MODEL=gpt-4o",
        "anthropic": "export ANTHROPIC_API_KEY",
    }
    return out


class LLMUnavailable(RuntimeError):
    pass


# ── Unified call ──────────────────────────────────────────────────────────────
def chat(messages: list, system: str = "", tools: list = None,
         max_tokens: int = LLM_MAX_TOKENS, temperature: float = 0.0) -> dict:
    """One turn of conversation, normalised across providers.

    Returns {"text": str, "tool_calls": [{"id", "name", "input"}], "raw": ...}
    so callers never branch on the provider.
    """
    if not available():
        raise LLMUnavailable(status()["hint"])
    p = provider()
    model = model_name(p)
    if p == "anthropic":
        return _anthropic(messages, system, tools, model, max_tokens, temperature)
    if p == "ollama":
        return _ollama(messages, system, tools, model, max_tokens, temperature)
    return _openai(messages, system, tools, model, max_tokens, temperature)


def _anthropic(messages, system, tools, model, max_tokens, temperature) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=_secret("ANTHROPIC_API_KEY"))
    kwargs = dict(model=model, max_tokens=max_tokens, temperature=temperature,
                  messages=messages)
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = [
            {"name": t["name"], "description": t["description"],
             "input_schema": t["input_schema"]} for t in tools
        ]
    resp = client.messages.create(**kwargs)
    text, calls = "", []
    for block in resp.content:
        if block.type == "text":
            text += block.text
        elif block.type == "tool_use":
            calls.append({"id": block.id, "name": block.name, "input": block.input})
    return {"text": text, "tool_calls": calls, "raw": resp,
            "stop_reason": resp.stop_reason}


def _openai(messages, system, tools, model, max_tokens, temperature) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=_secret("OPENAI_API_KEY"),
                    base_url=_secret("OPENAI_BASE_URL") or None)
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    kwargs = dict(model=model, messages=msgs, max_tokens=max_tokens,
                  temperature=temperature)
    if tools:
        kwargs["tools"] = [
            {"type": "function",
             "function": {"name": t["name"], "description": t["description"],
                          "parameters": t["input_schema"]}} for t in tools
        ]
    resp = client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    calls = [
        {"id": tc.id, "name": tc.function.name,
         "input": json.loads(tc.function.arguments or "{}")}
        for tc in (msg.tool_calls or [])
    ]
    return {"text": msg.content or "", "tool_calls": calls, "raw": resp,
            "stop_reason": resp.choices[0].finish_reason}


def _ollama(messages, system, tools, model, max_tokens, temperature) -> dict:
    """Local Ollama via /api/chat. Same tool-call shape as OpenAI, but arguments
    arrive as a dict rather than a JSON string, and ids are not returned."""
    import requests

    msgs = ([{"role": "system", "content": system}] if system else []) + [
        _plain(m) for m in messages
    ]
    payload = {
        "model": model, "messages": msgs, "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
        # Keep the model resident between calls — a tool-calling conversation
        # is several requests in a row, and without this Ollama can unload and
        # reload the weights between them, which is most of what makes a local
        # run feel slow. 30m is generous enough to cover a whole chat session.
        "keep_alive": "30m",
    }
    if tools:
        payload["tools"] = [
            {"type": "function",
             "function": {"name": t["name"], "description": t["description"],
                          "parameters": t["input_schema"]}} for t in tools
        ]
    r = requests.post(f"{ollama_url()}/api/chat", json=payload, timeout=180)
    if r.status_code == 404:
        raise LLMUnavailable(
            f"Ollama has no model called {model!r}. Run `ollama pull {model}` "
            f"(installed: {', '.join(ollama_models()) or 'none'}).")
    r.raise_for_status()
    data = r.json()
    msg = data.get("message", {}) or {}
    calls = []
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function", {})
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except json.JSONDecodeError:
                args = {}
        calls.append({"id": f"call_{i}", "name": fn.get("name"), "input": args or {}})
    return {"text": msg.get("content") or "", "tool_calls": calls, "raw": data,
            "stop_reason": data.get("done_reason")}


def _plain(m: dict) -> dict:
    """Flatten a structured message to text — Ollama takes plain strings."""
    c = m.get("content")
    if isinstance(c, str):
        return {"role": m["role"], "content": c}
    parts = []
    for b in c or []:
        if not isinstance(b, dict):
            parts.append(str(b))
        elif b.get("type") == "text":
            parts.append(b["text"])
        elif b.get("type") == "tool_use":
            parts.append(f"[called {b['name']} with {json.dumps(b['input'])}]")
        elif b.get("type") == "tool_result":
            parts.append(f"[tool result] {b['content']}")
    return {"role": m["role"], "content": "\n".join(parts)}


def complete_json(prompt: str, system: str = "", max_tokens: int = LLM_MAX_TOKENS):
    """Ask for JSON and parse it, tolerating ```json fences and stray prose."""
    out = chat([{"role": "user", "content": prompt}], system=system,
               max_tokens=max_tokens)
    txt = (out["text"] or "").strip()
    if "```" in txt:
        seg = txt.split("```")[1]
        txt = seg[4:] if seg.startswith("json") else seg
    starts = [i for i in (txt.find("["), txt.find("{")) if i != -1]
    if not starts:
        raise ValueError(f"model did not return JSON: {txt[:200]!r}")
    txt = txt[min(starts):]
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        # Smaller local models often trail off — salvage the complete objects.
        depth, last = 0, None
        for i, ch in enumerate(txt):
            depth += ch in "[{"
            depth -= ch in "]}"
            if depth == 0 and i:
                last = i + 1
                break
        if last:
            return json.loads(txt[:last])
        raise
