"""
Shared LLM status panel used by the chat and event-review pages.
pages/_llm_sidebar.py  (leading underscore keeps it out of Streamlit's page list)
"""

import streamlit as st

from ml.llm import ollama_models, ollama_url, status


def llm_sidebar() -> dict:
    """Render provider status + setup instructions. Returns the status dict."""
    llm = status()
    with st.sidebar:
        st.markdown("###  Language model")

        if llm["available"]:
            st.success(f"{llm['provider']} · {llm['model']}")
            if llm["provider"] == "ollama":
                st.caption(f"Local — {llm.get('endpoint', ollama_url())}. "
                           "Nothing leaves this machine.")
                models = llm.get("installed_models") or []
                if models:
                    st.caption("Installed: " + ", ".join(models[:6]))
            if llm.get("hint"):
                st.warning(llm["hint"])
            if not llm.get("supports_tools", True):
                st.warning("This model may not support tool calling, which the chat "
                           "assistant needs to reach the forecaster.")
        else:
            st.error("No language model configured")
            st.caption(llm.get("hint", ""))

        with st.expander("Set one up", expanded=not llm["available"]):
            st.markdown("**Ollama — free, local, no API key**")
            st.code("# ollama.com/download\n"
                    "ollama pull qwen2.5:7b\n"
                    "ollama serve\n\n"
                    "export DOTPOOL_LLM_PROVIDER=ollama\n"
                    "export DOTPOOL_LLM_MODEL=qwen2.5:7b", language="bash")
            st.caption("Pick a tool-calling model: qwen2.5, llama3.1/3.2, mistral-nemo. "
                       "A deployed Streamlit app can't reach Ollama on your laptop — "
                       "local runs only, unless you expose it at a reachable URL "
                       "via DOTPOOL_OLLAMA_URL.")

            st.markdown("**OpenAI — GPT-4o / GPT-4 Turbo**")
            st.code("export OPENAI_API_KEY=sk-...\n"
                    "export DOTPOOL_LLM_PROVIDER=openai\n"
                    "export DOTPOOL_LLM_MODEL=gpt-4o", language="bash")

            st.markdown("**Anthropic — Claude**")
            st.code("export ANTHROPIC_API_KEY=sk-ant-...", language="bash")

            st.markdown("**On Streamlit Cloud** — app → Settings → Secrets:")
            st.code('OPENAI_API_KEY = "sk-..."\n'
                    'DOTPOOL_LLM_PROVIDER = "openai"\n'
                    'DOTPOOL_LLM_MODEL = "gpt-4o"', language="toml")

        if st.button("Re-check connection", use_container_width=True):
            from ml.llm import _probe_cache
            _probe_cache["at"] = 0.0            # bust the Ollama probe cache
            st.rerun()
    return llm
