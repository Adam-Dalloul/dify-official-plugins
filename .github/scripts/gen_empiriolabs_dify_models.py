#!/usr/bin/env python3
"""Generate Dify LLM model YAMLs for ALL EmpirioLabs text chat models from the
public EmpirioLabs /v1/models catalog.

Dify model-provider plugins cannot fetch a model list at runtime (the host
plugin protocol has no list-models RPC and the legacy fetch-from-remote method
was removed), so the picker is populated from static predefined YAML. This
script regenerates that YAML from the live, public, no-auth catalog, so keeping
the Dify plugin current is one command (or a scheduled CI run), not hand work.

Usage:
    python dify_gen_empiriolabs_models.py <plugin_root>
where <plugin_root> is the path to the dify plugin dir (the one that contains
models/llm/), e.g. models/empiriolabs

No credentials are used or needed: https://api.empiriolabs.ai/v1/models is public.
"""
import json
import os
import sys
import urllib.request

API = os.environ.get("EMPIRIOLABS_MODELS_URL", "https://api.empiriolabs.ai/v1/models")


def fetch():
    req = urllib.request.Request(API, headers={"User-Agent": "empiriolabs-dify-gen"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["data"]


def price_per_m(tier_val):
    if tier_val is None:
        return None
    try:
        per_token = float(tier_val)
    except (TypeError, ValueError):
        return None
    if per_token < 0:
        return None
    s = f"{per_token * 1_000_000:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def features_for(model):
    feats = set(model.get("features") or [])
    caps = model.get("capabilities") or {}
    inp = [m.lower() for m in (model.get("input_modalities") or [])]
    out = []
    if caps.get("reasoning") or "reasoning" in feats:
        out.append("agent-thought")
    if "image" in inp:
        out.append("vision")
    if "video" in inp:
        out.append("video")
    if "audio" in inp:
        out.append("audio")
    if "function_calling" in feats or "tools" in feats:
        out += ["tool-call", "multi-tool-call", "stream-tool-call"]
    return out


def yaml_for(model):
    mid = model["id"]
    label = model.get("display_name") or mid
    context = int(model.get("context_length") or model.get("context_window") or 8192)
    max_out = int(model.get("max_output_tokens") or 8192)
    reasoning = bool((model.get("capabilities") or {}).get("reasoning"))
    pricing = model.get("pricing")
    if isinstance(pricing, list):
        tier = pricing[0] if pricing else {}
    elif isinstance(pricing, dict):
        tier = pricing
    else:
        tier = {}
    inp = price_per_m(tier.get("prompt")) or "0"
    outp = price_per_m(tier.get("completion")) or "0"

    lines = [f"model: {mid}", "label:", f"  en_US: {label}", "model_type: llm"]
    feats = features_for(model)
    if feats:
        lines.append("features:")
        lines += [f"  - {f}" for f in feats]
    lines += ["model_properties:", "  mode: chat", f"  context_size: {context}",
              "parameter_rules:",
              "  - name: temperature", "    use_template: temperature",
              "  - name: top_p", "    use_template: top_p",
              "  - name: max_tokens", "    use_template: max_tokens", "    type: int",
              "    default: 1024", "    min: 1", f"    max: {max_out}",
              "  - name: frequency_penalty", "    use_template: frequency_penalty",
              "  - name: presence_penalty", "    use_template: presence_penalty",
              "  - name: response_format", "    label:", "      en_US: Response Format",
              "    type: string", "    required: false", "    options:",
              "      - text", "      - json_object"]
    if reasoning:
        lines += ["  - name: enable_thinking", "    label:", "      en_US: Enable Thinking",
                  "    type: boolean", "    default: false", "    required: false",
                  "    help:", "      en_US: Enable step-by-step reasoning for the model.",
                  "  - name: reasoning_effort", "    label:", "      en_US: Reasoning Effort",
                  "    type: string", "    required: false", "    options:"]
        lines += [f"      - {v}" for v in ["none", "low", "medium", "high", "max"]]
    lines += ["pricing:", f"  input: '{inp}'", f"  output: '{outp}'",
              "  unit: '0.000001'", "  currency: USD"]
    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: dify_gen_empiriolabs_models.py <plugin_root>")
    root = sys.argv[1]
    llm_dir = os.path.join(root, "models", "llm")
    if not os.path.isdir(llm_dir):
        sys.exit(f"plugin llm dir not found: {llm_dir}")

    data = fetch()
    text_models = [m for m in data if m.get("category") == "text" and ":" not in m["id"]]

    # Remove stale generated YAMLs (any maker subdir), keep llm.py/_position/__init__.
    for maker in os.listdir(llm_dir):
        sub = os.path.join(llm_dir, maker)
        if os.path.isdir(sub):
            for f in os.listdir(sub):
                if f.endswith(".yaml"):
                    os.remove(os.path.join(sub, f))

    ids_by_maker = {}
    for m in text_models:
        maker = (m.get("provider") or "other").lower().strip() or "other"
        d = os.path.join(llm_dir, maker)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, m["id"] + ".yaml"), "w", encoding="utf-8", newline="\n") as f:
            f.write(yaml_for(m))
        ids_by_maker.setdefault(maker, []).append(m["id"])

    pos = []
    for maker in sorted(ids_by_maker):
        pos.append(f"# {maker} models ({len(ids_by_maker[maker])})")
        pos += [f"- {mid}" for mid in sorted(ids_by_maker[maker])]
        pos.append("")
    with open(os.path.join(llm_dir, "_position.yaml"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(pos))

    print(f"generated {len(text_models)} model YAMLs across {len(ids_by_maker)} makers")


if __name__ == "__main__":
    main()
