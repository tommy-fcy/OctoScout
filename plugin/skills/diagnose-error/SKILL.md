---
description: Diagnose upstream Python/ML framework errors that you cannot resolve from your own knowledge. Use when you suspect a version incompatibility, API breaking change, or CUDA/driver mismatch in packages like transformers, torch, vllm, peft, accelerate, deepspeed, flash-attn, trl, LLaMA-Factory.
allowed-tools: "Bash(octoscout *), Bash(python *), Bash(pip *), Read, Glob, Grep"
---

# OctoScout — Upstream Error Diagnosis

You are a powerful code AI, but upstream ML framework version conflicts are hard to diagnose from training data alone. OctoScout fills this gap with:
- **Offline compatibility matrix** (35,000+ version pairs from 9 repos)
- **Real-time GitHub Issues search** (targeted, version-aware)
- **Local API signature checking** (via Python inspect)

## When to use OctoScout (vs your own knowledge)

**USE OctoScout** when:
- The error involves **version-specific behavior** you're unsure about (breaking changes between package versions)
- The traceback crosses **multiple ML libraries** (transformers + torch + vllm + peft + etc.)
- You need to check **specific GitHub issues** for known bugs and workarounds
- The user's installed versions are **recent releases** that may not be in your training data
- CUDA/driver/GPU compatibility issues

**DON'T use OctoScout** when:
- It's clearly a user code bug (NameError, SyntaxError, wrong variable name)
- You already know the exact fix from your training data
- It's a basic Python error unrelated to ML frameworks

## How to invoke

```bash
# Direct mode (default) — skips heuristic triage, goes straight to agent
octoscout diagnose "PASTE_TRACEBACK_HERE" --verbose

# From a file
octoscout diagnose /path/to/traceback.txt --verbose

# Pipe from failing script
python the_script.py 2>&1 | octoscout diagnose - --verbose

# Quick compatibility check
octoscout matrix check --auto-env
```

## After diagnosis

1. Present OctoScout's findings to the user
2. If a fix is suggested, offer to apply it
3. If no solution found, offer to draft a GitHub issue:
   ```bash
   # OctoScout will prompt interactively
   octoscout diagnose "traceback..." --verbose
   # → "No solution found. Draft a GitHub issue? [Y/n]"
   ```
