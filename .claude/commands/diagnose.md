---
description: Diagnose a Python/ML error traceback using OctoScout. Identifies version incompatibilities, searches GitHub issues, and suggests fixes. Use when you encounter upstream ML framework errors.
argument-hint: "<traceback or error message>"
allowed-tools: "Bash(octoscout *), Bash(python *), Bash(pip *), Read, Glob, Grep"
---

# OctoScout Diagnose

Diagnose the following Python/ML error using OctoScout:

$ARGUMENTS

## Instructions

OctoScout runs in **direct mode** by default — it skips heuristic triage and goes straight to the ReAct agent with GitHub search + compatibility matrix.

1. If the user provided a traceback, save it to a temporary file and run:
   ```
   octoscout diagnose /path/to/traceback.txt --verbose
   ```

2. If the user provided a short error message, run directly:
   ```
   octoscout diagnose "the error message" --verbose
   ```

3. If the user points to a script that fails, run the script first to capture the traceback:
   ```
   python the_script.py 2>&1 | octoscout diagnose - --verbose
   ```

4. If you think this might be a simple local code error (NameError, SyntaxError), use `--triage` to let OctoScout's heuristic engine decide:
   ```
   octoscout diagnose "the error" --triage --verbose
   ```

5. Present the diagnosis results clearly. If OctoScout found related GitHub issues, include the links.

6. If the diagnosis suggests a fix, offer to apply it (e.g., `pip install transformers==4.52.3`).
