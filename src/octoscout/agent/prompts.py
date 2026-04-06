"""System prompts and few-shot examples for the diagnosis agent.

Prompts are loaded from src/octoscout/prompts/*.md files.
Edit those files to change prompt behavior without modifying code.
"""

from octoscout.prompts import load_prompt

SYSTEM_PROMPT = load_prompt("diagnosis_system")
DIAGNOSIS_REPORT_FORMAT = load_prompt("diagnosis_report_format")
