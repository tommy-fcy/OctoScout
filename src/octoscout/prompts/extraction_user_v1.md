Extract structured compatibility information from this GitHub issue.

## Issue: {repo}#{number}
**Title:** {title}

**Body:**
{body}

**Comments:**
{comments}

## Output JSON Schema:
{{
    "reported_versions": {{
        "package_name": "version_string"
    }},
    "python_version": "3.x.y or null if not mentioned",
    "cuda_version": "e.g. 11.8, 12.1, or null if not mentioned",
    "problem_type": "crash | wrong_output | performance | install | other",
    "error_type": "The Python exception class if applicable (TypeError, ImportError, etc.) or null",
    "error_message_summary": "One-line summary of the core error, or empty string if no error",
    "has_solution": true,
    "solution_type": "version_change | code_fix | config_change | workaround | none",
    "solution_detail": "Concise description of the fix, e.g. 'downgrade transformers to 4.52.3', or null",
    "fix_version": "The package version that contains the fix, e.g. '4.56.0', or null",
    "affected_version_range": "PEP 440 specifier, e.g. '>=4.53,<4.56', or null if unclear",
    "related_issues": ["owner/repo#123"]
}}

Example for a real issue:
{{
    "reported_versions": {{"transformers": "4.55.0", "torch": "2.3.0", "peft": "0.13.1"}},
    "python_version": "3.11.5",
    "cuda_version": "12.1",
    "problem_type": "crash",
    "error_type": "TypeError",
    "error_message_summary": "Trainer.__init__() got an unexpected keyword argument 'tokenizer'",
    "has_solution": true,
    "solution_type": "code_fix",
    "solution_detail": "Replace tokenizer=tokenizer with processing_class=tokenizer in Trainer()",
    "fix_version": null,
    "affected_version_range": ">=5.0.0",
    "related_issues": ["huggingface/transformers#43733"]
}}

Now extract from the issue above. Respond with ONLY the JSON object.
