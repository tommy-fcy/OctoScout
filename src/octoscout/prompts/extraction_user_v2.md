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

Example 1 — Multi-package crash with fix:
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

Example 2 — Single package, workaround found in comments (fix_version is null because downgrade is not a fix):
{{
    "reported_versions": {{"transformers": "4.55.0"}},
    "python_version": null,
    "cuda_version": null,
    "problem_type": "wrong_output",
    "error_type": null,
    "error_message_summary": "Qwen2.5-VL outputs garbage text 'addCriterion' mixed with responses",
    "has_solution": true,
    "solution_type": "workaround",
    "solution_detail": "Downgrade transformers to 4.52.3, or set padding_side='left' in processor",
    "fix_version": null,
    "affected_version_range": ">=4.53",
    "related_issues": []
}}

Example 3 — No version mentioned, but repo is the package:
{{
    "reported_versions": {{"vllm": null}},
    "python_version": null,
    "cuda_version": null,
    "problem_type": "crash",
    "error_type": "RuntimeError",
    "error_message_summary": "CUDA out of memory when loading 70B model with tensor parallelism",
    "has_solution": false,
    "solution_type": "none",
    "solution_detail": null,
    "fix_version": null,
    "affected_version_range": null,
    "related_issues": []
}}

Now extract from the issue above. Respond with ONLY the JSON object.
