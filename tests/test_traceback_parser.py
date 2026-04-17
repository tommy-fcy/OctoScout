"""Tests for traceback parsing."""

from pathlib import Path

from octoscout.diagnosis.traceback_parser import parse_traceback

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_transformers_type_error():
    text = (FIXTURES / "transformers_type_error.txt").read_text()
    tb = parse_traceback(text)

    assert tb.exception_type == "TypeError"
    assert "unexpected keyword argument" in tb.exception_message
    assert "'trust_remote_code'" in tb.exception_message
    assert len(tb.frames) > 0
    assert tb.is_user_code is False  # exception in library code
    assert "transformers" in tb.involved_packages


def test_parse_vllm_cuda_mismatch():
    text = (FIXTURES / "vllm_cuda_mismatch.txt").read_text()
    tb = parse_traceback(text)

    assert tb.exception_type == "ImportError"
    assert "CUDA" in tb.exception_message or "cuda" in tb.exception_message.lower()
    assert "vllm" in tb.involved_packages
    assert tb.is_user_code is False


def test_parse_user_name_error():
    text = (FIXTURES / "user_name_error.txt").read_text()
    tb = parse_traceback(text)

    assert tb.exception_type == "NameError"
    assert "Trainr" in tb.exception_message
    assert tb.is_user_code is True
    assert len(tb.involved_packages) == 0


def test_parse_preserves_frames():
    text = (FIXTURES / "transformers_type_error.txt").read_text()
    tb = parse_traceback(text)

    # Should have multiple frames
    assert len(tb.frames) >= 3

    # Innermost frame should be in transformers
    inner = tb.frames[-1]
    assert "qwen2_vl" in inner.file
    assert inner.line == 1203

    # Outermost frame should be user code
    outer = tb.frames[0]
    assert "run_model.py" in outer.file


def test_parse_rank_prefixed_traceback():
    """Tracebacks from distributed training have [rank0]: prefixes on every line."""
    text = """\
[rank0]: Traceback (most recent call last):
[rank0]:   File "/lib/python3.12/site-packages/datasets/arrow_dataset.py", line 3635, in apply_function
[rank0]:     processed_inputs = function(*fn_args)
[rank0]:   File "/src/llamafactory/data/processor/supervised.py", line 99, in preprocess_dataset
[rank0]:     input_ids, labels = self._encode_data_example(
[rank0]: ValueError: Processor was not found, please check and update your model file.
"""
    tb = parse_traceback(text)
    assert tb.exception_type == "ValueError"
    assert "Processor was not found" in tb.exception_message
    assert len(tb.frames) == 2
    assert tb.frames[0].file == "/lib/python3.12/site-packages/datasets/arrow_dataset.py"
    assert tb.frames[1].line == 99


def test_parse_pid_prefixed_traceback():
    """Tracebacks from Ray/vLLM have (ProcessName pid=12345) prefixes."""
    text = """\
(APIServer pid=75487) Traceback (most recent call last):
(APIServer pid=75487)   File "/lib/python3.11/site-packages/vllm/api_server.py", line 10, in main
(APIServer pid=75487)     run_server()
(APIServer pid=75487) RuntimeError: CUDA out of memory
"""
    tb = parse_traceback(text)
    assert tb.exception_type == "RuntimeError"
    assert "CUDA" in tb.exception_message
    assert len(tb.frames) == 1
