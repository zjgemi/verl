# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

"""
Custom reward function for scientific coding problems.

Extracts the final answer from the model's response, executes the code in a
**Docker sandbox** (via LBGSandboxManager), and compares the output with the
ground truth using value_check.

The sandbox ensures the code runs in the correct environment with all required
scientific computing packages pre-installed.

Supports nested data structures (lists, dicts, numbers) with configurable
floating-point tolerance.

Usage:
  Register in verl/utils/reward_score/__init__.py:
    data_source="coding_practice" -> coding_reward.compute_score
"""

import ast
import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def string2obj(answer: str) -> Any:
    """Parse the last line of output / ground truth into a Python object."""
    if not answer:
        return answer.strip() if isinstance(answer, str) else answer
    if isinstance(answer, str):
        answer = answer.strip()
        if not answer:
            return answer
    try:
        if isinstance(answer, str):
            lines = answer.splitlines()
            if lines:
                answer = lines[-1]
        answer = answer.replace("np.nan", "nan")
        answer = re.sub(r"np\.\w+\(([^)]+)\)", r"\1", answer)
        answer = answer.replace("true", "True").replace("false", "False").replace("nan", "None").strip()
        return ast.literal_eval(answer)
    except (ValueError, SyntaxError):
        return answer.strip() if isinstance(answer, str) else answer


def value_check(a: Any, b: Any, tol: float = 0.01, atol: float = 1e-6) -> bool:
    """Compare two values with nested structure support and numeric tolerance.

    ``tol`` is relative, ``atol`` is the absolute floor. Without the floor a ground
    truth of 0 collapses the relative bound to 0, so any float round-off
    (e.g. pred 2.2e-16 vs gt 0.0) is scored wrong.
    """
    if a is None or b is None:
        return False
    if isinstance(a, (int, float, complex)) and isinstance(b, (int, float, complex)):
        if abs(a) + abs(b) == 0:
            return True
        return abs(a - b) <= atol + max(abs(a), abs(b)) * tol
    if isinstance(a, bool) and isinstance(b, bool):
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(value_check(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(value_check(x, y, tol) for x, y in zip(a, b))
    return a == b


def extract_final_answer(text: str) -> tuple[str | None, str | None]:
    """Extract <final_answer> code and <result> from the model's response."""
    fa_matches = list(re.finditer(r"<final_answer>(.*?)</final_answer>", text, re.DOTALL))
    if not fa_matches:
        return None, None
    fa_match = fa_matches[-1]
    code = fa_match.group(1).strip()
    r_matches = list(re.finditer(r"<result>(.*?)</result>", text, re.DOTALL))
    result = r_matches[-1].group(1).strip() if r_matches else None
    return code, result


def extract_python_code(text: str) -> str | None:
    """Extract Python code from a markdown fenced code block."""
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\w*\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```python\s*\n(.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def run_code_in_sandbox(code: str, image: str, domain: str, timeout: int = 600) -> dict:
    """Run Python code in a Docker sandbox via LBGSandboxManager.

    Uses SandboxPool to reuse sandboxes across reward computations.

    Args:
        code: Python source code to execute.
        image: Docker image URL for the sandbox.
        domain: Problem domain name (fallback if image is empty).
        timeout: Maximum execution time in seconds.

    Returns:
        dict with keys: success, stdout, stderr, exit_code, execution_time.
    """
    try:
        from verl_coding.coding_sandbox_tool import SandboxPool

        pool = SandboxPool.get_instance()
        _, sandbox_info = pool.get_or_create(image=image, domain=domain)
        result = pool.run_code(sandbox_info, code, timeout=timeout)
        # Clean up sandbox after use
        sandbox_info["manager"].close_sandbox()
        return result

    except ImportError as e:
        logger.warning(f"Cannot import SandboxPool ({e}), falling back to subprocess")
        return _run_code_in_subprocess(code, timeout)
    except Exception as e:
        logger.warning(f"Sandbox execution failed ({e}), falling back to subprocess")
        return _run_code_in_subprocess(code, timeout)


def _run_code_in_subprocess(code: str, timeout: int = 600) -> dict:
    """Fallback: run Python code in a subprocess."""
    import subprocess
    import sys
    import time

    start = time.time()
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout,
        )
        elapsed = time.time() - start
        return {
            "success": r.returncode == 0,
            "stdout": r.stdout,
            "stderr": r.stderr,
            "exit_code": r.returncode,
            "execution_time": round(elapsed, 3),
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": "TIMEOUT", "exit_code": -1, "execution_time": timeout}


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    """Compute reward score for a coding problem.

    Flow:
    1. Extract <final_answer> code block and <result> from model response
    2. Run the extracted Python code in a Docker sandbox (with subprocess fallback)
    3. Parse the last line of stdout as the predicted answer
    4. Parse ground_truth (JSON string) as the expected answer
    5. Compare using value_check

    Args:
        data_source: "coding_practice"
        solution_str: Full decoded model response.
        ground_truth: JSON-encoded ground truth answer.
        extra_info: Must contain ``tools_kwargs.execute_command.create_kwargs``
                    with ``image`` and ``domain`` for sandbox creation.

    Returns:
        1.0 if correct, 0.0 otherwise.
    """
    extra_info = extra_info or {}

    # Step 1: Extract final answer code
    final_code, _ = extract_final_answer(solution_str)
    if final_code is None:
        return 0.0

    # Step 2: Extract pure Python code
    python_code = extract_python_code(final_code)
    if python_code is None:
        return 0.0

    # Step 3: Run code in sandbox (or subprocess fallback)
    tools_kwargs = extra_info.get("tools_kwargs", {})
    sandbox_kwargs = tools_kwargs.get("execute_command", {}).get("create_kwargs", {})
    image = sandbox_kwargs.get("image", "")
    domain = sandbox_kwargs.get("domain", "")

    exec_result = run_code_in_sandbox(python_code, image=image, domain=domain)

    if not exec_result["success"]:
        return 0.0

    # Step 4: Parse output and ground truth
    predicted = string2obj(exec_result["stdout"])
    try:
        expected = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    except json.JSONDecodeError:
        expected = string2obj(ground_truth)

    # Step 5: Compare
    try:
        correct = value_check(predicted, expected)
    except Exception as e:
        return 0.0

    return 1.0 if correct else 0.0