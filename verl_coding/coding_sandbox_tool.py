# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Coding sandbox tools for agentic RL in verl.

Provides:
- SandboxTool: executes shell commands inside a Docker sandbox (via LBGSandboxManager).
  The sandbox persists across multiple tool calls within the same trajectory.
  release() is a no-op; actual cleanup is done via cleanup() or calc_reward().
- WebSearchTool: searches the open web via Bohrium API.

These tools are designed to be used with verl's ToolAgentLoop for training
models to solve scientific coding problems.
"""

import base64
import json
import logging
import os
import time
import asyncio
from typing import Any, Optional
from uuid import uuid4

import ray
import requests

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# ---------------------------------------------------------------------------
# Global sandbox concurrency quota (shared across all agent loop workers)
# ---------------------------------------------------------------------------
_SANDBOX_QUOTA_ACTOR_NAME = "verl_coding_sandbox_quota"


@ray.remote
class _SandboxQuota:
    """Cluster-wide semaphore limiting how many sandboxes exist at once.

    Permits are keyed by trajectory instance_id (idempotent acquire) and carry
    a lease TTL so that permits orphaned by dead trajectories eventually expire
    instead of deadlocking the pool.
    """

    def __init__(self, limit: int, lease_ttl: float = 19800.0):
        self._limit = limit
        self._lease_ttl = lease_ttl
        self._holders: dict[str, float] = {}

    def try_acquire(self, owner: str) -> bool:
        now = time.time()
        for key, ts in list(self._holders.items()):
            if now - ts > self._lease_ttl:
                del self._holders[key]
        if owner in self._holders:
            return True
        if len(self._holders) < self._limit:
            self._holders[owner] = now
            return True
        return False

    def release(self, owner: str) -> None:
        self._holders.pop(owner, None)

    def stats(self) -> dict:
        return {"in_use": len(self._holders), "limit": self._limit}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_IMAGES_PATH = "/trisol/input/datasets/ds-0/images.json"
BOHR_WEB_SEARCH_URL = "https://open.bohrium.com/openapi/v2/search/web"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_images(images_path: str | None = None) -> dict[str, str]:
    """Load the images.json mapping from problem domain -> Docker image."""
    path = images_path or DEFAULT_IMAGES_PATH
    if not os.path.exists(path):
        logger.warning(f"images.json not found at {path}")
        return {}
    with open(path) as f:
        return json.load(f)


def _find_image_for_domain(domain: str, images: dict[str, str]) -> str | None:
    """Find the matching Docker image for a given problem domain."""
    domain_lower = domain.lower().replace("_", "").replace("-", "")
    for key, image in images.items():
        if key.lower().replace("_", "").replace("-", "") in domain_lower:
            return image
    for key, image in images.items():
        if key.lower().split("_")[0] in domain_lower:
            return image
    return None


# ---------------------------------------------------------------------------
# SandboxTool
# ---------------------------------------------------------------------------
class SandboxTool(BaseTool):
    """Execute shell commands inside a persistent Docker sandbox.

    Uses LBGSandboxManager. One sandbox is created per trajectory on the first
    tool call and persists across all ``execute()`` calls of that trajectory:
    ``create()`` reuses the instance keyed by the trajectory's request_id and
    ``release()`` (called after each tool execution) is a no-op. Actual cleanup
    happens in ``cleanup()``, which the agent loop calls once when the
    trajectory ends.

    Concurrent sandboxes are capped cluster-wide via a named Ray actor
    (``max_concurrent_sandboxes`` in the tool config, default 16); sandbox
    creation blocks until a permit is available.

    Required ``create_kwargs``:
        image: The Docker image to use for the sandbox.
        domain: The problem domain name. Used to auto-resolve the image from
               images.json if ``image`` is not provided.

    Tool parameters (execute):
        command: The shell command to execute in the sandbox.
        timeout: Optional timeout in seconds (default: 600).
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances: dict[str, Any] = {}  # instance_id -> {manager, sandbox_id, ...}
        self._images = _load_images(config.get("images_path"))
        self._max_concurrent = int(config.get("max_concurrent_sandboxes", 16))
        self._quota = None

    def _get_quota(self):
        """Lazy-init the cluster-wide sandbox quota actor."""
        if self._quota is None:
            try:
                self._quota = ray.get_actor(_SANDBOX_QUOTA_ACTOR_NAME)
            except ValueError:
                self._quota = _SandboxQuota.options(
                    name=_SANDBOX_QUOTA_ACTOR_NAME, get_if_exists=True
                ).remote(self._max_concurrent)
        return self._quota

    def _resolve_image(self, create_kwargs: dict) -> str:
        """Resolve the Docker image from create_kwargs."""
        image = create_kwargs.get("image", "")
        domain = create_kwargs.get("domain", "")
        if not image and domain:
            image = _find_image_for_domain(domain, self._images) or ""
        return image

    def _build_sandbox(self, image: str, create_kwargs: dict) -> tuple[Any, str, str]:
        """Create a new sandbox from the given image. Returns (manager, sandbox_id, template_name)."""
        from verl_coding.lbg_sandbox_manager import LBGSandboxManager

        template_name = image.split("/")[-1].split(":")[0]
        manager = LBGSandboxManager()

        logger.debug(f"[sandbox] Creating template: {template_name}")
        manager.create_template(
            image=image,
            sku_name=create_kwargs.get("sku_name", "c4_m8_cpu"),
            name=template_name,
        )

        max_retries = create_kwargs.get("max_retries", 10)
        for attempt in range(max_retries):
            try:
                sandbox_id = manager.create_sandbox(
                    template_name=template_name,
                    never_timeout=False,
                    timeout=18000,
                )
                logger.debug(f"[sandbox] Sandbox ready: {sandbox_id}")
                return manager, sandbox_id, template_name
            except Exception as e:
                logger.warning(f"[sandbox] Attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
        raise RuntimeError(f"Failed to create sandbox after {max_retries} attempts")

    async def create(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        """Create or reuse a sandbox instance.

        The instance_id is the trajectory's request_id. Sandbox is reused
        across tool calls within the same trajectory, and cleaned up in
        cleanup() when the trajectory ends.
        """
        create_kwargs = kwargs.get("create_kwargs", {})

        # Reuse existing sandbox for this trajectory
        if instance_id and instance_id in self._instances:
            return instance_id, ToolResponse()

        if instance_id is None:
            instance_id = str(uuid4())

        image = self._resolve_image(create_kwargs)
        if not image:
            return instance_id, ToolResponse(text="[error] No Docker image specified for sandbox")

        # Throttle global sandbox concurrency; wait until a permit is available.
        quota = self._get_quota()
        while not await quota.try_acquire.remote(instance_id):
            logger.info(f"[sandbox] concurrency limit ({self._max_concurrent}) reached, waiting: {instance_id}")
            await asyncio.sleep(5)

        try:
            manager, sandbox_id, template_name = self._build_sandbox(image, create_kwargs)
            self._instances[instance_id] = {
                "manager": manager,
                "sandbox_id": sandbox_id,
                "template_name": template_name,
                "image": image,
            }

            return instance_id, ToolResponse(text=f"Sandbox created. ID: {sandbox_id}")

        except Exception as e:
            await quota.release.remote(instance_id)
            logger.error(f"[sandbox] Failed to create sandbox: {e}")
            return instance_id, ToolResponse(text=f"[error] Failed to create sandbox: {e}")

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """Execute a command in the sandbox."""
        if instance_id not in self._instances:
            return ToolResponse(text="[error] Sandbox not initialized"), 0.0, {}

        command = parameters.get("command", "")
        timeout = parameters.get("timeout", 600)

        if not command:
            return ToolResponse(text="[error] No command provided"), 0.0, {}

        manager = self._instances[instance_id]["manager"]
        try:
            output = manager.execute_command(command, background=False, timeout=timeout)
            return ToolResponse(text=output if output else "(no output)"), 0.0, {}
        except Exception as e:
            return ToolResponse(text=f"[error] {e}"), 0.0, {}

    def run_python_code(self, instance_id: str, code: str, timeout: int = 600) -> dict:
        """Run Python code in the sandbox and return structured result.

        This is used by the reward function for final evaluation.
        Returns: {success, stdout, stderr, exit_code, execution_time}
        """
        if instance_id not in self._instances:
            return {"success": False, "stdout": "", "stderr": "Sandbox not initialized", "exit_code": -1, "execution_time": 0}

        manager = self._instances[instance_id]["manager"]
        sandbox_id = self._instances[instance_id]["sandbox_id"]

        start = time.time()
        try:
            # Write code to sandbox via base64 to avoid shell escaping issues
            encoded = base64.b64encode(code.encode()).decode()
            manager.execute_command(
                f"echo {encoded} | base64 -d > /tmp/eval_solve.py",
                background=False, timeout=30,
            )

            # Run via lbg CLI with JSON output for structured result
            from verl_coding.lbg_sandbox_manager import LBGSandboxManager
            result = manager._run_command(
                ["lbg", "sdbx", "exec", "--json", "--timeout", str(timeout), sandbox_id, "python3 /tmp/eval_solve.py"],
                parse_json=True,
            )
            elapsed = time.time() - start
            if isinstance(result, dict):
                return {
                    "success": result.get("exit_code", -1) == 0,
                    "stdout": result.get("stdout", ""),
                    "stderr": result.get("stderr", ""),
                    "exit_code": result.get("exit_code", -1),
                    "execution_time": round(elapsed, 3),
                }
            return {"success": False, "stdout": "", "stderr": str(result), "exit_code": -1, "execution_time": round(elapsed, 3)}
        except Exception as e:
            elapsed = time.time() - start
            return {"success": False, "stdout": "", "stderr": str(e), "exit_code": -1, "execution_time": round(elapsed, 3)}

    async def release(self, instance_id: str, **kwargs) -> None:
        """No-op: the sandbox is shared by all tool calls within a trajectory.

        The sandbox is created on the first tool call of a trajectory and kept
        alive for subsequent calls; cleanup() destroys it when the trajectory
        ends.
        """
        return

    async def cleanup(self, instance_id: str) -> None:
        """Destroy the trajectory's sandbox and release its quota permit."""
        instance = self._instances.pop(instance_id, None)
        try:
            await self._get_quota().release.remote(instance_id)
        except Exception as e:
            logger.warning(f"[sandbox] Error releasing quota for {instance_id}: {e}")
        if instance is None:
            return
        try:
            instance["manager"].close_sandbox()
            logger.debug(f"[sandbox] Cleaned up sandbox: {instance['sandbox_id']}")
        except Exception as e:
            logger.warning(f"[sandbox] Error during cleanup: {e}")


# ---------------------------------------------------------------------------
# SandboxPool (singleton for reward computation)
# ---------------------------------------------------------------------------
class SandboxPool:
    """A singleton pool of sandboxes for reward computation.

    Since the reward function runs in a different process from the agent loop,
    it needs its own sandbox instances. This class manages sandbox creation
    and reuse to avoid creating a new sandbox for every reward computation.
    """

    _instance: Optional["SandboxPool"] = None

    def __init__(self):
        self._sandboxes: dict[str, Any] = {}  # image -> {manager, sandbox_id, template_name}
        self._images = _load_images()

    @classmethod
    def get_instance(cls) -> "SandboxPool":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_or_create(self, image: str, domain: str = "") -> tuple[str, Any]:
        """Create a new sandbox. Always creates a fresh one to avoid leaks."""
        if not image and domain:
            image = _find_image_for_domain(domain, self._images) or ""

        if not image:
            raise ValueError(f"No image found for domain '{domain}'")

        from verl_coding.lbg_sandbox_manager import LBGSandboxManager

        template_name = image.split("/")[-1].split(":")[0]
        manager = LBGSandboxManager()

        logger.debug(f"[sandbox_pool] Creating template: {template_name}")
        manager.create_template(
            image=image,
            sku_name="c4_m8_cpu",
            name=template_name,
        )

        sandbox_id = manager.create_sandbox(
            template_name=template_name,
            never_timeout=False,
            timeout=3600,
        )
        logger.debug(f"[sandbox_pool] Sandbox ready: {sandbox_id}")

        info = {
            "manager": manager,
            "sandbox_id": sandbox_id,
            "template_name": template_name,
            "image": image,
        }
        return template_name, info

    def run_code(self, sandbox_info: dict, code: str, timeout: int = 600) -> dict:
        """Run Python code in the sandbox."""
        manager = sandbox_info["manager"]
        sandbox_id = sandbox_info["sandbox_id"]

        start = time.time()
        try:
            encoded = base64.b64encode(code.encode()).decode()
            manager.execute_command(
                f"echo {encoded} | base64 -d > /tmp/eval_solve.py",
                background=False, timeout=30,
            )
            result = manager._run_command(
                ["lbg", "sdbx", "exec", "--json", "--timeout", str(timeout), sandbox_id, "python3 /tmp/eval_solve.py"],
                parse_json=True,
            )
            elapsed = time.time() - start
            if isinstance(result, dict):
                return {
                    "success": result.get("exit_code", -1) == 0,
                    "stdout": result.get("stdout", ""),
                    "stderr": result.get("stderr", ""),
                    "exit_code": result.get("exit_code", -1),
                    "execution_time": round(elapsed, 3),
                }
            return {"success": False, "stdout": "", "stderr": str(result), "exit_code": -1, "execution_time": round(elapsed, 3)}
        except Exception as e:
            elapsed = time.time() - start
            return {"success": False, "stdout": "", "stderr": str(e), "exit_code": -1, "execution_time": round(elapsed, 3)}

    def cleanup_all(self):
        """Release all sandboxes."""
        for key, info in list(self._sandboxes.items()):
            try:
                info["manager"].close_sandbox()
            except Exception as e:
                logger.warning(f"[sandbox_pool] Error cleaning up {key}: {e}")
        self._sandboxes.clear()


# ---------------------------------------------------------------------------
# WebSearchTool
# ---------------------------------------------------------------------------
class WebSearchTool(BaseTool):
    """Search the open web via the Bohrium API.

    Tool parameters:
        query: The search query string.
        num: Number of results to return (1-10, default: 5).
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        query = parameters.get("query", "")
        num = min(max(parameters.get("num", 5), 1), 10)

        if not query:
            return ToolResponse(text="[error] No search query provided"), 0.0, {}

        api_key = (
            self.config.get("bohrium_api_key")
            or os.environ.get("BOHRIUM_ACCESS_KEY", "")
        )

        if not api_key:
            return ToolResponse(text="[error] BOHRIUM_ACCESS_KEY not set"), 0.0, {}

        try:
            resp = requests.get(
                BOHR_WEB_SEARCH_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                params={"q": query, "num": num},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("data", {}).get("organic_results", [])

            if not results:
                return ToolResponse(text="(no results found)"), 0.0, {}

            lines = []
            for i, hit in enumerate(results, 1):
                title = hit.get("title", "Untitled")
                link = hit.get("link", "")
                snippet = (hit.get("snippet", "") or "")[:300]
                lines.append(f"[{i}] {title}\n    URL: {link}\n    {snippet}")

            return ToolResponse(text="\n\n".join(lines)), 0.0, {"num_results": len(results)}

        except Exception as e:
            logger.warning(f"[web_search] Error: {e}")
            return ToolResponse(text=f"[error] Web search failed: {e}"), 0.0, {}