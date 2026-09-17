# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Teacher client for a model served outside the verl Ray cluster.

Unlike the student's rollout engine, a distillation teacher never has its weights
synchronized: it is a frozen forward pass. So it does not need to live inside the
Ray cluster at all. This client talks to any OpenAI-compatible server that accepts
a token-id array as ``prompt`` and returns ``prompt_logprobs`` (vLLM does), which
decouples the teacher's inference stack from the student rollout's.

The output is shaped to the same contract as
``verl.workers.rollout.vllm_rollout.utils.extract_prompt_logprobs`` so that
``AsyncTeacherLLMServerManager`` cannot tell the two paths apart.
"""

import asyncio
import logging
import os
from typing import Any, Optional

import aiohttp

from verl.workers.config import DistillationExternalTeacherConfig
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def shape_prompt_logprobs(
    raw_prompt_logprobs: list[Optional[dict[str, Any]]],
    num_prompt_logprobs: int,
) -> tuple[list[list[int]], list[list[float]]]:
    """Reshape an OpenAI-server ``prompt_logprobs`` payload into verl's teacher contract.

    The server returns one entry per prompt position, the first being ``None`` (the
    first token has no conditional logprob), each remaining entry a dict keyed by
    token id (JSON turns the int key into a string) with ``logprob`` and ``rank``.

    Returns ``(prompt_ids, prompt_logprobs)``, both ``[S, max(k, 1)]`` where ``S``
    is the prompt length: position ``i`` holds the logprob of token ``i + 1``, and
    the last row is a dummy so the length matches the input sequence. This mirrors
    ``verl.workers.rollout.vllm_rollout.utils.extract_prompt_logprobs``.
    """
    prompt_ids_ls: list[list[int]] = []
    prompt_logprobs_ls: list[list[float]] = []

    # NOTE: logprob of first prompt token is None.
    for position, logprobs_dict in enumerate(raw_prompt_logprobs[1:], start=1):
        if logprobs_dict is None:
            raise ValueError(f"External teacher returned a null prompt_logprobs entry at position {position}.")
        if num_prompt_logprobs == 0:
            token_id_str = next(iter(logprobs_dict))
            prompt_ids_ls.append([int(token_id_str)])
            prompt_logprobs_ls.append([float(logprobs_dict[token_id_str]["logprob"])])
        else:
            ids: list[Optional[int]] = [None] * num_prompt_logprobs
            logprobs: list[Optional[float]] = [None] * num_prompt_logprobs
            # Either the top-k logprobs, or the top-k plus the sampled token when it
            # is not itself in the top-k.
            if len(logprobs_dict) not in (num_prompt_logprobs, num_prompt_logprobs + 1):
                raise ValueError(
                    f"External teacher returned {len(logprobs_dict)} logprobs at position {position}, "
                    f"expected {num_prompt_logprobs} or {num_prompt_logprobs + 1}."
                )
            for token_id_str, token_logprob in logprobs_dict.items():
                rank = token_logprob["rank"]
                if rank > num_prompt_logprobs:
                    continue  # the sampled token is not in the top-k
                ids[rank - 1] = int(token_id_str)
                logprobs[rank - 1] = float(token_logprob["logprob"])
            prompt_ids_ls.append(ids)
            prompt_logprobs_ls.append(logprobs)

    # NOTE: pad a dummy prompt logprob for last prompt token.
    width = max(num_prompt_logprobs, 1)
    prompt_ids_ls.append([0] * width)
    prompt_logprobs_ls.append([0.0] * width)
    return prompt_ids_ls, prompt_logprobs_ls


class ExternalTeacherRequestError(RuntimeError):
    """An HTTP error response from the external teacher, carrying its status code.

    The status is kept as a field rather than parsed back out of the message, whose
    tail is the server's response body and could contain anything.
    """

    def __init__(self, status: int, body: str, url: str):
        super().__init__(f"HTTP {status} from external teacher {url}: {body}")
        self.status = status


class ExternalTeacherClient:
    """Drop-in replacement for ``LLMServerClient`` backed by an external HTTP service.

    Only ``generate`` is implemented: that is the entire surface
    ``AsyncTeacherLLMServerManager`` uses.
    """

    def __init__(self, config: DistillationExternalTeacherConfig):
        if not config.base_url:
            raise ValueError("ExternalTeacherClient requires external.base_url to be set.")
        self.config = config
        self.url = config.base_url.rstrip("/") + "/v1/completions"
        self.model_name = config.model_name
        self._api_key = config.resolve_api_key()
        if not self._api_key:
            logger.warning(
                "No API key for external teacher %s (checked external.api_key and $%s); "
                "sending requests unauthenticated.",
                self.url,
                config.api_key_env,
            )
        # Created lazily: this object is constructed on the driver but used inside
        # agent-loop workers, where a different event loop owns the connections.
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_s),
                connector=aiohttp.TCPConnector(limit=0),
            )
        return self._session

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.config.max_concurrency)
        return self._semaphore

    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TokenOutput:
        """Score ``prompt_ids`` with the external teacher and return its prompt logprobs.

        The teacher generates a single throwaway token; the payload we want rides
        along in ``prompt_logprobs``.
        """
        if image_data or video_data or audio_data or mm_processor_kwargs:
            raise NotImplementedError(
                "External distillation teacher does not support multi-modal inputs; "
                "colocate the teacher (external.base_url=null) to use them."
            )

        num_prompt_logprobs = sampling_params.get("prompt_logprobs")
        if num_prompt_logprobs is None:
            raise ValueError("External teacher expects sampling_params['prompt_logprobs'] to be set.")

        payload = {
            "model": self.model_name,
            "prompt": prompt_ids,
            "max_tokens": sampling_params.get("max_tokens", 1),
            "temperature": sampling_params.get("temperature", 1.0),
            "prompt_logprobs": num_prompt_logprobs,
        }

        async with self._get_semaphore():
            response = await self._post_with_retry(payload, request_id=request_id)

        raw_prompt_logprobs = response["choices"][0].get("prompt_logprobs")
        if raw_prompt_logprobs is None:
            raise ValueError(
                f"External teacher {self.url} returned no prompt_logprobs. The server must be "
                f"OpenAI-compatible with prompt_logprobs support (e.g. vLLM)."
            )
        if len(raw_prompt_logprobs) != len(prompt_ids):
            raise ValueError(
                f"External teacher returned {len(raw_prompt_logprobs)} prompt_logprobs for "
                f"{len(prompt_ids)} prompt tokens. A tokenizer mismatch or server-side prompt "
                f"truncation would cause this."
            )

        prompt_ids_ls, prompt_logprobs_ls = shape_prompt_logprobs(raw_prompt_logprobs, num_prompt_logprobs)
        return TokenOutput(
            token_ids=[],
            extra_fields={"prompt_ids": prompt_ids_ls, "prompt_logprobs": prompt_logprobs_ls},
        )

    async def _post_with_retry(self, payload: dict[str, Any], request_id) -> dict[str, Any]:
        session = self._get_session()
        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                async with session.post(self.url, json=payload) as response:
                    if response.status != 200:
                        body = (await response.text())[:512]
                        raise ExternalTeacherRequestError(response.status, body, self.url)
                    return await response.json()
            except (aiohttp.ClientError, asyncio.TimeoutError, ExternalTeacherRequestError) as e:
                if isinstance(e, ExternalTeacherRequestError) and 400 <= e.status < 500:
                    # A 4xx will not fix itself on retry: context length exceeded, an
                    # unknown model name, prompt_logprobs above the server's cap.
                    raise
                last_error = e
                if attempt == self.config.max_retries:
                    break
                delay = self.config.retry_backoff_s * (2**attempt)
                logger.warning(
                    "External teacher request %s failed (attempt %d/%d): %s. Retrying in %.1fs.",
                    request_id,
                    attempt + 1,
                    self.config.max_retries + 1,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)
        raise RuntimeError(
            f"External teacher request {request_id} failed after {self.config.max_retries + 1} attempts: {last_error}"
        ) from last_error
