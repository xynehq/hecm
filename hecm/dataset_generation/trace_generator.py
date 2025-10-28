#!/usr/bin/env python3
"""
hecm/dataset_generation/trace_generator.py

Generate traces by driving agents (local vLLM or remote) and save them as JSONL.

Usage:
    python3 hecm/dataset_generation/trace_generator.py \
        --model archit11/qwen-30b-hyperswitch-v1 \
        --out traces.jsonl

Make sure to import the runtime patch before importing create_agent; we do that here.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, Iterable, List, Optional

from cursor_agent_tools.factory import create_agent

# IMPORTANT: patch must be imported before any module that imports create_agent.
# This ensures the factory is patched and archit11/... models will return LocalVLLMAgent.
import hecm.runtime_patches.local_vllm  # noqa: E402  (must be first)

# --- logging setup ---
logging.basicConfig(level=os.getenv("TRACE_LOG_LEVEL", "INFO"))
logger = logging.getLogger("hecm.trace_generator")


# --- Helper: write JSON Lines safely ---
def append_jsonl(path: str, records: Iterable[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(
        "Appended %d records to %s", sum(1 for _ in records), path
    )  # note: sum will be 0 here


# --- Trace generator class ---
class AgentTraceGenerator:
    """
    Generate conversational traces by driving an agent with prompts.
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        timeout: int = 180,
        default_tool_timeout: int = 300,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout
        self.default_tool_timeout = default_tool_timeout

    def _create_agent(self):
        """
        Create an agent through cursor_agent_tools.factory.create_agent.
        The runtime patch will return LocalVLLMAgent for archit11/ models.
        """
        kwargs = {
            "temperature": self.temperature,
            "timeout": self.timeout,
            "default_tool_timeout": self.default_tool_timeout,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        logger.info(
            "Creating agent for model %s (kwargs=%s)",
            self.model,
            {k: v for k, v in kwargs.items() if k != "timeout"},
        )
        return create_agent(model=self.model, **kwargs)

    async def generate_trace(
        self, prompts: List[str], save_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Run the agent on the list of prompts and return list of trace dicts.
        If save_path provided, append to that file as JSONL.
        """
        agent = self._create_agent()
        traces: List[Dict[str, Any]] = []

        for idx, prompt in enumerate(prompts):
            logger.info("Running prompt %d/%d", idx + 1, len(prompts))
            start_ts = time.time()
            try:
                # prefer async chat if available
                if asyncio.iscoroutinefunction(getattr(agent, "chat", None)):
                    resp = await agent.chat(prompt)
                else:
                    # some agents expose sync chat_sync or chat_sync wrapper
                    if hasattr(agent, "chat_sync"):
                        resp = agent.chat_sync(prompt)
                    else:
                        # fallback: try calling .chat synchronously
                        resp = agent.chat(prompt)
                duration = time.time() - start_ts

                trace = {
                    "timestamp": int(start_ts),
                    "duration_s": duration,
                    "model": self.model,
                    "prompt": prompt,
                    "response": resp.get("message")
                    if isinstance(resp, dict)
                    else str(resp),
                    "raw": resp.get("raw") if isinstance(resp, dict) else resp,
                }
                logger.debug(
                    "Trace produced: %s",
                    trace["response"][:200] if trace["response"] else "<empty>",
                )
            except Exception as exc:
                duration = time.time() - start_ts
                logger.exception("Error while executing prompt: %s", exc)
                trace = {
                    "timestamp": int(start_ts),
                    "duration_s": duration,
                    "model": self.model,
                    "prompt": prompt,
                    "response": None,
                    "raw": {"error": str(exc)},
                }

            traces.append(trace)

            # optionally flush after each prompt to disk
            if save_path:
                # append single record (open/close is fine for small runs; optimize if necessary)
                with open(save_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(trace, ensure_ascii=False) + "\n")

        return traces


# --- Utility: sample prompts if none provided ---
DEFAULT_PROMPTS = [
    "Summarize the tradeoffs between LRU and LFU caching in one short paragraph.",
    "Given a Python function that fetches data from DB, how would you add a simple in-process cache?",
    "Write a small code snippet using functools.lru_cache for an I/O bound function.",
    "Describe when to use Redis as cache instead of in-memory caching.",
    "What are the main challenges of cache invalidation in distributed systems?",
]


# --- CLI + main ---
async def main_async(args):
    # load prompts: if a file provided, read lines, else use defaults
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as fh:
            prompts = [line.strip() for line in fh if line.strip()]
        if not prompts:
            logger.warning(
                "Prompts file provided but no prompts found. Falling back to defaults."
            )
            prompts = DEFAULT_PROMPTS
    else:
        prompts = DEFAULT_PROMPTS

    # optionally repeat prompts to reach N conversations
    if args.count and args.count > 1:
        prompts = (prompts * ((args.count + len(prompts) - 1) // len(prompts)))[
            : args.count
        ]

    tracer = AgentTraceGenerator(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        timeout=args.timeout,
        default_tool_timeout=args.default_tool_timeout,
    )

    # ensure output file exists (create/truncate if requested)
    if args.overwrite and args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            pass

    traces = await tracer.generate_trace(prompts, save_path=args.out)
    logger.info("Generated %d traces", len(traces))
    if not args.out:
        # if not saved to a file, print the traces
        print(json.dumps(traces, ensure_ascii=False, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description="Generate agent traces for testing/eval")
    p.add_argument(
        "--model",
        type=str,
        default=os.getenv("TRACE_MODEL", "archit11/qwen-30b-hyperswitch-v1"),
        help="Agent model name",
    )
    p.add_argument(
        "--base-url",
        dest="base_url",
        type=str,
        default=os.getenv("LOCAL_VLLM_URL", None),
        help="Optional base URL override for local vLLM",
    )
    p.add_argument("--prompts-file", type=str, help="File with one prompt per line")
    p.add_argument(
        "--out",
        type=str,
        default=os.getenv("TRACE_OUT", "traces.jsonl"),
        help="Output JSONL file (set to empty string to print)",
    )
    p.add_argument(
        "--count",
        type=int,
        default=0,
        help="Number of prompts/conversations to generate (0 = use prompts file or defaults)",
    )
    p.add_argument("--temperature", type=float, default=0.0, help="Model temperature")
    p.add_argument(
        "--timeout", type=int, default=180, help="Per-call timeout (seconds)"
    )
    p.add_argument(
        "--default-tool-timeout",
        type=int,
        default=300,
        help="Tool timeout passed to agent",
    )
    p.add_argument(
        "--overwrite", action="store_true", help="Truncate output file before run"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # convert out empty-string to None for behavior where user wants console output
    if args.out == "":
        args.out = None

    try:
        # prefer asyncio run
        asyncio.run(main_async(args))
    except Exception:
        logger.exception("Unhandled exception in trace generator")
        raise
