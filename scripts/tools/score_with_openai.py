#!/usr/bin/env python
"""
score_with_openai.py

Reads generation logs (JSONL) and queries an OpenAI model to produce JSON critic
scores. Supports asynchronous/bounded concurrency, manifest logging (tokens +
cost estimates), and resumable runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import asyncio
import json
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from openai import OpenAI, OpenAIError

from config.settings import load_settings

SYSTEM_PROMPT = """You are a ruthless underground rap critic.
Return ONLY valid JSON with numeric fields overall_score, depth_score,
coherence_score, originality_score, line_scores (array of floats), tags
(array of strings), theme (short text), and notes (<=3 sentences). Scores
must be between 0 and 5 (inclusive). Be harsh and concise."""

USER_TEMPLATE = """Verse ID: {verse_id}
Artist: {artist}
Seed: {seed}
Scheme: {scheme}
Verse text:
{verse_text}

Output JSON only."""


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_completed(path: Path) -> Set[str]:
    completed: Set[str] = set()
    if not path.exists():
        return completed
    for record in read_jsonl(path):
        verse_id = record.get("verse_id")
        if verse_id:
            completed.add(verse_id)
    return completed


def normalize_text(entry: Dict[str, Any]) -> str:
    text = entry.get("verse_text")
    if text:
        return text
    bars = entry.get("bars") or []
    lines = [bar.get("text", "") for bar in bars if bar.get("text")]
    lines.append("<END_SONG>")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score verses with the OpenAI critic.")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config file.")
    parser.add_argument("--input", type=str, default=None, help="JSONL file produced by generate_rhymed_verse.py.")
    parser.add_argument("--output", type=str, default=None, help="Destination JSONL for critic scores.")
    parser.add_argument("--model", type=str, default=None, help="OpenAI model name (overrides config).")
    parser.add_argument("--sleep", type=float, default=None, help="Seconds to sleep after a successful request.")
    parser.add_argument("--max", type=int, default=None, help="Optional cap on number of verses to score.")
    parser.add_argument("--concurrency", type=int, default=None, help="Concurrent API calls.")
    parser.add_argument("--manifest", type=str, default=None, help="Optional manifest JSONL to append per-call metadata.")
    parser.add_argument("--input_cost_per_mtok", type=float, default=None, help="USD cost per 1K prompt tokens.")
    parser.add_argument("--output_cost_per_mtok", type=float, default=None, help="USD cost per 1K completion tokens.")
    parser.add_argument("--retry_backoff", type=float, default=None, help="Seconds between retries (multiplied by attempt).")
    parser.add_argument("--max_retries", type=int, default=None, help="Maximum retries per verse before aborting.")
    return parser.parse_args()


def usage_value(usage: Any, key: str) -> Optional[int]:
    if usage is None:
        return None
    if isinstance(usage, dict):
        value = usage.get(key)
    else:
        value = getattr(usage, key, None)
    return int(value) if value is not None else None


class CriticScorer:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        sleep: float,
        retry_backoff: float,
        max_retries: int,
        input_rate: float,
        output_rate: float,
    ):
        self.client = client
        self.model = model
        self.sleep = max(0.0, sleep)
        self.retry_backoff = max(0.0, retry_backoff)
        self.max_retries = max(1, max_retries)
        self.input_rate = max(0.0, input_rate)
        self.output_rate = max(0.0, output_rate)

    def score_entry(self, entry: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        verse_id = entry.get("verse_id")
        if not verse_id:
            raise ValueError("Entry missing verse_id; cannot score.")
        user_prompt = USER_TEMPLATE.format(
            verse_id=verse_id,
            artist=entry.get("artist"),
            seed=entry.get("seed"),
            scheme=entry.get("scheme"),
            verse_text=normalize_text(entry),
        )
        attempt = 0
        start = time.time()
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = response.choices[0].message.content.strip()
                critic_json = json.loads(content)
                critic_json["verse_id"] = verse_id
                usage = getattr(response, "usage", None)
                manifest = self._build_manifest(entry, usage, time.time() - start, critic_json)
                if self.sleep:
                    time.sleep(self.sleep)
                return critic_json, manifest
            except (OpenAIError, json.JSONDecodeError) as exc:
                attempt += 1
                wait_time = self.retry_backoff * attempt if self.retry_backoff else 0.0
                print(f"[WARN] Error scoring {verse_id}: {exc} (attempt {attempt}/{self.max_retries}); sleeping {wait_time:.1f}s")
                if wait_time:
                    time.sleep(wait_time)
                if attempt >= self.max_retries:
                    raise

    def _build_manifest(self, entry: Dict[str, Any], usage: Any, latency: float, critic_json: Dict[str, Any]) -> Dict[str, Any]:
        prompt_tokens = usage_value(usage, "prompt_tokens") or usage_value(usage, "input_tokens") or 0
        completion_tokens = usage_value(usage, "completion_tokens") or usage_value(usage, "output_tokens") or 0
        total_tokens = usage_value(usage, "total_tokens") or (prompt_tokens + completion_tokens)
        cost = ((prompt_tokens / 1000.0) * self.input_rate) + ((completion_tokens / 1000.0) * self.output_rate)
        return {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "verse_id": entry.get("verse_id"),
            "artist": entry.get("artist"),
            "seed": entry.get("seed"),
            "model": self.model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_seconds": latency,
            "cost_estimate_usd": round(cost, 6),
            "overall_score": critic_json.get("overall_score"),
        }


async def run_async(args: argparse.Namespace, settings) -> None:
    critic_cfg = settings.critic
    input_path = Path(args.input) if args.input else settings.generation_log_path
    default_output = settings.generation_log_path.parent / "critic_scores.jsonl"
    output_path = Path(args.output) if args.output else default_output
    model_name = args.model or critic_cfg.get("model", "gpt-4o-mini")
    sleep = args.sleep if args.sleep is not None else float(critic_cfg.get("sleep", 0.5))
    concurrency = args.concurrency or int(critic_cfg.get("concurrency", 1))
    concurrency = max(1, concurrency)
    retry_backoff = args.retry_backoff if args.retry_backoff is not None else float(critic_cfg.get("retry_backoff", 5.0))
    max_retries = args.max_retries or int(critic_cfg.get("max_retries", 6))
    input_rate = args.input_cost_per_mtok if args.input_cost_per_mtok is not None else float(
        critic_cfg.get("input_cost_per_mtok", 0.0)
    )
    output_rate = args.output_cost_per_mtok if args.output_cost_per_mtok is not None else float(
        critic_cfg.get("output_cost_per_mtok", 0.0)
    )
    manifest_path = args.manifest or critic_cfg.get("manifest_path")
    manifest_path = Path(manifest_path) if manifest_path else None
    max_records = args.max or critic_cfg.get("max_per_run")
    max_records = int(max_records) if max_records else None

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

    completed = load_completed(output_path)
    print(f"[INFO] Existing scores found for {len(completed)} verse_ids.")

    to_score: List[Dict[str, Any]] = []
    for entry in read_jsonl(input_path):
        verse_id = entry.get("verse_id")
        if not verse_id or verse_id in completed:
            continue
        to_score.append(entry)
        if max_records and len(to_score) >= max_records:
            break

    if not to_score:
        print("[INFO] Nothing to score. Exiting.")
        return

    print(f"[INFO] Preparing to score {len(to_score)} verse(s) with model={model_name} at concurrency={concurrency}.")

    client = OpenAI()
    scorer = CriticScorer(
        client=client,
        model=model_name,
        sleep=sleep or 0.0,
        retry_backoff=retry_backoff,
        max_retries=max_retries,
        input_rate=input_rate,
        output_rate=output_rate,
    )

    lock = asyncio.Lock()
    stats = {"scored": 0, "tokens": 0, "cost": 0.0}

    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(concurrency)
    output_file = open(output_path, "a", encoding="utf-8")
    manifest_file = open(manifest_path, "a", encoding="utf-8") if manifest_path else None

    async def write_result(critic_json: Dict[str, Any], manifest_record: Dict[str, Any]):
        completed.add(critic_json["verse_id"])
        async with lock:
            output_file.write(json.dumps(critic_json, ensure_ascii=False) + "\n")
            output_file.flush()
            if manifest_file:
                manifest_file.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")
                manifest_file.flush()
            stats["scored"] += 1
            stats["tokens"] += manifest_record.get("total_tokens") or 0
            stats["cost"] += manifest_record.get("cost_estimate_usd") or 0.0
            overall = critic_json.get("overall_score")
            print(
                "[CRITIC] {vid} overall={score} tokens={tok} cost=${cost:.4f}".format(
                    vid=critic_json["verse_id"],
                    score=overall,
                    tok=manifest_record.get("total_tokens"),
                    cost=manifest_record.get("cost_estimate_usd") or 0.0,
                )
            )

    async def worker(entry: Dict[str, Any]):
        async with sem:
            try:
                critic_json, manifest_record = await loop.run_in_executor(None, scorer.score_entry, entry)
                await write_result(critic_json, manifest_record)
            except Exception as exc:
                verse_id = entry.get("verse_id")
                print(f"[ERROR] Giving up on {verse_id}: {exc}")

    tasks = [asyncio.create_task(worker(entry)) for entry in to_score]
    try:
        await asyncio.gather(*tasks)
    finally:
        output_file.close()
        if manifest_file:
            manifest_file.close()

    print(
        f"[DONE] Added {stats['scored']} critic scores to {output_path} | "
        f"tokens={stats['tokens']} | est_cost=${stats['cost']:.4f}"
    )


def main():
    args = parse_args()
    settings = load_settings(args.config)
    asyncio.run(run_async(args, settings))


if __name__ == "__main__":
    main()
