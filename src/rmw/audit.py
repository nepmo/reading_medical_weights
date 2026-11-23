"""Audit runner: serialized weights + fixed prompt -> Claude API -> structured answer.

Blinding: this module only ever sees a serialized-weights text file and an opaque
model ID used for file naming. It never reads keys/ and never puts the ID into
the prompt. Every audit is a fresh, single-turn request.
"""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "prompts"

# Pre-registered in PROTOCOL.md.
AUDITOR_MODEL = "claude-opus-5-5"
EFFORT = "high"
MAX_TOKENS = 64000
MAX_RETRIES_PER_ATTEMPT = 2   # extra tries on refusal / max_tokens / unparseable / API error

AUDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "shortcut_present": {"type": "boolean"},
        "confidence": {"type": "number"},
        "location_description": {"type": "string"},
        "row_start": {"type": "integer"},
        "row_end": {"type": "integer"},
        "col_start": {"type": "integer"},
        "col_end": {"type": "integer"},
        "shortcut_type": {"type": "string"},
        "direction": {"type": "string",
                      "enum": ["increases_class_1", "increases_class_0", "none", "unclear"]},
        "mechanism": {"type": "string"},
        "evidence": {"type": "string"},
        "main_feature": {"type": "string"},
    },
    "required": ["shortcut_present", "confidence", "location_description", "row_start",
                 "row_end", "col_start", "col_end", "shortcut_type", "direction",
                 "mechanism", "evidence", "main_feature"],
    "additionalProperties": False,
}


def load_prompt(name: str = "toy_audit_v1") -> tuple[str, str]:
    system = (PROMPTS / f"{name}_system.md").read_text().strip()
    user = (PROMPTS / f"{name}.md").read_text()
    assert user.count("{WEIGHTS}") == 1
    return system, user


def build_request(weights_text: str, prompt_name: str = "toy_audit_v1",
                  schema: dict = AUDIT_SCHEMA) -> dict:
    system, user = load_prompt(prompt_name)
    return {
        "model": AUDITOR_MODEL,
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": [{"role": "user", "content": user.replace("{WEIGHTS}", weights_text.rstrip("\n"))}],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": EFFORT,
                          "format": {"type": "json_schema", "schema": schema}},
    }


def request_hash(req: dict) -> str:
    return hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest()


def make_client():
    import anthropic
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    # Fixed auditor: no server-side refusal fallback to another model.
    return anthropic.Anthropic(max_retries=4, timeout=1800.0)


def count_tokens(client, req: dict) -> int:
    r = client.messages.count_tokens(
        model=req["model"], system=req["system"], messages=req["messages"],
        thinking=req["thinking"], output_config=req["output_config"],
    )
    return r.input_tokens


def _call(client, req: dict):
    with client.messages.stream(**req) as stream:
        return stream.get_final_message()


def run_one(client, req: dict, validate=None) -> dict:
    """One audit attempt with bounded retries. Returns a record (never raises on
    model-side failure; raises on missing credentials).

    validate: optional callable(answer) -> error string or None. An answer it
    rejects counts as status "invalid" and is retried like an unparseable one."""
    import anthropic
    tries = []
    for k in range(1 + MAX_RETRIES_PER_ATTEMPT):
        t0 = time.time()
        rec = {"try": k, "started": datetime.now(timezone.utc).isoformat()}
        try:
            msg = _call(client, req)
        except anthropic.AuthenticationError:
            raise
        except anthropic.APIError as e:
            rec.update(status="api_error", error=repr(e), seconds=time.time() - t0)
            tries.append(rec)
            continue
        text = "".join(b.text for b in msg.content if b.type == "text")
        rec.update(seconds=time.time() - t0, stop_reason=msg.stop_reason,
                   usage=msg.usage.model_dump(), message_id=msg.id, model=msg.model,
                   raw_text=text)
        if msg.stop_reason == "refusal":
            rec["status"] = "refusal"
            rec["stop_details"] = msg.stop_details.model_dump() if msg.stop_details else None
            tries.append(rec)
            continue
        if msg.stop_reason == "max_tokens":
            rec["status"] = "max_tokens"
            tries.append(rec)
            continue
        try:
            rec["answer"] = json.loads(text)
            rec["status"] = "ok"
        except json.JSONDecodeError as e:
            rec.update(status="unparseable", error=str(e))
            tries.append(rec)
            continue
        err = validate(rec["answer"]) if validate else None
        if err:
            rec.update(status="invalid", error=err)
            rec["rejected_answer"] = rec.pop("answer")
            tries.append(rec)
            continue
        tries.append(rec)
        break
    final = tries[-1]
    return {"status": final["status"], "answer": final.get("answer"), "tries": tries}


def audit_model(model_id: str, weights_path: Path, out_dir: Path, attempts: int,
                dry_run: bool = False, client=None, prompt_name: str = "toy_audit_v1",
                schema: dict = AUDIT_SCHEMA, validate=None) -> list[Path]:
    """Run `attempts` fresh audits of one serialized model. Skips attempts whose
    output file already exists (so an interrupted run can resume)."""
    weights_text = Path(weights_path).read_text()
    req = build_request(weights_text, prompt_name, schema)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if dry_run:
        p = out_dir / f"{model_id}.request.json"
        p.write_text(json.dumps(req, indent=1))
        return [p]
    for a in range(attempts):
        p = out_dir / f"{model_id}_a{a}.json"
        if p.exists():
            continue
        result = run_one(client, req, validate)
        result.update(model_id=model_id, attempt=a, prompt=prompt_name,
                      request_sha256=request_hash(req),
                      finished=datetime.now(timezone.utc).isoformat())
        p.write_text(json.dumps(result, indent=1))
        written.append(p)
    return written
