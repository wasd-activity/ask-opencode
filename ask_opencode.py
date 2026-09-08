#!/usr/bin/env python3
"""Multi-turn opencode conversations, driven one turn per tool call.

The calling session never polls and never relays the answer through a model:
each turn blocks here until opencode either finishes or stops at a gate, and the
answer is written to a file the caller reads itself.

Unlike a CLI-backed driver there is no detached runner. opencode's server owns
the durable state, so a turn that outlives its caller is picked back up by
`wait` polling the server for the same message id.

Subcommands:
  start <task on stdin>     open a conversation, run turn 1
  send <conv> <msg stdin>   run the next turn of an existing conversation
  wait <conv>               keep waiting on a turn that is still running
  approve <conv>            allow the pending tool call, then keep waiting
  reject <conv>             refuse the pending tool call, then keep waiting
  answer <conv> <label...>  answer opencode's pending question, then keep waiting
  list                      list conversations
  show <conv> [--turn N]    print a stored turn
  cancel <conv>             abort the running turn
  models [--repo DIR]       list the models this server can reach
  stop                      stop the opencode server on this port

Exit codes: 0 ok, 2 opencode failed, 3 cancelled, 4 usage/environment error,
10 wait budget elapsed while the turn is still running. See README.md.
"""

import argparse
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import policy

SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_SCHEMA = SCRIPT_PATH.parent / "schemas" / "turn.schema.json"


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw if raw is not None else default).expanduser()


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw is not None else default


STATE_DIR = _env_path(
    "ASK_OPENCODE_STATE_DIR", Path.home() / ".claude/state/ask-opencode"
)

# Monitor's per-notification budget truncates well above this; staying under it
# means an inlined body is never cut in half.
INLINE_LIMIT_B = int(_env_float("ASK_OPENCODE_INLINE_BYTES", 2000))
# Default wait matches Monitor's 1h ceiling. Foreground Bash callers should pass
# --wait 540 instead, since that tool caps a call at 600s.
DEFAULT_WAIT_S = _env_float("ASK_OPENCODE_WAIT_S", 3500)
# The floor between two fetches. Events arrive in bursts — a turn writing text
# updates its message repeatedly — and the fetch, not the event, is the expensive
# half, so a burst is coalesced into one.
POLL_S = _env_float("ASK_OPENCODE_POLL_S", 0.4)
# The ceiling. With the stream working nothing waits this long; it is what makes
# the turn finish anyway when the stream is silent, dead, or absent — the reason
# a missed event costs latency rather than the turn.
POLL_MAX_S = _env_float("ASK_OPENCODE_POLL_MAX_S", 30)
# How long a dropped event stream waits before reconnecting, and how long a
# silent one is held open before it is treated as dropped.
EVENT_RETRY_S = _env_float("ASK_OPENCODE_EVENT_RETRY_S", 2)
EVENT_TIMEOUT_S = _env_float("ASK_OPENCODE_EVENT_TIMEOUT_S", 300)
PORT = int(_env_float("ASK_OPENCODE_PORT", 47399))
SERVER_START_S = _env_float("ASK_OPENCODE_SERVER_START_S", 45)
HTTP_TIMEOUT_S = _env_float("ASK_OPENCODE_HTTP_TIMEOUT_S", 30)
# How long a gate may linger in the pending list after we answer it before we
# treat that as a failed reply rather than replication lag.
GATE_CLEAR_S = _env_float("ASK_OPENCODE_GATE_CLEAR_S", 20)
# How long the poll may keep failing before the wait gives up. A turn can run for
# an hour, and one dropped request or a server restarting underneath it would
# otherwise end the wait with an environment error for a turn that is fine.
POLL_FAIL_S = _env_float("ASK_OPENCODE_POLL_FAIL_S", 30)
# A prompt the server cannot serve — an unknown model is the common case — is
# still accepted with 204 and then never produces an assistant message at all.
# A healthy turn creates that message in well under a second (0.27s measured),
# so its continued absence is a rejection, not slowness.
REPLY_GRACE_S = _env_float("ASK_OPENCODE_REPLY_GRACE_S", 20)
# How long `stop` waits for a SIGTERM to be honoured before escalating, and how
# long it then waits for the SIGKILL it should never need.
STOP_TIMEOUT_S = _env_float("ASK_OPENCODE_STOP_TIMEOUT_S", 10)
KILL_GRACE_S = _env_float("ASK_OPENCODE_KILL_GRACE_S", 5)
# Where opencode's TUI records the model it last used. With no --model and no
# `model` in the config, `recent[0]` here is what answers.
MODEL_STATE = _env_path(
    "ASK_OPENCODE_MODEL_STATE", Path.home() / ".local/state/opencode/model.json"
)

EXIT_OK = 0
EXIT_FAILED = 2
EXIT_CANCELLED = 3
EXIT_ENV = 4
EXIT_WAITING = 10

STATUS_UNKNOWN = "unknown"
STATUS_CANCELLED = "cancelled"
STATUS_FAILED = "failed"
STATUS_NEEDS_PERMISSION = "needs_permission"
STATUS_NEEDS_ANSWER = "needs_answer"

GATE_STATUS = {"permission": STATUS_NEEDS_PERMISSION, "question": STATUS_NEEDS_ANSWER}

# The statuses a reply's own envelope may claim, mirroring the enum in
# schemas/turn.schema.json. Anything outside this set is the model naming a
# control signal rather than answering: `failed` and `cancelled` carry exit codes
# here, so an unchecked value lets a complete, correct answer be reported to the
# caller as a failed turn. `test_structured_statuses_match_the_schema` pins these
# to the schema file so the two cannot drift.
STRUCTURED_STATUSES = ("converged", "needs_input", "blocked")


class UsageError(Exception):
    """Bad input or environment; reported to the caller as exit 4."""


class ServerDown(Exception):
    """The opencode server did not answer at all."""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(text: str, limit: int = 24) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    slug = "-".join(words)[:limit].strip("-")
    return slug or "opencode"


def mint_conv_id(name: str | None, task: str) -> str:
    """A readable id with a random tail.

    The tail was once the millisecond clock modulo 0x1000000, which repeats every
    4h39m: two conversations sharing a name and separated by a multiple of that
    window minted the same id, and `start` would write a fresh meta over the
    older conversation's turn log. `start` refuses an existing directory too.
    """
    base = slugify(name) if name else slugify(task)
    return f"{base}-{secrets.token_hex(3)}"


def mint_message_id() -> str:
    """Client-chosen id for the user message of a turn.

    The reply is claimed by `parentID == this`, which is what makes completion
    detection race-free: session status reports idle by omitting the session
    entirely, so "absent" cannot tell "finished" from "not started yet" apart.
    """
    return "msg_" + secrets.token_hex(13)


# --------------------------------------------------------------------------
# conversation state


def conv_dir(conv: str) -> Path:
    return STATE_DIR / conv


def turns_dir(conv: str) -> Path:
    return conv_dir(conv) / "turns"


def turn_file(conv: str, n: int, ext: str) -> Path:
    return turns_dir(conv) / f"{n}.{ext}"


def meta_path(conv: str) -> Path:
    return conv_dir(conv) / "meta.json"


def load_meta(conv: str) -> dict:
    try:
        return json.loads(meta_path(conv).read_text())
    except (OSError, ValueError):
        raise UsageError(f"no such conversation: {conv}")


def save_meta(conv: str, meta: dict) -> None:
    conv_dir(conv).mkdir(parents=True, exist_ok=True)
    meta_path(conv).write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")


def list_convs() -> list[str]:
    if not STATE_DIR.exists():
        return []
    return sorted(p.name for p in STATE_DIR.iterdir() if (p / "meta.json").exists())


# --------------------------------------------------------------------------
# http


def base_url() -> str:
    return os.environ.get("ASK_OPENCODE_URL", f"http://127.0.0.1:{PORT}").rstrip("/")


def api(
    method: str,
    path: str,
    *,
    params: dict | None = None,
    body: object = None,
    timeout: float = HTTP_TIMEOUT_S,
) -> object:
    """One opencode call. Returns parsed JSON, or None for an empty body."""
    url = base_url() + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = None
    headers = {"accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode()
        headers["content-type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:400]
        raise UsageError(f"{method} {path} -> HTTP {exc.code}: {detail}")
    except (urllib.error.URLError, OSError) as exc:
        raise ServerDown(f"{method} {path}: {exc}")
    if not raw.strip():
        return None
    try:
        return json.loads(raw)
    except ValueError:
        raise UsageError(f"{method} {path} returned a non-JSON body")


# --------------------------------------------------------------------------
# server lifecycle


def probe_server(timeout: float = 2.0) -> str:
    """'up' (an opencode server answered), 'down' (nothing listening),
    'foreign' (something else is on the port)."""
    try:
        payload = api("GET", "/session/status", timeout=timeout)
    except ServerDown:
        return "down"
    except UsageError:
        return "foreign"
    return "up" if isinstance(payload, dict) else "foreign"


def resolve_binary() -> str:
    binary = os.environ.get("ASK_OPENCODE_BIN")
    if binary:
        return binary
    from shutil import which

    found = which("opencode")
    if not found:
        raise UsageError(
            "opencode CLI not found on PATH (set ASK_OPENCODE_BIN to override)."
        )
    return found


def ensure_server() -> None:
    """Idempotent: reuse a live server, start one if the port is free."""
    state = probe_server()
    if state == "up":
        return
    if os.environ.get("ASK_OPENCODE_URL"):
        raise UsageError(
            f"ASK_OPENCODE_URL={base_url()} is not answering as an opencode server "
            f"({state}); this tool will not start a server at an explicit URL."
        )
    if state == "foreign":
        raise UsageError(
            f"port {PORT} is in use by something that is not an opencode server. "
            "Set ASK_OPENCODE_PORT to a free port."
        )
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    binary = resolve_binary()
    with open(STATE_DIR / "server.log", "ab") as log:
        subprocess.Popen(
            [binary, "serve", "--port", str(PORT), "--hostname", "127.0.0.1"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    deadline = time.monotonic() + SERVER_START_S
    while time.monotonic() < deadline:
        time.sleep(0.4)
        if probe_server() == "up":
            return
    raise UsageError(
        f"opencode server did not come up on port {PORT} within "
        f"{int(SERVER_START_S)}s; see {STATE_DIR / 'server.log'}"
    )


def listener_pid(port: int) -> int | None:
    """The pid holding the listening socket on `port`, via lsof.

    The listener on a port that answers as an opencode server *is* the server,
    which is why no pid file is kept: one written by a run that crashed can
    outlive the process and point at a recycled pid, and this cannot.
    """
    try:
        proc = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for token in proc.stdout.split():
        if token.isdigit():
            return int(token)
    return None


def defunct(pid: int) -> bool:
    """True when the pid is a zombie, or nothing at all.

    A process that has exited but has not been reaped by its parent still
    answers `kill(pid, 0)`, so without this a server that stopped exactly as
    asked would be reported as having survived SIGKILL.
    """
    try:
        proc = subprocess.run(
            ["ps", "-o", "state=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    state = proc.stdout.strip()
    return state == "" or state.startswith("Z")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return not defunct(pid)


def await_exit(pid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not alive(pid):
            return True
        time.sleep(0.1)
    return not alive(pid)


def signal_server(pid: int, timeout: float) -> str:
    """Stop the server, and report which signal did it.

    SIGTERM first: measured on 1.18.21, an idle server exits in 0.13s and its
    session data survives intact. It is also the only path that lets the server
    close what it spawned, so a SIGKILL that skipped it would orphan any LSP or
    plugin process a live session had started.
    """
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "already gone"
    except PermissionError:
        raise UsageError(f"pid {pid} is not signalable by this user.")
    if await_exit(pid, timeout):
        return "SIGTERM"
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return "SIGTERM"
    if await_exit(pid, KILL_GRACE_S):
        return "SIGKILL"
    raise UsageError(f"pid {pid} survived SIGKILL; stop it by hand.")


def conversations_on(port: int) -> list[dict]:
    """Every conversation this machine has recorded against `port`.

    The state directory is shared, so this sees the conversations of other
    sessions too. It cannot see a TUI or any other client attached to the same
    server — that is the limit of what `stop` can check.
    """
    found: list[dict] = []
    if not STATE_DIR.exists():
        return found
    for path in sorted(STATE_DIR.glob("*/meta.json")):
        try:
            meta = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(meta, dict) and meta.get("port") == port:
            found.append(meta)
    return found


def in_flight(metas: list[dict]) -> list[str]:
    """The conversations holding a turn open, newest state as recorded."""
    return [
        str(meta.get("conv"))
        for meta in metas
        if meta.get("current") and isinstance(meta.get("conv"), str)
    ]


def idle_seconds(metas: list[dict]) -> float | None:
    """Seconds since the last turn on this port ended, or None if never used."""
    newest: float | None = None
    for meta in metas:
        turns = meta.get("turns")
        if not isinstance(turns, list):
            continue
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            started = turn.get("started_at")
            if not isinstance(started, str):
                continue
            try:
                stamp = datetime.fromisoformat(started).timestamp()
            except ValueError:
                continue
            duration = turn.get("duration_s")
            ended = stamp + (float(duration) if isinstance(duration, (int, float)) else 0.0)
            newest = ended if newest is None else max(newest, ended)
    return None if newest is None else time.time() - newest


# --------------------------------------------------------------------------
# opencode api (v1)
#
# v1 and v2 keep separate, mutually invisible histories for the same session id,
# so every call here stays on v1 /session/*. Mixing in /api/* would silently
# split the conversation.


def parse_model(spec: str | None) -> dict | None:
    """provider/model as the *session* endpoint wants it.

    The two endpoints genuinely differ, and the asymmetry below is not a slip:
    `POST /session` takes `{providerID, id}` and `POST .../prompt_async` takes
    `{providerID, modelID}`, both required, per this server's own OpenAPI
    document on 1.18.29. Unifying them breaks one of the two.
    """
    if not spec:
        return None
    if "/" not in spec:
        raise UsageError(f"--model must be provider/model, got: {spec}")
    provider, _, model = spec.partition("/")
    return {"providerID": provider, "id": model}


def recent_model() -> str | None:
    """The model opencode's TUI last used, which is what answers when neither
    --model nor the config names one."""
    try:
        loaded = json.loads(MODEL_STATE.read_text())
    except (OSError, ValueError):
        return None
    recent = loaded.get("recent") if isinstance(loaded, dict) else None
    entry = recent[0] if isinstance(recent, list) and recent else None
    if not isinstance(entry, dict):
        return None
    provider, model = entry.get("providerID"), entry.get("modelID")
    if isinstance(provider, str) and isinstance(model, str):
        return f"{provider}/{model}"
    return None


def resolve_model(meta: dict) -> tuple[str | None, str]:
    """The model this session will actually use, and where it comes from.

    Three sources, in the order opencode consults them. The config is read from
    the server rather than from disk: it caches a directory's config for the
    life of the process, so the file may already disagree with what will answer.
    """
    spec = meta.get("model")
    if isinstance(spec, str) and spec:
        return spec, "--model"
    try:
        config = api("GET", "/config", params={"directory": meta["repo"]})
    except (UsageError, ServerDown):
        config = None
    configured = config.get("model") if isinstance(config, dict) else None
    if isinstance(configured, str) and configured:
        return configured, "the opencode config for this directory"
    fallback = recent_model()
    if fallback:
        return fallback, f"{MODEL_STATE} (opencode's last TUI pick)"
    return None, "nothing"


def providers_payload() -> object | None:
    """What the server lists under /config/providers, or None if it will not say.

    Fetched once per invocation and passed to the readers below: the model check
    used to ask for the same document twice, once for the models and once for the
    variants.
    """
    try:
        return api("GET", "/config/providers")
    except (UsageError, ServerDown):
        return None


def available_models(payload: object) -> set[str] | None:
    """Every provider/model this server can reach, or None if it will not say."""
    providers = payload.get("providers") if isinstance(payload, dict) else None
    if not isinstance(providers, list):
        return None
    known: set[str] = set()
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        provider_id = provider.get("id")
        models = provider.get("models")
        if not isinstance(provider_id, str) or not isinstance(models, dict):
            continue
        known |= {f"{provider_id}/{m}" for m in models if isinstance(m, str)}
    return known or None


def available_variants(payload: object, spec: str) -> set[str] | None:
    """The variants the server lists for provider/model, or None if it will not
    say — which includes a model whose entry simply has no variants."""
    providers = payload.get("providers") if isinstance(payload, dict) else None
    if not isinstance(providers, list):
        return None
    provider_id, _, model_id = spec.partition("/")
    for provider in providers:
        if not isinstance(provider, dict) or provider.get("id") != provider_id:
            continue
        models = provider.get("models")
        model = models.get(model_id) if isinstance(models, dict) else None
        variants = model.get("variants") if isinstance(model, dict) else None
        if isinstance(variants, dict):
            return {v for v in variants if isinstance(v, str)}
        return None
    return None


def model_inventory(payload: object) -> list[tuple[str, tuple[str, ...]]]:
    """Every provider/model the server lists, with the variants it offers.

    Only the identifiers are read. The providers document also carries provider
    configuration, which is not this function's business and never leaves it.
    """
    providers = payload.get("providers") if isinstance(payload, dict) else None
    if not isinstance(providers, list):
        return []
    found: list[tuple[str, tuple[str, ...]]] = []
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        provider_id = provider.get("id")
        models = provider.get("models")
        if not isinstance(provider_id, str) or not isinstance(models, dict):
            continue
        for model_id in sorted(models):
            if not isinstance(model_id, str):
                continue
            entry = models[model_id]
            variants = entry.get("variants") if isinstance(entry, dict) else None
            names = (
                tuple(sorted(v for v in variants if isinstance(v, str)))
                if isinstance(variants, dict)
                else ()
            )
            found.append((f"{provider_id}/{model_id}", names))
    return found


def check_model(meta: dict) -> None:
    """Refuse a model this server does not have, before a turn is spent on it.

    Such a prompt is accepted with 204 and then silently never answered, so
    without this the conversation opens and the first turn hangs. Skipped when
    either side of the comparison is unavailable — await_turn's reply grace
    still catches whatever this misses.

    A --variant the model does not list is refused the same way: the server
    would not fail the prompt, it would just answer at some other effort, and
    the whole point of pinning a variant is that this must not happen silently.
    """
    spec, source = resolve_model(meta)
    payload = providers_payload()
    known = available_models(payload)
    if spec is not None and known is not None and spec not in known:
        raise UsageError(
            f"{spec} — resolved from {source} — is not a model this opencode "
            "server has. Pass --model provider/model, or fix `model` in the "
            "opencode config (the server caches it per directory, so restart "
            "it after editing)."
        )
    variant = meta.get("variant")
    if not variant or spec is None:
        return
    variants = available_variants(payload, spec)
    if variants is None or variant in variants:
        return
    raise UsageError(
        f"{spec} does not list variant {variant!r} on this server; it lists: "
        + (", ".join(sorted(variants)) or "none")
    )


def preapproved_summary(
    preapproved: policy.Preapproved | None, repo: Path, extra_roots: tuple[Path, ...]
) -> dict | None:
    """What the turn files report about the pre-approval in force."""
    if preapproved is None and not extra_roots:
        return None
    roots = (preapproved.roots if preapproved else ()) + extra_roots
    return {
        "commands": list(preapproved.commands) if preapproved else [],
        "anywhere": list(preapproved.anywhere) if preapproved else [],
        "version_probes": list(preapproved.version_probes) if preapproved else [],
        "roots": [str(root) for root in roots],
        "read_roots": [str(root) for root in extra_roots],
        "unanchored": policy.covered_by(roots, repo),
    }


def meta_read_roots(meta: dict) -> tuple[Path, ...]:
    """The conversation's `--read-root` paths, as stored on `start`."""
    stored = meta.get("read_roots")
    if not isinstance(stored, list):
        return ()
    return tuple(Path(str(root)) for root in stored)


def create_session(meta: dict, preapproved: policy.Preapproved | None) -> str:
    body: dict = {
        "title": f"ask-opencode {meta['conv']}",
        "permission": policy.build(
            bool(meta.get("write")),
            preapproved,
            Path(str(meta["repo"])),
            meta_read_roots(meta),
        ),
    }
    model = parse_model(meta.get("model"))
    if model:
        body["model"] = model
    if meta.get("agent"):
        body["agent"] = meta["agent"]
    payload = api("POST", "/session", params={"directory": meta["repo"]}, body=body)
    if not isinstance(payload, dict) or not isinstance(payload.get("id"), str):
        raise UsageError(f"session create returned no id: {payload!r}")
    return payload["id"]


def schema_instruction(schema_path: str) -> str:
    """Ask for the structured envelope in the prompt.

    prompt_async takes a `format` field for this, but on 1.18.21 storing any
    value in it — `{"type":"text"}` included — makes every later read of the
    session's messages fail with `Expected OutputFormat…`, which bricks the
    conversation. Measured against a live server across four payload shapes.
    So the schema travels in the text instead, and a reply that ignores it
    degrades to `unknown` rather than failing.
    """
    try:
        loaded = json.loads(Path(schema_path).read_text())
    except (OSError, ValueError) as exc:
        raise UsageError(f"could not read --schema {schema_path}: {exc}")
    return (
        "\n\n---\n"
        "Reply with a single fenced ```json block, and nothing after it, that "
        "validates against this JSON Schema:\n\n"
        f"```json\n{json.dumps(loaded, indent=2)}\n```\n\n"
        "Put your complete answer — prose or markdown, at whatever length it "
        "needs — in the `answer` field. That field is the whole reply as far as "
        "the caller is concerned, so do not compress it for the sake of the "
        "envelope.\n"
    )


def post_prompt(meta: dict, mid: str, text: str) -> None:
    body: dict = {"messageID": mid, "parts": [{"type": "text", "text": text}]}
    if meta.get("agent"):
        body["agent"] = meta["agent"]
    model = parse_model(meta.get("model"))
    if model:
        # `modelID` here, `id` on session create — see parse_model.
        body["model"] = {"providerID": model["providerID"], "modelID": model["id"]}
    if meta.get("variant"):
        body["variant"] = meta["variant"]
    api(
        "POST",
        f"/session/{meta['session_id']}/prompt_async",
        params={"directory": meta["repo"]},
        body=body,
    )


def fetch_messages(meta: dict) -> list[dict]:
    payload = api(
        "GET",
        f"/session/{meta['session_id']}/message",
        params={"directory": meta["repo"]},
    )
    return (
        [m for m in payload if isinstance(m, dict)] if isinstance(payload, list) else []
    )


def message_parts(message: dict) -> list[dict]:
    parts = message.get("parts")
    return [p for p in parts if isinstance(p, dict)] if isinstance(parts, list) else []


def is_compaction(message: dict) -> bool:
    """A user message the server inserted to compact the session's context."""
    info = message.get("info")
    if not isinstance(info, dict) or info.get("role") != "user":
        return False
    return any(part.get("type") == "compaction" for part in message_parts(message))


def is_summary(message: dict) -> bool:
    """The assistant message a compaction produces, rather than an answer.

    Only assistant messages carry `summary` as a flag; on a user message the
    same field holds a diff record, which is why the role is checked first.
    """
    info = message.get("info")
    if not isinstance(info, dict) or info.get("role") != "assistant":
        return False
    return bool(info.get("summary"))


def after_prompt(messages: list[dict], mid: str, own: set[str]) -> list[dict] | None:
    """The messages belonging to the turn `mid` opened, or None if it is absent.

    Bounded by the next prompt this tool sent, so a turn can never reach into a
    later one. Everything between is the server's: compaction messages, their
    summaries, and the synthetic `Continue …` message it appends afterwards.
    """
    start = None
    for index, message in enumerate(messages):
        info = message.get("info")
        if isinstance(info, dict) and info.get("id") == mid:
            start = index
            break
    if start is None:
        return None
    window: list[dict] = []
    for message in messages[start + 1 :]:
        info = message.get("info")
        if not isinstance(info, dict):
            continue
        if info.get("role") == "user" and info.get("id") in own:
            break
        window.append(message)
    return window


# What a completed assistant message's `finish` says about the turn. opencode
# completes every intermediate step of a tool-using turn, not only the last one.
# Counted in the local message store on 1.18.29: 196 assistant messages carry
# `finish: "tool-calls"` *with* a completion timestamp against 61 carrying
# `stop`, and one prompt owns seven completed `tool-calls` children followed by
# one `stop`. Reading "completed" as "finished" therefore publishes an
# intermediate step as the answer whenever a fetch lands between one step
# completing and the next being created.
#
# The predicate is positive on purpose, because the two mistakes do not cost the
# same. An unrecognised terminal value read as unresolved costs a wait the
# caller recovers from with `wait`. An unrecognised continuation value read as
# terminal publishes a partial answer and closes the turn, and nothing recovers
# that. So only values known to end a turn end one.
TERMINAL_FINISH = frozenset({"stop"})
CONTINUING_FINISH = frozenset({"tool-calls"})

REPLY_RUNNING = "running"
REPLY_TERMINAL = "terminal"
REPLY_UNRESOLVED = "unresolved"


def message_finish(message: dict) -> str | None:
    info = message.get("info")
    finish = info.get("finish") if isinstance(info, dict) else None
    return finish if isinstance(finish, str) else None


def reply_state(message: dict) -> str:
    """Whether the message this turn selected actually ends the turn.

    Selection is `find_reply`'s job; this is only the terminal decision, kept
    apart because the newest child of a prompt is routinely an intermediate step.
    """
    if not is_completed(message):
        return REPLY_RUNNING
    if has_error(message):
        # An error is terminal however it finished; the failure path reports it.
        # Asked of the error itself, not of its wording: an error whose message
        # is blank is still an error.
        return REPLY_TERMINAL
    finish = message_finish(message)
    if finish in CONTINUING_FINISH:
        return REPLY_RUNNING
    if finish in TERMINAL_FINISH:
        return REPLY_TERMINAL
    return REPLY_UNRESOLVED


def is_completed(message: dict) -> bool:
    info = message.get("info")
    if not isinstance(info, dict):
        return False
    time_ = info.get("time")
    return bool(time_.get("completed")) if isinstance(time_, dict) else False


def find_reply(messages: list[dict], mid: str, own: set[str] | None = None) -> dict | None:
    """The assistant message this turn's prompt produced, complete or not.

    Usually the child of `mid`. But when the session compacts, the server
    inserts a compaction message of its own, answers it with a summary, appends
    a synthetic `Continue if you have next steps …` user message and parents the
    real answer to *that* — so a compacted turn's answer is never our prompt's
    child. Measured on 1.18.21; `TestReplyAfterCompaction` reproduces the shape.

    A *terminal* child always wins, because the compaction machinery never
    parents anything to our prompt: that is what keeps a compaction which happens
    *after* a turn from dragging its continuation back into it. Only when there
    is no terminal child does a compaction in the window redirect this to the
    chain, which is the turn opencode compacted mid-flight. Terminal, not merely
    completed — an intermediate tool step is completed too, and letting one win
    here would hide the chain for the whole turn.
    """
    window = after_prompt(messages, mid, own or {mid})
    if window is None:
        return None
    direct = None
    chained = None
    for message in window:
        info = message.get("info")
        if not isinstance(info, dict) or info.get("role") != "assistant":
            continue
        if is_summary(message):
            continue
        if info.get("parentID") == mid:
            direct = message
        chained = message
    if direct is not None and reply_state(direct) == REPLY_TERMINAL:
        return direct
    if any(is_compaction(m) for m in window):
        return chained
    return direct


def compacting(messages: list[dict], mid: str, own: set[str] | None = None) -> bool:
    """True when the server has compacted inside this turn.

    The prompt was served, so the absence of a reply is work in progress rather
    than the silent rejection `REPLY_GRACE_S` exists to catch.
    """
    window = after_prompt(messages, mid, own or {mid})
    return any(is_compaction(m) for m in window or [])


def own_message_ids(meta: dict) -> set[str]:
    """Every prompt id this tool has sent in this conversation."""
    turns = meta.get("turns")
    if not isinstance(turns, list):
        return set()
    return {
        t["mid"] for t in turns if isinstance(t, dict) and isinstance(t.get("mid"), str)
    }


def pending_gate(meta: dict) -> tuple[str, dict] | None:
    """The permission or question this session is blocked on, if any.

    Both lists are server-wide, so they are filtered down to this session.
    """
    for kind, path in (("permission", "/permission"), ("question", "/question")):
        payload = api("GET", path, params={"directory": meta["repo"]})
        if not isinstance(payload, list):
            continue
        for item in payload:
            if isinstance(item, dict) and item.get("sessionID") == meta["session_id"]:
                return kind, item
    return None


def reply_permission(meta: dict, gate_id: str, response: str) -> None:
    api(
        "POST",
        f"/session/{meta['session_id']}/permissions/{gate_id}",
        params={"directory": meta["repo"]},
        body={"response": response},
    )


def reply_question(meta: dict, gate_id: str, labels: list[str]) -> None:
    """Answer a question gate, one inner list per question asked.

    A gate carries a *list* of questions, so `answers` is parallel to it and each
    element holds that question's chosen labels. Sending every label in a single
    inner list put them all on question one. The single-question case — the only
    one seen so far — is identical either way; the multi-question shape is
    inferred from the API's own list-of-questions, not measured, because
    provoking a two-question gate costs a real model turn.
    """
    api(
        "POST",
        f"/question/{gate_id}/reply",
        params={"directory": meta["repo"]},
        body={"answers": [[label] for label in labels]},
    )


def question_options(request: dict) -> list[list[str]]:
    """The option labels of each question in a pending gate, in order."""
    questions = request.get("questions")
    if not isinstance(questions, list):
        return []
    found: list[list[str]] = []
    for question in questions:
        if not isinstance(question, dict):
            continue
        labels: list[str] = []
        options = question.get("options")
        if isinstance(options, list):
            for option in options:
                label = option.get("label") if isinstance(option, dict) else option
                if isinstance(label, str):
                    labels.append(label)
        found.append(labels)
    return found


def abort_session(meta: dict) -> None:
    api(
        "POST",
        f"/session/{meta['session_id']}/abort",
        params={"directory": meta["repo"]},
    )


# --------------------------------------------------------------------------
# event stream
#
# Measured on 1.18.29 against the server's own OpenAPI document and a live
# subscription: `GET /event` is a v1 `text/event-stream` of `data: {json}` lines,
# each `{id, type, properties}`. It is scoped by the `directory` query parameter
# the rest of this file already passes — subscribing without it delivers nothing
# for the session, which is silent rather than an error.

# The event types that mean "something this turn cares about has changed".
# Deltas are deliberately absent: text and tool-input arrive token by token, and
# waking on those would fetch the whole message history per token.
WAKE_EVENTS = frozenset(
    {
        "message.updated",
        "permission.asked",
        "question.asked",
        "session.idle",
        "session.compacted",
        "session.error",
    }
)


class Events:
    """A subscription to one directory's event stream.

    Reduced to a wake-up signal rather than read as data. An event says something
    changed; the fetch that follows says what. That keeps every interpretation —
    `find_reply`, `compacting`, the two gate queues — on exactly the code the
    poll used, so a missed or unknown event costs latency, never correctness.

    Nothing here is required for a turn to finish. `await_turn` polls anyway
    every `POLL_MAX_S`, which is what covers a server too old to have `/event`, a
    stream that dies quietly, and the gap between a reconnect and the events it
    did not replay.
    """

    def __init__(self, directory: str, session_id: str) -> None:
        self.directory = directory
        self.session_id = session_id
        self.woken = threading.Event()
        self.connects = 0
        # Set when the server has no /event at all. Waiting then returns at once
        # and the caller's POLL_S floor governs, which is the cadence this
        # driver polled at before the stream existed — a server too old to
        # stream degrades to exactly the old behaviour, not to something slower.
        self.unavailable = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "Events":
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()

    def wait(self, timeout: float) -> bool:
        """Block until something happened, or `timeout` elapses."""
        if self.unavailable or timeout <= 0:
            return False
        fired = self.woken.wait(timeout)
        self.woken.clear()
        return fired

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._read()
            except urllib.error.HTTPError as exc:
                # Closed explicitly: an HTTPError is a response, and letting it
                # be collected implicitly is a ResourceWarning per attempt.
                code = exc.code
                exc.close()
                if code == 404:
                    self.unavailable = True
                    self.woken.set()
                    return
            except Exception:
                pass
            # Nothing is woken here. The gap a dropped stream leaves is closed
            # by the wake on the next successful connect, below — waking now
            # would fetch before the gap exists, and the reconnect is at most
            # EVENT_RETRY_S away. If the server is gone for good, the safety
            # poll in await_turn is what notices.
            if self._stop.wait(EVENT_RETRY_S):
                return

    def _read(self) -> None:
        url = base_url() + "/event?" + urllib.parse.urlencode(
            {"directory": self.directory}
        )
        request = urllib.request.Request(
            url, headers={"accept": "text/event-stream"}
        )
        with urllib.request.urlopen(request, timeout=EVENT_TIMEOUT_S) as stream:
            self.connects += 1
            # On every connect, not only when one drops. Whatever happened while
            # this was reconnecting was delivered to nobody and will not be
            # replayed, so the gap a reconnect leaves is closed by the fetch this
            # wake causes — waking only at drop time fetches before the gap
            # exists and never looks again.
            self.woken.set()
            for raw in stream:
                if self._stop.is_set():
                    return
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[5:])
                except ValueError:
                    continue
                if self._ours(event):
                    self.woken.set()

    def _ours(self, event: object) -> bool:
        if not isinstance(event, dict) or event.get("type") not in WAKE_EVENTS:
            return False
        properties = event.get("properties")
        session = (
            properties.get("sessionID") if isinstance(properties, dict) else None
        )
        # An event of a type we watch but cannot attribute is taken as ours: the
        # cost is one fetch, and the alternative is missing the turn's own end.
        return session is None or session == self.session_id


def subscribe(meta: dict) -> Events:
    """Open the stream for a conversation's session."""
    return Events(str(meta["repo"]), str(meta.get("session_id"))).start()


# --------------------------------------------------------------------------
# turn artifacts


def message_text(message: dict) -> str:
    parts = message.get("parts")
    if not isinstance(parts, list):
        return ""
    chunks = []
    for part in parts:
        if isinstance(part, dict) and part.get("type") == "text":
            text = part.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "".join(chunks)


def message_tools(message: dict) -> list[dict]:
    parts = message.get("parts")
    if not isinstance(parts, list):
        return []
    return [p for p in parts if isinstance(p, dict) and p.get("type") == "tool"]


def parse_structured(text: str) -> dict | None:
    """Pull the structured envelope out of a reply, or None if it is prose.

    Scans for JSON objects rather than for code fences. `answer` routinely
    contains fenced blocks of its own — quoted command output, a diff — and
    matching on fences finds the inner closing fence and truncates the object.
    A JSON decoder stops at the right brace because those inner fences are
    inside a quoted string.

    Nested objects are skipped by resuming the scan past each object that
    decodes, so a `{"status": ...}` mentioned inside `answer` cannot win. Of
    the top-level objects, the last one wins: a reply that echoes the schema
    before answering would otherwise be read as its own answer.
    """
    if not text.strip():
        return None
    decoder = json.JSONDecoder()
    found = None
    index = text.find("{")
    while index != -1:
        try:
            value, end = decoder.raw_decode(text, index)
        except ValueError:
            index = text.find("{", index + 1)
            continue
        if (
            isinstance(value, dict)
            and isinstance(value.get("status"), str)
            and isinstance(value.get("answer"), str)
        ):
            found = value
        index = text.find("{", end)
    return found


def message_error(message: dict) -> str | None:
    """The reason a reply failed, or None when it did not.

    Never the empty string when an error is present. A blank `data.message` used
    to be returned as-is, and every caller tests the result for truth: the turn
    then read as running rather than terminal, and its record lost the failure
    it was reporting. Whether there *is* an error is `has_error`'s question;
    this one only phrases it.
    """
    info = message.get("info")
    error = info.get("error") if isinstance(info, dict) else None
    if not isinstance(error, dict):
        return None
    for candidate in (
        error.get("data", {}).get("message")
        if isinstance(error.get("data"), dict)
        else None,
        error.get("message"),
        error.get("name"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return "opencode reported an error"


def has_error(message: dict) -> bool:
    """Whether the reply carries an error at all, however it is worded."""
    info = message.get("info")
    return isinstance(info, dict) and isinstance(info.get("error"), dict)


def message_usage(message: dict) -> dict:
    info = message.get("info")
    tokens = info.get("tokens") if isinstance(info, dict) else None
    return tokens if isinstance(tokens, dict) else {}


def message_model(message: dict) -> str:
    """Which model actually answered, as `provider/model` or `provider/model:variant`.

    Worth recording on every turn because nothing here chooses it. Without an
    explicit --model, opencode resolves the first entry of `recent` in
    ~/.local/state/opencode/model.json — what was last picked in its TUI. That
    is neither the config's `model` (which may be unset) nor the provider
    default, and it changes under you when the TUI is used. The variant is not
    inherited the same way, so a session here can run at a different reasoning
    effort than the same model in the TUI.
    """
    info = message.get("info")
    if not isinstance(info, dict):
        return ""
    provider, model = info.get("providerID"), info.get("modelID")
    if not isinstance(provider, str) or not isinstance(model, str):
        return ""
    variant = info.get("variant")
    suffix = f":{variant}" if isinstance(variant, str) and variant != "default" else ""
    return f"{provider}/{model}{suffix}"


def build_artifacts(
    conv: str,
    n: int,
    mid: str,
    started_at: float,
    started_iso: str,
    *,
    reply: dict | None = None,
    gate: tuple[str, dict] | None = None,
    cancelled: bool = False,
    failure: str | None = None,
    compacted: bool = False,
) -> dict:
    """Turn whatever ended this turn into the .md the caller reads and a .json record."""
    meta = load_meta(conv)
    reply = reply or {}
    failure = failure or (message_error(reply) if reply else None)
    # `info.structured` is where opencode would put a schema-validated reply if
    # the API's format field were usable; prefer it, fall back to the envelope
    # the prompt asked for.
    structured = None
    info = reply.get("info")
    if isinstance(info, dict) and isinstance(info.get("structured"), dict):
        structured = info["structured"]
    if structured is None and meta.get("schema"):
        structured = parse_structured(message_text(reply))
    schema_applied = structured is not None

    if cancelled:
        status = STATUS_CANCELLED
    elif gate:
        status = GATE_STATUS[gate[0]]
    elif failure:
        status = STATUS_FAILED
    elif (
        schema_applied
        and isinstance(structured, dict)
        and structured.get("status") in STRUCTURED_STATUSES
    ):
        status = structured["status"]
    else:
        status = STATUS_UNKNOWN

    record = {
        "conv": conv,
        "turn": n,
        "status": status,
        "reason": turn_reason(meta, n),
        "mid": mid,
        "session_id": meta.get("session_id"),
        "started_at": started_iso,
        "duration_s": round(time.time() - started_at, 1),
        "schema_applied": schema_applied,
        "model": message_model(reply),
        "usage": message_usage(reply),
        "failure": failure,
        "result": structured,
        "raw": message_text(reply),
        "tools": [summarize_tool(t) for t in message_tools(reply)],
        "gate": {"kind": gate[0], "request": gate[1]} if gate else None,
        "preapproved": meta.get("preapproved"),
        "compacted": compacted,
    }
    turns_dir(conv).mkdir(parents=True, exist_ok=True)
    turn_file(conv, n, "json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n"
    )
    turn_file(conv, n, "md").write_text(render_markdown(record))
    return record


def summarize_tool(part: dict) -> dict:
    state = part.get("state")
    state = state if isinstance(state, dict) else {}
    payload = state.get("input")
    payload = payload if isinstance(payload, dict) else {}
    output = state.get("output")
    if not isinstance(output, str):
        error = state.get("error")
        output = error if isinstance(error, str) else ""
    return {
        "tool": part.get("tool"),
        "status": state.get("status"),
        "command": payload.get("command") or payload.get("filePath") or "",
        "output": output[:400],
    }


def gate_command(request: dict) -> str:
    metadata = request.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    for key in ("command", "filePath", "path", "url"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value
    patterns = request.get("patterns")
    if isinstance(patterns, list) and patterns:
        return str(patterns[0])
    return ""


def fence(text: str) -> str:
    """A code fence long enough to hold `text` whatever it contains.

    A failed turn's raw body is quoted verbatim, and a reply carrying a fenced
    block of its own would otherwise close the quote early and spill the rest of
    the page out of it.
    """
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def ticked(names: list) -> str:
    return ", ".join(f"`{name}`" for name in names)


def compaction_note(record: dict) -> str:
    """One line when the server summarised the session inside this turn.

    Worth saying: the model answered from a summary of everything before this
    turn rather than from the conversation itself, and opencode appends a
    synthetic `Continue …` turn of its own afterwards, so the next turn starts
    against a session that has already moved.
    """
    if not record.get("compacted"):
        return ""
    return (
        "_opencode compacted this session mid-turn: the context before this "
        "turn was replaced by a summary._\n\n"
    )


def reason_note(record: dict) -> str:
    """The caller's own note on why it answered the gate the way it did."""
    reason = record.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return ""
    return f"_your reason, recorded here and never sent to opencode: {reason.strip()}_\n\n"


def preapproved_note(record: dict) -> str:
    """One line naming what will never reach the caller as a gate.

    A loosened ruleset that nobody can see is the failure mode this file guards
    against, so every turn says which commands were pre-approved and where.
    """
    summary = record.get("preapproved")
    if not isinstance(summary, dict):
        return ""

    def listed(key: str) -> list:
        value = summary.get(key)
        return value if isinstance(value, list) else []

    commands = listed("commands")
    anywhere = listed("anywhere")
    probes = listed("version_probes")
    # Read from the record, never generated any more. A conversation started
    # before git reads were withdrawn keeps the ruleset it was created with —
    # the server fixed it at session creation — so its turns must keep saying so.
    # Dropping the rendering with the feature would have left those sessions
    # running git without approval while their headers no longer disclosed it.
    legacy_git = listed("git_read")
    roots = listed("roots")
    read_roots = listed("read_roots")
    scope = ", ".join(str(root) for root in roots)
    clauses = []
    if commands and roots:
        tail = (
            " and on any path in the session directory (it sits inside a root)"
            if summary.get("unanchored")
            else ""
        )
        clauses.append(f"{ticked(commands)} under {scope}{tail}")
    if anywhere:
        clauses.append(f"{ticked(anywhere)} in the session directory")
    if probes:
        clauses.append(f"{ticked(probes)} asked for `--version`")
    if legacy_git:
        where = f" and via `git -C` on {scope}" if roots else ""
        clauses.append(
            f"git {ticked(legacy_git)} in the session directory{where} "
            "— a standing grant from before git reads were withdrawn, still in "
            "force for this session because its ruleset was fixed when it started"
        )
    if not clauses and not read_roots:
        return ""
    note = ""
    if read_roots:
        # A root named on the command line widens this one conversation, so it is
        # called out separately from the standing ones in the file.
        note = (
            f"_readable for this conversation via `--read-root`: "
            f"{ticked(read_roots)}._\n\n"
        )
    if not clauses:
        return note
    return note + f"_pre-approved, never gated: {'; '.join(clauses)}._\n\n"


def render_markdown(record: dict) -> str:
    model = record.get("model")
    head = (
        f"# {record['conv']} · turn {record['turn']} · {record['status']}"
        f" · {fmt_duration(record['duration_s'])} · {fmt_usage(record['usage'])}"
        + (f" · {model}" if model else "")
        + "\n\n"
        + preapproved_note(record)
        + reason_note(record)
        + compaction_note(record)
    )
    gate = record.get("gate")
    if isinstance(gate, dict):
        return head + render_gate(record, gate)
    if record["failure"]:
        raw = record["raw"]
        bars = fence(raw)
        return head + f"**opencode failed:** {record['failure']}\n\n{bars}\n{raw}\n{bars}\n"
    if record["status"] == STATUS_CANCELLED:
        return head + "The turn was aborted.\n\n" + progress_section(record)
    if not record["schema_applied"]:
        raw = record.get("raw", "")
        if raw.strip():
            return (
                head
                + (
                    "_schema not applied — body below is the model's raw final message._\n\n"
                )
                + raw
            )
        # A rejected tool call ends the turn with no prose at all. Without the
        # tool results the caller would be handed an empty file and no reason.
        return (
            head
            + ("_opencode ended the turn without a final message._\n\n")
            + progress_section(record)
        )
    parsed = record.get("result")
    body = parsed.get("answer", "") if isinstance(parsed, dict) else ""
    text = head + str(body).rstrip() + "\n"
    questions = parsed.get("open_questions") if isinstance(parsed, dict) else None
    if isinstance(questions, list) and questions:
        text += "\n## open questions\n\n"
        text += "".join(f"- {q}\n" for q in questions)
    return text


def render_gate(record: dict, gate: dict) -> str:
    request = gate.get("request")
    request = request if isinstance(request, dict) else {}
    conv = record["conv"]
    if gate.get("kind") == "question":
        text = "opencode is asking you a question and cannot continue until it is answered.\n\n"
        questions = request.get("questions")
        if isinstance(questions, list):
            for question in questions:
                if not isinstance(question, dict):
                    continue
                text += f"**{question.get('message') or question.get('title') or 'question'}**\n\n"
                options = question.get("options")
                if isinstance(options, list):
                    for option in options:
                        label = (
                            option.get("label") if isinstance(option, dict) else option
                        )
                        text += f"- `{label}`\n"
                text += "\n"
        text += progress_section(record)
        text += (
            "## how to continue\n\n"
            f"```\nask_opencode.py answer {conv} <label> [<label> ...]\n```\n"
        )
        return text

    command = gate_command(request)
    text = (
        f"opencode is waiting for approval to use **{request.get('permission')}** "
        "and cannot continue until you answer.\n\n"
    )
    if command:
        text += f"```\n{command}\n```\n\n"
    patterns = request.get("patterns")
    if isinstance(patterns, list) and patterns:
        text += "- matched patterns: " + ", ".join(f"`{p}`" for p in patterns) + "\n"
    always = request.get("always")
    if isinstance(always, list) and always:
        text += (
            "- opencode suggests generalising to "
            + ", ".join(f"`{p}`" for p in always)
            + " — `approve` deliberately does not, it answers `once` so the next "
            "call of the same shape is asked again\n"
        )
    text += "\n" + progress_section(record)
    text += (
        "## how to continue\n\n"
        f"```\nask_opencode.py approve {conv}\nask_opencode.py reject {conv} "
        '--reason "why not"\n```\n'
    )
    return text


def progress_section(record: dict) -> str:
    tools = record.get("tools")
    raw = record.get("raw", "")
    if not tools and not raw.strip():
        return ""
    text = "## this turn so far\n\n"
    if raw.strip():
        text += raw.strip() + "\n\n"
    for tool in tools if isinstance(tools, list) else []:
        text += f"- `{tool.get('tool')}` {tool.get('status')}: {tool.get('command')}\n"
        # The reason a call failed — a refusal, a ruleset block — lives here and
        # is often the only thing on the page worth reading.
        output = str(tool.get("output") or "").strip()
        if output:
            text += "  > " + output.replace("\n", "\n  > ") + "\n"
    return text + "\n"


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds or 0)
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m{seconds % 60:02d}s"


def fmt_usage(usage: dict) -> str:
    total = usage.get("total")
    if total is None:
        inp, out = usage.get("input"), usage.get("output")
        if inp is None and out is None:
            return "tok n/a"
        total = (inp or 0) + (out or 0)
    total = int(total)
    if total >= 1000:
        return f"{total / 1000:.1f}k tok"
    return f"{total} tok"


# --------------------------------------------------------------------------
# emitting


def emit(record: dict, mode: str) -> None:
    md_path = turn_file(record["conv"], record["turn"], "md")
    body = md_path.read_text(errors="replace") if md_path.exists() else ""
    model = record.get("model")
    summary = (
        f"[ask-opencode] {record['conv']} · turn {record['turn']} · {record['status']}"
        f" · {fmt_duration(record['duration_s'])} · {fmt_usage(record['usage'])}"
        + (f" · {model}" if model else "")
        + f" · {md_path}"
    )
    if mode == "result":
        print(body, end="" if body.endswith("\n") else "\n")
        return
    print(summary)
    if mode == "auto" and len(body.encode()) <= INLINE_LIMIT_B:
        print()
        print(body, end="" if body.endswith("\n") else "\n")


def exit_code_for(record: dict) -> int:
    if record["status"] == STATUS_CANCELLED:
        return EXIT_CANCELLED
    if record["status"] == STATUS_FAILED:
        return EXIT_FAILED
    return EXIT_OK


# --------------------------------------------------------------------------
# waiting


def open_turn(conv: str, n: int, mid: str, kind: str, reason: str | None = None) -> None:
    meta = load_meta(conv)
    meta["current"] = n
    meta["current_mid"] = mid
    entry = {
        "n": n,
        "mid": mid,
        "kind": kind,
        "status": "running",
        "started_at": now_iso(),
    }
    if reason:
        entry["reason"] = reason
    meta.setdefault("turns", []).append(entry)
    save_meta(conv, meta)


def turn_reason(meta: dict, n: int) -> str | None:
    """The `--reason` the caller gave when it answered the gate this turn resumes."""
    for entry in meta.get("turns", []):
        if entry.get("n") == n and isinstance(entry.get("reason"), str):
            return entry["reason"]
    return None


def turn_start(meta: dict, n: int) -> tuple[str, float]:
    """When turn `n` began, from the entry `open_turn` wrote.

    Read from meta rather than taken as now, so a turn picked back up by `wait`
    reports how long it has actually been running. `idle_seconds` derives the end
    of a turn as this stamp plus its duration, so a clock that restarted on every
    reattach would make a long turn read as having ended long before it did — and
    `stop --if-idle` reads exactly that.
    """
    for entry in meta.get("turns", []):
        if entry.get("n") != n:
            continue
        stamp = entry.get("started_at")
        if isinstance(stamp, str):
            try:
                return stamp, datetime.fromisoformat(stamp).timestamp()
            except ValueError:
                break
        break
    return now_iso(), time.time()


def close_turn(conv: str, n: int, record: dict) -> None:
    meta = load_meta(conv)
    meta["current"] = None
    # A gate keeps the message id alive: approve/reject resumes the same reply.
    if not record.get("gate"):
        meta["current_mid"] = None
    for entry in meta.get("turns", []):
        if entry.get("n") == n:
            entry["status"] = record["status"]
            entry["duration_s"] = record["duration_s"]
    save_meta(conv, meta)


def await_turn(
    conv: str,
    n: int,
    mid: str,
    wait_s: float,
    mode: str,
    events: "Events | None" = None,
) -> int:
    """Block until opencode finishes the turn or stops at a gate.

    Woken by the event stream and bounded by `POLL_MAX_S`, so an idle turn costs
    a fetch every `POLL_MAX_S` rather than one every `POLL_S`, and a stream that
    is silent, dead or absent costs latency rather than the turn. The
    caller passes `events` when it subscribed before sending the prompt; every
    other entry point — `wait`, and the resume after a gate — opens its own and
    closes the gap with the fetch at the top of the loop, which is also what lets
    a turn that finished while nothing was attached be picked up from cold.
    """
    meta = load_meta(conv)
    if events is not None:
        return _await_turn(conv, n, mid, wait_s, mode, events, meta)
    events = subscribe(meta)
    try:
        return _await_turn(conv, n, mid, wait_s, mode, events, meta)
    finally:
        events.close()


def _await_turn(
    conv: str,
    n: int,
    mid: str,
    wait_s: float,
    mode: str,
    events: "Events",
    meta: dict,
) -> int:
    own = own_message_ids(meta)
    started_iso, started_at = turn_start(meta, n)
    grace_start = time.monotonic()
    deadline = time.monotonic() + wait_s
    compacted = False
    failing_since: float | None = None
    while True:
        cycle = time.monotonic()
        try:
            gate = pending_gate(meta)
            messages = fetch_messages(meta)
        except (UsageError, ServerDown):
            # The turn belongs to the server, not to this poll. One dropped
            # request, or a server restarting under an hour-long wait, is not a
            # reason to hand the caller an environment error for a turn that is
            # still fine — but a failure that persists is.
            now = time.monotonic()
            failing_since = now if failing_since is None else failing_since
            if now - failing_since > POLL_FAIL_S:
                raise
            time.sleep(POLL_S)
            continue
        failing_since = None

        if gate:
            record = build_artifacts(
                conv,
                n,
                mid,
                started_at,
                started_iso,
                reply=find_reply(messages, mid, own),
                gate=gate,
                compacted=compacted,
            )
            close_turn(conv, n, record)
            emit(record, mode)
            return EXIT_OK

        compacted = compacted or compacting(messages, mid, own)
        reply = find_reply(messages, mid, own)
        state = reply_state(reply) if reply is not None else REPLY_RUNNING
        if state == REPLY_TERMINAL:
            record = build_artifacts(
                conv,
                n,
                mid,
                started_at,
                started_iso,
                reply=reply,
                compacted=compacted,
            )
            close_turn(conv, n, record)
            emit(record, mode)
            return exit_code_for(record)

        if reply is None and compacted:
            # A compaction is rewriting the session's context. The prompt was
            # served, so the grace check below — which reads a missing reply as
            # a rejected prompt — must not claim this turn. Keep waiting.
            pass
        elif reply is None and time.monotonic() - grace_start > REPLY_GRACE_S:
            # No assistant message at all, rather than one still being written:
            # the server took the prompt with a 204 and dropped it. Reporting
            # that beats holding the caller for the whole wait budget.
            spec, source = resolve_model(meta)
            named = f"{spec}, resolved from {source}," if spec else "the model"
            record = build_artifacts(
                conv,
                n,
                mid,
                started_at,
                started_iso,
                failure=(
                    "opencode accepted the prompt and then created no reply at "
                    f"all within {fmt_duration(REPLY_GRACE_S)}; a healthy turn "
                    "creates one in under a second. The server rejected the "
                    f"prompt outright — most often because {named} is not a "
                    "model it has."
                ),
            )
            close_turn(conv, n, record)
            emit(record, mode)
            return exit_code_for(record)

        if time.monotonic() >= deadline:
            note = " · compacting" if compacted and reply is None else ""
            if state == REPLY_UNRESOLVED:
                # Completed, no error, and a `finish` this tool does not know to
                # be terminal. Held rather than published: the turn stays intact
                # and recoverable, and the value is named so a change in
                # opencode's vocabulary is diagnosable rather than mysterious.
                note = f" · unresolved finish {message_finish(reply)!r}"
            print(
                f"[ask-opencode] {conv} · turn {n} · running{note} · "
                f"waited {fmt_duration(wait_s)} · "
                f"continue with: {SCRIPT_PATH} wait {conv}"
            )
            return EXIT_WAITING

        # Sleep until the stream says something changed, or until the next
        # moment this loop has something to decide on its own — the wait budget,
        # and the reply grace while there is still no reply. Waking only on the
        # safety poll would push both of those out to POLL_MAX_S.
        now = time.monotonic()
        until = [POLL_MAX_S, deadline - now]
        if reply is None and not compacted:
            # `and not compacted` for the same reason the grace *check* above is
            # suppressed during a compaction: the deadline cannot fire, so
            # keeping it here only pins the wait at its floor — and it does so
            # for the whole compaction, on the largest history this tool ever
            # handles, which is precisely where the stream was meant to help.
            until.append(REPLY_GRACE_S - (now - grace_start))
        events.wait(max(0.05, min(until)))
        # Never re-fetch sooner than POLL_S after the last one: a turn writing
        # its answer updates its message continuously, and the fetch is the
        # expensive half.
        idle = POLL_S - (time.monotonic() - cycle)
        if idle > 0:
            time.sleep(idle)


def wait_gate_cleared(meta: dict, kind: str, gate_id: str) -> None:
    """Block until the gate we just answered leaves the pending list.

    Without this the next poll can re-read the same request and report the gate
    a second time, which would look to the caller like opencode asked twice.
    """
    deadline = time.monotonic() + GATE_CLEAR_S
    while time.monotonic() < deadline:
        gate = pending_gate(meta)
        if gate is None or gate[1].get("id") != gate_id:
            return
        time.sleep(POLL_S)
    raise UsageError(
        f"{kind} {gate_id} is still pending after {int(GATE_CLEAR_S)}s; the reply "
        "did not take effect."
    )


# --------------------------------------------------------------------------
# subcommands


def read_stdin_text(what: str) -> str:
    if sys.stdin.isatty():
        raise UsageError(f"{what} must be piped on stdin.")
    text = sys.stdin.read()
    if not text.strip():
        raise UsageError(f"{what} is empty.")
    return text


def next_turn_number(meta: dict) -> int:
    return len(meta.get("turns", [])) + 1


def require_idle(conv: str) -> dict:
    meta = load_meta(conv)
    if meta.get("current"):
        raise UsageError(
            f"turn {meta['current']} of {conv} is still running. "
            f"Wait for it ({SCRIPT_PATH} wait {conv}) or cancel it."
        )
    return meta


def resume_after_gate(
    conv: str, wait_s: float, mode: str, reason: str | None = None
) -> int:
    meta = load_meta(conv)
    mid = meta.get("current_mid")
    if not isinstance(mid, str):
        raise UsageError(f"{conv} has no turn waiting on a gate.")
    n = next_turn_number(meta)
    open_turn(conv, n, mid, "resume", reason)
    return await_turn(conv, n, mid, wait_s, mode)


def last_gate(conv: str) -> tuple[str, dict]:
    meta = load_meta(conv)
    turns = meta.get("turns", [])
    if not turns:
        raise UsageError(f"{conv} has no turns yet.")
    record_path = turn_file(conv, turns[-1]["n"], "json")
    try:
        record = json.loads(record_path.read_text())
    except (OSError, ValueError):
        raise UsageError(f"{conv} has no stored result for turn {turns[-1]['n']}.")
    gate = record.get("gate")
    if not isinstance(gate, dict):
        raise UsageError(
            f"{conv} is not waiting on a gate (last turn was {record.get('status')})."
        )
    request = gate.get("request")
    return str(gate.get("kind")), request if isinstance(request, dict) else {}


def cmd_start(args) -> int:
    task = read_stdin_text("task text")
    repo = Path(args.repo).expanduser().resolve()
    if not repo.is_dir():
        raise UsageError(f"--repo is not a directory: {repo}")
    schema = None
    if args.schema != "none":
        schema_path = Path(args.schema).expanduser().resolve()
        if not schema_path.exists():
            raise UsageError(f"--schema file not found: {schema_path}")
        schema = str(schema_path)

    read_roots: list[Path] = []
    for given in args.read_root or ():
        root = Path(given).expanduser()
        if not root.is_absolute():
            root = (Path.cwd() / root).resolve()
        if not root.is_dir():
            raise UsageError(f"--read-root is not a directory: {root}")
        if str(root.resolve()) == "/":
            # `root_forms` strips the trailing slash and drops the empty string,
            # so this currently grants nothing while the turn header advertises
            # it. Refusing beats granting the whole filesystem and beats
            # silently granting none of it.
            raise UsageError("--read-root / is not supported; name a directory.")
        bad = policy.unsafe_root(root)
        if bad:
            raise UsageError(
                f"--read-root {root} contains {bad!r}, which is a pattern "
                "character in the permission ruleset: the root would also match "
                "sibling paths. Rename the directory or name a different root."
            )
        if root not in read_roots:
            read_roots.append(root)

    preapproved = None
    if not args.no_preapproved:
        try:
            preapproved = policy.load_preapproved()
        except policy.PreapprovedError as exc:
            raise UsageError(str(exc))

    ensure_server()
    conv = mint_conv_id(args.name, task)
    if conv_dir(conv).exists():
        raise UsageError(
            f"conversation {conv} already exists; run start again for a new id."
        )
    meta = {
        "conv": conv,
        "name": args.name,
        "created_at": now_iso(),
        "repo": str(repo),
        "write": bool(args.write),
        "model": args.model,
        "variant": args.variant,
        "agent": args.agent,
        "schema": schema,
        "port": PORT,
        "session_id": None,
        "turns": [],
        "current": None,
        "current_mid": None,
        "read_roots": [str(root) for root in read_roots],
        "preapproved": preapproved_summary(preapproved, repo, tuple(read_roots)),
    }
    check_model(meta)
    save_meta(conv, meta)
    meta["session_id"] = create_session(meta, preapproved)
    save_meta(conv, meta)

    mid = mint_message_id()
    sent = task + (schema_instruction(schema) if schema else "")
    turns_dir(conv).mkdir(parents=True, exist_ok=True)
    turn_file(conv, 1, "prompt").write_text(sent)
    open_turn(conv, 1, mid, "prompt")
    # Subscribed before the prompt goes out — as early as it can be, not as a
    # guarantee. `Events.start` only starts a thread, so the connection may not
    # exist yet when the prompt lands, and the stream replays nothing. What
    # actually covers the gap is the pair of fetches: the one at the top of
    # await_turn, and the one the first successful connect wakes. Neither
    # ordering is observable, which is why no test distinguishes them.
    events = subscribe(load_meta(conv))
    try:
        post_prompt(load_meta(conv), mid, sent)
        return await_turn(conv, 1, mid, args.wait, args.emit, events)
    finally:
        events.close()


def cmd_send(args) -> int:
    text = read_stdin_text("message text")
    meta = require_idle(args.conv)
    if meta.get("current_mid"):
        raise UsageError(
            f"{args.conv} is waiting on a gate. Answer it with approve/reject/answer "
            "before sending a new message."
        )
    ensure_server()
    n = next_turn_number(meta)
    mid = mint_message_id()
    schema = meta.get("schema")
    sent = text + (schema_instruction(str(schema)) if schema else "")
    turns_dir(args.conv).mkdir(parents=True, exist_ok=True)
    turn_file(args.conv, n, "prompt").write_text(sent)
    open_turn(args.conv, n, mid, "prompt")
    events = subscribe(load_meta(args.conv))
    try:
        post_prompt(load_meta(args.conv), mid, sent)
        return await_turn(args.conv, n, mid, args.wait, args.emit, events)
    finally:
        events.close()


def cmd_wait(args) -> int:
    meta = load_meta(args.conv)
    n = meta.get("current")
    if n:
        mid = meta.get("current_mid")
        if not isinstance(mid, str):
            raise UsageError(f"{args.conv} has a turn in flight with no message id.")
        return await_turn(args.conv, n, mid, args.wait, args.emit)
    turns = meta.get("turns", [])
    if not turns:
        raise UsageError(f"{args.conv} has no turns yet.")
    last = turns[-1]["n"]
    try:
        record = json.loads(turn_file(args.conv, last, "json").read_text())
    except (OSError, ValueError):
        raise UsageError(f"{args.conv} has no stored result for turn {last}.")
    emit(record, args.emit)
    return exit_code_for(record)


def cmd_approve(args) -> int:
    kind, request = last_gate(args.conv)
    if kind != "permission":
        raise UsageError(f"{args.conv} is waiting on a {kind}, not a permission.")
    meta = require_idle(args.conv)
    ensure_server()
    gate_id = str(request.get("id"))
    # Always `once`. `always` would persist the generalised pattern opencode
    # suggests and stop asking, which would silently remove the approval point
    # this whole tool exists to provide.
    reply_permission(meta, gate_id, "once")
    wait_gate_cleared(meta, kind, gate_id)
    return resume_after_gate(args.conv, args.wait, args.emit, args.reason)


def cmd_reject(args) -> int:
    kind, request = last_gate(args.conv)
    if kind != "permission":
        raise UsageError(f"{args.conv} is waiting on a {kind}, not a permission.")
    meta = require_idle(args.conv)
    ensure_server()
    gate_id = str(request.get("id"))
    reply_permission(meta, gate_id, "reject")
    wait_gate_cleared(meta, kind, gate_id)
    return resume_after_gate(args.conv, args.wait, args.emit, args.reason)


def cmd_answer(args) -> int:
    kind, request = last_gate(args.conv)
    if kind != "question":
        raise UsageError(f"{args.conv} is waiting on a {kind}, not a question.")
    # Checked against the gate this tool stored, before anything is sent: a label
    # the question does not offer is accepted by the server and then never clears
    # the gate, which reaches the caller as a gate-clear timeout naming nothing.
    options = question_options(request)
    if options and len(args.labels) != len(options):
        raise UsageError(
            f"{args.conv} is waiting on {len(options)} question(s) and takes one "
            f"label each, in order; got {len(args.labels)}."
        )
    for label, choices in zip(args.labels, options):
        if choices and label not in choices:
            raise UsageError(
                f"{label!r} is not one of that question's options: "
                + ", ".join(choices)
            )
    meta = require_idle(args.conv)
    ensure_server()
    gate_id = str(request.get("id"))
    reply_question(meta, gate_id, list(args.labels))
    wait_gate_cleared(meta, kind, gate_id)
    return resume_after_gate(args.conv, args.wait, args.emit)


def cmd_list() -> int:
    convs = list_convs()
    if not convs:
        print("no conversations yet.")
        return EXIT_OK
    for conv in convs:
        try:
            meta = load_meta(conv)
        except UsageError:
            continue
        turns = meta.get("turns", [])
        last = turns[-1]["status"] if turns else "-"
        flag = "running" if meta.get("current") else last
        print(f"{conv:<32} turns={len(turns):<3} {flag:<16} {meta.get('repo', '')}")
    return EXIT_OK


def cmd_models(args) -> int:
    """List what this server can actually be asked for.

    `--model` is only usable if the caller can learn a legal value, and nothing
    else here tells it one: the resolved default comes from opencode's own TUI
    state and drifts. So this prints the inventory and marks what would answer
    for `--repo` today, which is also what a wrong `--variant` is checked against.
    """
    repo = Path(args.repo).expanduser().resolve()
    if not repo.is_dir():
        raise UsageError(f"--repo is not a directory: {repo}")
    ensure_server()
    inventory = model_inventory(providers_payload())
    if not inventory:
        raise UsageError(
            "this opencode server listed no providers; nothing can be pinned "
            "with --model until it does."
        )
    spec, source = resolve_model({"repo": str(repo), "model": None})
    width = max(len(name) for name, _ in inventory)
    for name, variants in inventory:
        listed = ", ".join(variants) if variants else "none listed"
        mark = "  <- answers without --model" if name == spec else ""
        print(f"{name:<{width}}  variants: {listed}{mark}")
    print()
    if spec:
        print(f"[ask-opencode] {spec} answers here, resolved from {source}.")
    else:
        print(
            "[ask-opencode] nothing resolves a model for this directory; pass "
            "--model provider/model."
        )
    if spec is not None and spec not in {name for name, _ in inventory}:
        print(
            "[ask-opencode] that model is not in the list above, so a turn "
            "would be refused before it started."
        )
    return EXIT_OK


def cmd_show(args) -> int:
    meta = load_meta(args.conv)
    turns = meta.get("turns", [])
    if not turns:
        raise UsageError(f"{args.conv} has no turns yet.")
    n = args.turn or turns[-1]["n"]
    path = turn_file(args.conv, n, "md")
    if not path.exists():
        raise UsageError(f"turn {n} of {args.conv} has no stored result.")
    sys.stdout.write(path.read_text(errors="replace"))
    return EXIT_OK


def cancel_conversation(conv: str) -> int | None:
    """Abort the turn `conv` is holding open, if any, and close it out.

    A turn left `running` in meta wedges the conversation: `send` refuses it and
    `wait` polls out its whole budget against a reply that will never come. Both
    `cancel` and `stop --force` go through here so a stopped server never leaves
    that behind.
    """
    meta = load_meta(conv)
    n = meta.get("current")
    if not n:
        return None
    mid = meta.get("current_mid")
    abort_session(meta)
    # From meta, as `_await_turn` does. Taking the clock as now would store a
    # near-zero duration against the turn's original start, and `idle_seconds`
    # adds the two — so cancelling a long turn would make it look as though it
    # had ended near its beginning, and `stop --if-idle` would collect the
    # server immediately afterwards.
    started_iso, started_at = turn_start(meta, int(n))
    record = build_artifacts(
        conv,
        n,
        str(mid),
        started_at,
        started_iso,
        reply=find_reply(fetch_messages(meta), str(mid), own_message_ids(meta)),
        cancelled=True,
    )
    close_turn(conv, n, record)
    return int(n)


def cmd_cancel(args) -> int:
    n = cancel_conversation(args.conv)
    if n is None:
        print(f"[ask-opencode] {args.conv} has no turn in flight.")
        return EXIT_OK
    print(f"[ask-opencode] {args.conv} · turn {n} · aborted")
    return EXIT_CANCELLED


def cmd_stop(args) -> int:
    """Stop the server this tool would otherwise leave running forever."""
    if os.environ.get("ASK_OPENCODE_URL"):
        raise UsageError(
            f"ASK_OPENCODE_URL={base_url()} names a server this tool did not "
            "start; it will not stop one either."
        )
    state = probe_server()
    if state == "down":
        print(f"[ask-opencode] no server on port {PORT}")
        return EXIT_OK
    if state == "foreign":
        raise UsageError(
            f"port {PORT} is in use by something that is not an opencode "
            "server; refusing to signal it."
        )

    metas = conversations_on(PORT)
    busy = in_flight(metas)

    if args.if_idle is not None:
        # Unattended guard: report and leave, never cancel someone's turn.
        if busy:
            print(
                f"[ask-opencode] server on port {PORT} is busy · "
                f"{', '.join(busy)} · left running"
            )
            return EXIT_OK
        idle = idle_seconds(metas)
        if idle is not None and idle < args.if_idle:
            print(
                f"[ask-opencode] server on port {PORT} was active "
                f"{fmt_duration(idle)} ago · left running"
            )
            return EXIT_OK
    elif busy and not args.force:
        raise UsageError(
            f"{len(busy)} conversation(s) hold a turn open on port {PORT}: "
            f"{', '.join(busy)}. Wait for them, cancel them, or pass --force "
            "to cancel them and stop anyway."
        )

    pid = listener_pid(PORT)
    if pid is None:
        raise UsageError(
            f"a server answers on port {PORT} but its pid could not be read "
            "(is lsof installed?); stop it by hand."
        )

    cancelled = []
    if busy and args.force:
        for conv in busy:
            if cancel_conversation(conv) is not None:
                cancelled.append(conv)

    started = time.monotonic()
    how = signal_server(pid, args.timeout)
    took = fmt_duration(time.monotonic() - started)
    note = f" · cancelled {', '.join(cancelled)}" if cancelled else ""
    print(
        f"[ask-opencode] server stopped · pid {pid} · port {PORT} · "
        f"{took} · {how}{note}"
    )
    return EXIT_OK


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ask_opencode.py", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_wait_opts(p):
        p.add_argument(
            "--wait",
            type=float,
            default=DEFAULT_WAIT_S,
            help="seconds to block before reporting 'still running' (exit 10)",
        )
        p.add_argument(
            "--emit",
            choices=["auto", "summary", "result"],
            default="auto",
            help="auto: summary plus the body when it is small enough",
        )

    start = sub.add_parser("start", help="open a conversation (task text on stdin)")
    start.add_argument("--repo", default=".", help="working root handed to opencode")
    start.add_argument(
        "--write",
        action="store_true",
        help="let opencode ask to edit files (default: edits are denied)",
    )
    start.add_argument("--model", default=None, help="provider/model")
    start.add_argument(
        "--variant",
        default=None,
        help="model variant to run every turn at, e.g. xhigh — the "
        "reasoning-effort presets the provider lists for the model",
    )
    start.add_argument(
        "--agent", default=None, help="opencode agent, e.g. build or plan"
    )
    start.add_argument(
        "--schema",
        default=str(DEFAULT_SCHEMA),
        help="JSON Schema for the final message, or 'none'",
    )
    start.add_argument(
        "--name", default=None, help="readable prefix for the conversation id"
    )
    start.add_argument(
        "--read-root",
        action="append",
        metavar="PATH",
        help="a directory this conversation may read outside --repo; repeatable. "
        "Makes the root and everything under it reachable by the native read "
        "tools and, unless --no-preapproved, by the pre-approved read commands. "
        "Refused if the path contains a pattern character",
    )
    start.add_argument(
        "--no-preapproved",
        action="store_true",
        help="ignore preapproved.json; every tool call outside the read-only "
        "tools comes back as a gate. --read-root still opens the root to the "
        "native read tools",
    )
    add_wait_opts(start)
    start.set_defaults(func=cmd_start)

    send = sub.add_parser("send", help="next turn of a conversation (message on stdin)")
    send.add_argument("conv")
    add_wait_opts(send)
    send.set_defaults(func=cmd_send)

    wait = sub.add_parser("wait", help="keep waiting on a running turn")
    wait.add_argument("conv")
    add_wait_opts(wait)
    wait.set_defaults(func=cmd_wait)

    approve = sub.add_parser(
        "approve", help="allow the pending tool call, then keep waiting"
    )
    approve.add_argument("conv")
    approve.add_argument(
        "--reason",
        default=None,
        help="why, recorded in the resumed turn's record and rendered in its "
        ".md; opencode never sees it",
    )
    add_wait_opts(approve)
    approve.set_defaults(func=cmd_approve)

    reject = sub.add_parser(
        "reject", help="refuse the pending tool call, then keep waiting"
    )
    reject.add_argument("conv")
    reject.add_argument(
        "--reason",
        default=None,
        help="why, recorded in the resumed turn's record and rendered in its "
        ".md; opencode never sees it",
    )
    add_wait_opts(reject)
    reject.set_defaults(func=cmd_reject)

    answer = sub.add_parser(
        "answer", help="answer the pending question, then keep waiting"
    )
    answer.add_argument("conv")
    answer.add_argument(
        "labels", nargs="+", help="chosen option labels, in question order"
    )
    add_wait_opts(answer)
    answer.set_defaults(func=cmd_answer)

    listing = sub.add_parser("list", help="list conversations")
    listing.set_defaults(func=lambda _: cmd_list())

    models = sub.add_parser(
        "models", help="list the models and variants this server can reach"
    )
    models.add_argument(
        "--repo",
        default=".",
        help="resolve the default model as it would resolve for this directory",
    )
    models.set_defaults(func=cmd_models)

    show = sub.add_parser("show", help="print a stored turn")
    show.add_argument("conv")
    show.add_argument("--turn", type=int, default=None)
    show.set_defaults(func=cmd_show)

    cancel = sub.add_parser("cancel", help="abort the running turn")
    cancel.add_argument("conv")
    cancel.set_defaults(func=cmd_cancel)

    stop = sub.add_parser("stop", help="stop the opencode server on this port")
    stop.add_argument(
        "--force",
        action="store_true",
        help="cancel turns still in flight instead of refusing to stop",
    )
    stop.add_argument(
        "--if-idle",
        type=float,
        default=None,
        metavar="SECONDS",
        help="stop only if no conversation on this port has been active for "
        "that long — for an unattended sweep; never cancels a live turn",
    )
    stop.add_argument(
        "--timeout",
        type=float,
        default=STOP_TIMEOUT_S,
        help="seconds to wait for SIGTERM before escalating to SIGKILL",
    )
    stop.set_defaults(func=cmd_stop)

    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv[1:])
    try:
        return args.func(args)
    except UsageError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ENV
    except ServerDown as exc:
        print(f"ERROR: opencode server unreachable: {exc}", file=sys.stderr)
        return EXIT_ENV
    except KeyboardInterrupt:
        return EXIT_CANCELLED


if __name__ == "__main__":
    sys.exit(main(sys.argv))
