#!/usr/bin/env python3
"""Stand-in for `opencode serve` so the test suite makes no real model calls.

Implements the slice of the v1 HTTP API that ask_opencode.py depends on, driven
by a script file so a test can stage gates, failures and delays.

Env knobs:
  FAKE_OPENCODE_PORT    port to bind (required)
  FAKE_OPENCODE_SCRIPT  JSON file describing each turn (optional)
  FAKE_OPENCODE_LOG     file to append received requests to, one JSON per line
  FAKE_OPENCODE_FLAKY   fail this many message reads with 503 before serving them,
                        so a poll can be made to blip the way a real one does
  FAKE_OPENCODE_NO_EVENT     404 the /event endpoint, like a server too old to stream
  FAKE_OPENCODE_EVENT_SILENT serve /event but never send anything on it
  FAKE_OPENCODE_EVENT_DROP   close the stream after this many events, to force a reconnect
  FAKE_OPENCODE_EVENT_STORM  emit this many extra message.updated events per prompt,
                             so a burst can be told apart from a burst of fetches
  FAKE_OPENCODE_COMPLETE_ON_DROP  hold the reply until a stream has actually been
                             dropped, so the completion is inside the gap by
                             construction rather than by timing

Script shape:
  {"turns": [
     {"gates": [<permission or question request>, ...],
      "text": "...", "structured": {...}, "error": {...},
      "tools": [...], "tokens": {...}, "delay_s": 0, "hang": false,
      "no_reply": false, "compaction": false, "gate_delay_s": 0,
      "tool_steps": 0, "finish": "stop"},
     ...
   ],
   "config": {"model": "provider/model"},
   "providers": {"providers": [{"id": "...", "models": {...}}]}}

`tool_steps` reproduces what a tool-using turn looks like on the wire: that many
completed assistant messages carrying `finish: "tool-calls"`, parented to the
caller's prompt, before the final one exists at all. With `delay_s` the final
message is created only after the delay, which leaves the window in which the
newest child of the prompt is a completed intermediate step — the window a
driver that equates "completed" with "finished" publishes as the answer.

A turn with no entry in the script falls back to a converged reply. Gates are
surfaced one at a time and cleared as they are answered; the reply completes
once the last one is gone.

`hang` and `no_reply` are different failures: `hang` leaves an assistant message
open forever, which is what a slow turn looks like, while `no_reply` never
creates one, which is what the real server does with a prompt it cannot serve.

`compaction` reproduces what the server does when the session outgrows its
context budget: it inserts a compaction message of its own, answers it with a
summary, appends a synthetic `Continue …` user message and parents the real
answer to that — so the caller's prompt never gets a child. With `delay_s` the
compaction message lands first and the rest follows later, which is what the
driver sees while a compaction is still running.
"""

import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TypedDict, cast
from urllib.parse import parse_qs, urlparse

Json = dict[str, object]


# The double stays independent of the driver, so the narrowing it needs for a
# decoded script or request body is written here rather than imported from it.
def as_dict(value: object) -> Json:
    return cast("Json", value) if isinstance(value, dict) else {}


def as_object(value: object) -> Json | None:
    return cast("Json", value) if isinstance(value, dict) else None


def as_list(value: object) -> list[object]:
    return cast("list[object]", value) if isinstance(value, list) else []


def as_str(value: object, default: str = "") -> str:
    return value if isinstance(value, str) else default


def as_int(value: object, default: int = 0) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def as_float(value: object, default: float = 0.0) -> float:
    return (
        float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default
    )


class Message(TypedDict):
    info: Json
    parts: list[Json]


class Session(TypedDict):
    permission: object
    directory: str | None
    messages: list[Message]


class Event(TypedDict):
    _directory: str | None
    event: Json


class State(TypedDict):
    sessions: dict[str, Session]
    gates: list[Json]
    turn: int
    seq: int
    events: list[Event]
    on_drop: tuple[str, str, Json] | None


DEFAULT_TOKENS = {
    "total": 2000,
    "input": 1200,
    "output": 800,
    "reasoning": 0,
    "cache": {"read": 0, "write": 0},
}
DEFAULT_PROVIDERS: Json = {
    "providers": [{"id": "test-provider", "models": {"model-a": {}, "model-b": {}}}],
    "default": {"test-provider": "model-b"},
}

STATE: State = {
    "sessions": {},  # sid -> {"permission": [...], "directory": str, "messages": [...]}
    "gates": [],  # pending gate requests, each with _kind
    "turn": 0,  # how many prompts have been received
    "seq": 0,
    "events": [],  # broadcast to every /event subscriber, append-only
    "on_drop": None,  # (sid, mid, spec) held until a stream is dropped
}
LOCK = threading.Lock()

# Message reads still owed a 503. The driver polls this endpoint, so failing it
# a few times is what a dropped request or a restarting server looks like from
# the other side.
FLAKY = {"left": int(os.environ.get("FAKE_OPENCODE_FLAKY", 0))}


def flaky_read() -> bool:
    if FLAKY["left"] <= 0:
        return False
    FLAKY["left"] -= 1
    return True


def script() -> Json:
    path = os.environ.get("FAKE_OPENCODE_SCRIPT")
    if not path or not os.path.exists(path):
        return {"turns": []}
    with open(path) as handle:
        return as_dict(json.load(handle))


def turn_spec(index: int) -> Json:
    turns = as_list(script().get("turns"))
    if index < len(turns):
        return as_dict(turns[index])
    return {}


# The two endpoints take different model shapes, per opencode's own OpenAPI
# document on 1.18.29, and a fake that accepted either would let a "cleanup"
# that unified them pass its tests. Both are required-field checks here.
MODEL_KEYS = {"/session": {"providerID", "id"}, "prompt_async": {"providerID", "modelID"}}


def model_shape_error(where: str, body: object) -> str | None:
    raw = as_dict(body).get("model")
    if raw is None:
        return None
    model = as_object(raw)
    if model is None:
        return f"{where}: model must be an object"
    missing = MODEL_KEYS[where] - set(model)
    if missing:
        return f"{where}: model is missing {sorted(missing)}"
    return None


def emit(kind: str, sid: str) -> None:
    """Queue an event for whatever is subscribed to /event.

    Shape taken from opencode 1.18.29: `{id, type, properties}` with the session
    id inside `properties`, delivered as `data: {json}` lines. The directory is
    carried alongside, not inside the event, because the real stream is scoped by
    the subscriber's `directory` query parameter rather than by the payload.
    """
    session = STATE["sessions"].get(sid)
    STATE["events"].append(
        {
            "_directory": session["directory"] if session else None,
            "event": {"id": next_id("evt"), "type": kind, "properties": {"sessionID": sid}},
        }
    )


def log_request(method: str, path: str, body: object) -> None:
    path_env = os.environ.get("FAKE_OPENCODE_LOG")
    if not path_env:
        return
    with open(path_env, "a") as handle:
        handle.write(json.dumps({"method": method, "path": path, "body": body}) + "\n")


def next_id(prefix: str) -> str:
    STATE["seq"] += 1
    return f"{prefix}_{STATE['seq']:012d}"


def reply_text(spec: Json) -> str:
    """What the model actually writes.

    The real server cannot carry a schema on the request — storing anything in
    prompt_async's `format` makes later message reads fail — so the envelope
    arrives as a fenced block in the reply text, and the fake mirrors that.
    """
    body = as_str(spec.get("text"), "here is the answer")
    if spec.get("unstructured"):
        return body
    envelope = {
        "status": spec.get("status", "converged"),
        "answer": body,
        "open_questions": spec.get("open_questions", []),
    }
    return f"{body}\n\n```json\n{json.dumps(envelope)}\n```\n"


def complete_reply(sid: str, mid: str, spec: Json, rejected: bool = False) -> None:
    """Mark the assistant message for `mid` finished, per the turn's spec."""
    for message in STATE["sessions"][sid]["messages"]:
        info = message["info"]
        stamps = as_dict(info.get("time"))
        if (
            info.get("parentID") == mid
            and info.get("role") == "assistant"
            # The one still open. Picking the first child would re-stamp a
            # completed intermediate step as the turn's `stop` and leave the
            # real answer unfinished for good.
            and stamps.get("completed") is None
        ):
            stamps["completed"] = int(time.time() * 1000)
            info["finish"] = spec.get("finish", "stop")
            emit("message.updated", sid)
            emit("session.idle", sid)
            if rejected:
                # The real server ends the turn with no prose at all when a
                # tool call is refused; only the failed tool part remains.
                message["parts"] = [
                    {"type": "text", "text": ""},
                    {
                        "type": "tool",
                        "tool": "bash",
                        "state": {
                            "status": "error",
                            "input": {"command": "date +%Y"},
                            "error": "The user rejected permission to use this specific tool call.",
                        },
                    },
                ]
                return
            if spec.get("error"):
                info["error"] = spec["error"]
            elif "structured" in spec:
                # Forward compatibility: if a later opencode fills this in, the
                # driver prefers it over parsing the text.
                info["structured"] = spec["structured"]
            return


CONTINUE_TEXT = (
    "Continue if you have next steps, or stop and ask for "
    "clarification if you are unsure how to proceed."
)


def open_compaction(sid: str, mid: str) -> None:
    """The user message the server inserts when it compacts the session.

    Shape taken from a real compaction on 1.18.21: a user message carrying a
    `compaction` part, with no reply of its own yet. `TestReplyAfterCompaction`
    asserts against the same shape.
    """
    del mid
    now = int(time.time() * 1000)
    STATE["sessions"][sid]["messages"].append(
        {
            "info": {
                "id": next_id("msg"),
                "sessionID": sid,
                "role": "user",
                "summary": {"diffs": []},
                "time": {"created": now},
            },
            "parts": [{"type": "compaction", "auto": True, "overflow": False}],
        }
    )
    emit("session.compacted", sid)


def close_compaction(sid: str, mid: str, spec: Json) -> None:
    """The rest of a compaction: summary, synthetic continue, then the answer.

    The answer is parented to the continue message the server writes, never to
    the caller's prompt — which is the whole reason find_reply cannot match on
    parentID alone.
    """
    del mid
    session = STATE["sessions"][sid]
    now = int(time.time() * 1000)
    compaction_id = next(
        m["info"]["id"]
        for m in reversed(session["messages"])
        if any(part.get("type") == "compaction" for part in m["parts"])
    )
    session["messages"].append(
        {
            "info": {
                "id": next_id("msg"),
                "sessionID": sid,
                "role": "assistant",
                "parentID": compaction_id,
                "summary": True,
                "time": {"created": now, "completed": now},
                "providerID": spec.get("providerID", "test-provider"),
                "modelID": spec.get("modelID", "model-a"),
                "tokens": DEFAULT_TOKENS,
                "cost": 0,
                "finish": "stop",
            },
            "parts": [{"type": "text", "text": "## Objective\nsummary of the session so far"}],
        }
    )
    continue_id = next_id("msg")
    session["messages"].append(
        {
            "info": {
                "id": continue_id,
                "sessionID": sid,
                "role": "user",
                "summary": {"diffs": []},
                "time": {"created": now},
            },
            "parts": [
                {
                    "type": "text",
                    "text": CONTINUE_TEXT,
                    "synthetic": True,
                    "metadata": {"compaction_continue": True},
                }
            ],
        }
    )
    session["messages"].append(
        {
            "info": {
                "id": next_id("msg"),
                "sessionID": sid,
                "role": "assistant",
                "parentID": continue_id,
                "time": {"created": now, "completed": now},
                "providerID": spec.get("providerID", "test-provider"),
                "modelID": spec.get("modelID", "model-a"),
                "variant": spec.get("variant", "default"),
                "finish": "stop",
                "tokens": spec.get("tokens", DEFAULT_TOKENS),
                "cost": 0,
            },
            "parts": [{"type": "text", "text": reply_text(spec)}],
        }
    )
    emit("message.updated", sid)
    emit("session.idle", sid)


def _locked_close_compaction(sid: str, mid: str, spec: Json) -> None:
    with LOCK:
        close_compaction(sid, mid, spec)


def _storm(sid: str, count: int) -> None:
    """Emit `count` events spaced across the turn, one wake each.

    Spread rather than burst: a burst all lands before the driver's first fetch
    and the wake flag coalesces it on its own, which would let a test claiming to
    pin the fetch floor pass without one.
    """
    for _ in range(count):
        time.sleep(0.02)
        with LOCK:
            emit("message.updated", sid)


def start_turn(sid: str, mid: str) -> None:
    index = STATE["turn"]
    STATE["turn"] += 1
    spec = turn_spec(index)
    session = STATE["sessions"][sid]
    now = int(time.time() * 1000)
    session["messages"].append(
        {
            "info": {"id": mid, "sessionID": sid, "role": "user", "time": {"created": now}},
            "parts": [{"type": "text", "text": spec.get("prompt_echo", "")}],
        }
    )
    if spec.get("compaction"):
        open_compaction(sid, mid)
        delay = as_float(spec.get("delay_s"))
        if delay:
            threading.Timer(delay, lambda: _locked_close_compaction(sid, mid, spec)).start()
        else:
            close_compaction(sid, mid, spec)
        return
    if spec.get("no_reply"):
        # A prompt the server cannot serve is accepted with 204 and then no
        # assistant message is ever created for it.
        return
    storm = int(os.environ.get("FAKE_OPENCODE_EVENT_STORM", 0))
    if storm:
        threading.Thread(target=_storm, args=(sid, storm), daemon=True).start()
    steps = as_int(spec.get("tool_steps"))
    for _ in range(steps):
        session["messages"].append(
            {
                "info": {
                    "id": next_id("msg"),
                    "sessionID": sid,
                    "role": "assistant",
                    "parentID": mid,
                    "finish": "tool-calls",
                    "time": {"created": now, "completed": now},
                    "providerID": spec.get("providerID", "test-provider"),
                    "modelID": spec.get("modelID", "model-a"),
                    "variant": spec.get("variant", "default"),
                    "tokens": spec.get("tokens", DEFAULT_TOKENS),
                    "cost": 0,
                },
                "parts": [
                    {"type": "text", "text": "calling a tool"},
                    {
                        "type": "tool",
                        "tool": "read",
                        "state": {
                            "status": "completed",
                            "input": {"filePath": "note.txt"},
                            "output": "...",
                        },
                    },
                ],
            }
        )
        # The real server announces a completed step the same way it announces a
        # completed turn, which is what puts a driver inside the window.
        emit("message.updated", sid)
    delay = as_float(spec.get("delay_s"))
    if steps and delay:
        # The answer does not exist yet: for `delay` seconds the newest child of
        # this prompt is a completed intermediate step. Deferred through the same
        # function the ordinary path uses, so `hang` and `gates` still apply.
        threading.Timer(delay, lambda: _locked_deliver(sid, mid, spec)).start()
        return
    deliver_reply(sid, mid, spec)


def deliver_reply(sid: str, mid: str, spec: Json, delayed: bool = False) -> None:
    """The turn's answer, the gates it may stop at, and its completion.

    One function for both paths. `delayed` only says the wait already happened,
    so a deferred answer is not delayed a second time.
    """
    session = STATE["sessions"][sid]
    now = int(time.time() * 1000)
    parts: list[Json] = [{"type": "text", "text": reply_text(spec)}]
    parts.extend(
        {
            "type": "tool",
            "tool": as_dict(tool).get("tool", "bash"),
            "state": as_dict(tool).get("state", {"status": "completed"}),
        }
        for tool in as_list(spec.get("tools"))
    )
    session["messages"].append(
        {
            "info": {
                "id": next_id("msg"),
                "sessionID": sid,
                "role": "assistant",
                "parentID": mid,
                "time": {"created": now, "completed": None},
                "providerID": spec.get("providerID", "test-provider"),
                "modelID": spec.get("modelID", "model-a"),
                "variant": spec.get("variant", "default"),
                "tokens": spec.get("tokens", DEFAULT_TOKENS),
                "cost": 0,
            },
            "parts": parts,
        }
    )
    emit("message.updated", sid)
    gate_delay = as_float(spec.get("gate_delay_s"))
    if gate_delay:
        # A gate that is not already pending when the driver first looks, so
        # only the stream can tell it one arrived.
        threading.Timer(gate_delay, lambda: _locked_raise_gates(sid, spec)).start()
        return
    raise_gates(sid, spec)
    if spec.get("hang"):
        return
    if STATE["gates"]:
        return
    if os.environ.get("FAKE_OPENCODE_COMPLETE_ON_DROP"):
        # Held, not timed: the reply is finished when a subscriber is dropped,
        # so it lands inside the disconnected gap by construction. A timer only
        # makes that likely — a late subscription could still watch it happen.
        # Placed here, after the message and its creation event exist, so the
        # subscriber has something to receive and be dropped on.
        STATE["on_drop"] = (sid, mid, spec)
        return
    delay = 0.0 if delayed else as_float(spec.get("delay_s"))
    if delay:
        threading.Timer(delay, lambda: _locked_complete(sid, mid, spec)).start()
    else:
        complete_reply(sid, mid, spec)


def raise_gates(sid: str, spec: Json) -> None:
    for gate in as_list(spec.get("gates")):
        entry = dict(as_dict(gate))
        entry.setdefault(
            "id",
            next_id("per" if entry.get("_kind", "permission") == "permission" else "que"),
        )
        entry["sessionID"] = sid
        STATE["gates"].append(entry)
        emit(
            "question.asked" if entry.get("_kind") == "question" else "permission.asked",
            sid,
        )


def _locked_raise_gates(sid: str, spec: Json) -> None:
    with LOCK:
        raise_gates(sid, spec)


def _locked_deliver(sid: str, mid: str, spec: Json) -> None:
    with LOCK:
        deliver_reply(sid, mid, spec, delayed=True)


def _locked_complete(sid: str, mid: str, spec: Json) -> None:
    with LOCK:
        complete_reply(sid, mid, spec)


def clear_gate(gate_id: str, response: str = "once") -> None:
    """Drop the answered gate; complete the reply once none are left."""
    remaining = [g for g in STATE["gates"] if g.get("id") != gate_id]
    STATE["gates"] = remaining
    if remaining:
        return
    index = STATE["turn"] - 1
    spec = turn_spec(index)
    if spec.get("hang"):
        return
    for sid, session in STATE["sessions"].items():
        for message in session["messages"]:
            info = message["info"]
            stamps = as_dict(info.get("time"))
            if info.get("role") == "assistant" and stamps.get("completed") is None:
                complete_reply(
                    sid, str(info.get("parentID")), spec, rejected=(response == "reject")
                )
                return


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # `format` shadows a builtin, but the name is the base class's, not ours.
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Keep the test output clean."""
        del format, args

    def _send(self, code: int, payload: object) -> None:
        body = b"" if payload is None else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _read_body(self) -> object:
        length = int(self.headers.get("content-length") or 0)
        if not length:
            return None
        try:
            return json.loads(self.rfile.read(length))
        except ValueError:
            return None

    def _write_event(self, event: Json) -> None:
        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        self.wfile.flush()

    def _event_stream(self) -> None:
        """Serve /event the way opencode does: SSE until the client goes away.

        Delimited by connection close rather than content-length, which is what
        the real one does and what the driver's line-by-line read expects.
        """
        if os.environ.get("FAKE_OPENCODE_NO_EVENT"):
            return self._send(404, {"error": "not found"})
        # Scoped by `directory`, as measured on 1.18.29: a subscription without
        # it receives nothing for the session, silently rather than as an error.
        # A fake that broadcast regardless could not tell that failure from a
        # working stream, which is the whole point of the parameter.
        wanted = (parse_qs(urlparse(self.path).query).get("directory") or [None])[0]
        drop_after = int(os.environ.get("FAKE_OPENCODE_EVENT_DROP", 0))
        silent = bool(os.environ.get("FAKE_OPENCODE_EVENT_SILENT"))
        self.close_connection = True
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.end_headers()
        # From here on, not from the beginning: a real SSE stream replays
        # nothing from before it was opened, and a fake that replayed the
        # backlog would hide the gap a reconnect leaves.
        with LOCK:
            cursor = len(STATE["events"])
        sent = 0
        deadline = time.time() + 120
        try:
            self._write_event({"id": "evt_hello", "type": "server.connected", "properties": {}})
            with LOCK:
                held = STATE["on_drop"]
                STATE["on_drop"] = None
            if held:
                # Drop *this* subscriber, whenever it arrived, and only then
                # finish the reply. Waiting for an ordinary event to arrive
                # first raced the subscription: an event emitted before the
                # stream connected is never replayed, so the drop would never
                # come and the reply would never land.
                threading.Timer(0.5, lambda: _locked_complete(*held)).start()
                return None
            while time.time() < deadline:
                if silent:
                    time.sleep(0.05)
                    continue
                with LOCK:
                    pending = STATE["events"][cursor:]
                    cursor = len(STATE["events"])
                for entry in pending:
                    if wanted is None or entry["_directory"] != wanted:
                        continue
                    self._write_event(entry["event"])
                    sent += 1
                    if drop_after and sent >= drop_after:
                        return None
                time.sleep(0.02)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return None

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        log_request("GET", path, None)
        if path == "/event":
            # Outside the lock: this one is held open for the life of the turn.
            return self._event_stream()
        with LOCK:
            if path == "/session/status":
                busy: dict[str, Json] = {}
                for sid, session in STATE["sessions"].items():
                    for message in session["messages"]:
                        info = message["info"]
                        stamps = as_dict(info.get("time"))
                        if info.get("role") == "assistant" and stamps.get("completed") is None:
                            busy[sid] = {"type": "busy"}
                return self._send(200, busy)
            if path == "/config":
                return self._send(200, script().get("config", {}))
            if path == "/config/providers":
                return self._send(200, script().get("providers", DEFAULT_PROVIDERS))
            if path == "/permission":
                return self._send(
                    200,
                    [
                        self._public(g)
                        for g in STATE["gates"]
                        if g.get("_kind", "permission") == "permission"
                    ],
                )
            if path == "/question":
                return self._send(
                    200, [self._public(g) for g in STATE["gates"] if g.get("_kind") == "question"]
                )
            parts = path.strip("/").split("/")
            if len(parts) == 3 and parts[0] == "session" and parts[2] == "message":
                if flaky_read():
                    return self._send(503, {"error": "transient"})
                session = STATE["sessions"].get(parts[1])
                if session is None:
                    return self._send(404, {"error": "no such session"})
                return self._send(200, session["messages"])
        return self._send(404, {"error": "not found"})

    @staticmethod
    def _public(gate: Json) -> Json:
        return {k: v for k, v in gate.items() if k != "_kind"}

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        body = self._read_body()
        log_request("POST", path, body)
        with LOCK:
            parts = path.strip("/").split("/")
            if path == "/session":
                bad = model_shape_error("/session", body)
                if bad:
                    return self._send(400, {"error": bad})
                sid = next_id("ses")
                request = as_dict(body)
                STATE["sessions"][sid] = {
                    "permission": request.get("permission"),
                    # From the query string, which is where the driver and the
                    # real API both carry it — not from the body.
                    "directory": (parse_qs(urlparse(self.path).query).get("directory") or [None])[
                        0
                    ],
                    "messages": [],
                }
                return self._send(200, {"id": sid, "permission": request.get("permission")})
            if len(parts) == 3 and parts[0] == "session" and parts[2] == "prompt_async":
                sid = parts[1]
                if sid not in STATE["sessions"]:
                    return self._send(404, {"error": "no such session"})
                bad = model_shape_error("prompt_async", body)
                if bad:
                    return self._send(400, {"error": bad})
                request = as_dict(body)
                start_turn(sid, str(request.get("messageID")))
                return self._send(204, None)
            if len(parts) == 4 and parts[0] == "session" and parts[2] == "permissions":
                request = as_dict(body)
                clear_gate(parts[3], str(request.get("response", "once")))
                return self._send(200, True)
            if len(parts) == 3 and parts[0] == "question" and parts[2] == "reply":
                clear_gate(parts[1])
                return self._send(200, True)
            if len(parts) == 3 and parts[0] == "session" and parts[2] == "abort":
                STATE["gates"] = []
                return self._send(200, True)
        return self._send(404, {"error": "not found"})


def main() -> int:
    port = int(os.environ["FAKE_OPENCODE_PORT"])
    if os.environ.get("FAKE_OPENCODE_IGNORE_SIGTERM"):
        # A server that will not go quietly, so `stop` has something real to
        # escalate against.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
