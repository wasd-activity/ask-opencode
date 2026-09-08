# ask-opencode

Hold a multi-turn conversation with the [opencode](https://opencode.ai) coding agent from another
program — one turn per call, with every tool call opencode wants to make routed back to you for
approval.

`ask_opencode.py` drives an `opencode serve` process over HTTP. Each command blocks until the turn
lands, prints one summary line, and writes the answer to a file:

```console
$ ask_opencode.py start --repo ~/src/myproject <<'TASK'
Why does the retry loop in fetch.py give up after one attempt?
TASK
[ask-opencode] retry-loop-8f2a1c · turn 1 · needs_permission · 6s · 1.4k tok · openai/gpt-5 · <state>/1.md
```

Three properties are the point:

- **The reply is never paraphrased.** opencode's text lands in a file you read yourself.
- **You approve every tool call**, apart from a read-only allowlist you configure.
- **The caller never polls.** The driver waits on opencode's event stream, falling back to polling when the stream is silent or unavailable, and returns when the turn lands.

The server owns the conversation, not the driver, so a caller that is killed or times out loses
nothing — `ask_opencode.py wait <conv-id>` reattaches to the same turn.

## Requirements

- Python 3.11 or newer, standard library only
- The `opencode` CLI on `PATH` (developed against 1.18.29)

## Install

Clone the repository anywhere and run `ask_opencode.py` from it. `SKILL.md` is the entry point if
you want to install it as an agent skill: point your agent's skill directory at this one.

## Commands

```
ask_opencode.py start   [--repo DIR] [--write] [--model provider/model] [--variant V]
                        [--agent A] [--schema FILE|none] [--name NAME]
                        [--read-root PATH ...] [--no-preapproved]
                        [--wait S] [--emit MODE]
ask_opencode.py send    <conv-id>             next turn; message on stdin
ask_opencode.py wait    <conv-id>             keep waiting on a running turn
ask_opencode.py approve <conv-id>             allow the pending tool call, then keep waiting
ask_opencode.py reject  <conv-id> [--reason]  refuse it, then keep waiting
ask_opencode.py answer  <conv-id> <label...>  answer a pending question
ask_opencode.py models  [--repo DIR]          list the models this server can reach
ask_opencode.py list                          list conversations
ask_opencode.py show    <conv-id> [--turn N]  print a stored turn
ask_opencode.py cancel  <conv-id>             abort the running turn
ask_opencode.py stop    [--force] [--if-idle SECONDS]
```

Task and message text arrive on stdin. `--emit result` prints the answer instead of a summary line,
which is what you want when driving it by hand.

`--repo` is the only thing that sets opencode's working directory, so pass it explicitly. Without
`--model`, opencode resolves the model itself and that choice can drift, so `models` lists the legal
values and every summary line names the one that actually answered.

## Permissions

Each session is created with an explicit ruleset, so nothing here touches your global opencode
configuration. The native `read`, `grep`, `glob` and `list` tools are allowed. Edits are denied
unless you pass `--write`, which turns them into approval requests. Paths outside the configured
roots are denied, as is a short blacklist of irreversible commands. Everything else becomes an
approval request that ends the turn and hands you the exact command to judge.

`preapproved.json`, beside the script, layers a few read-only shell commands on top so that reading
the same file does not cost an approval every turn. It is a local file: copy
`preapproved.example.json` to create one, or leave it absent and every shell call is an approval
request. Entries are literal command prefixes anchored to named roots, and the file cannot widen its
own grant — a glob, a redirect or a chain in an entry is refused, and a command head must appear in
`policy.REVIEWED_HEADS`, reviewed for what a pre-approved read of it can be turned into, before it
is admitted at all.

Two heads were reviewed and **withdrawn**, and the loader refuses to re-admit them. `rg` and `git`
can run a program of their own through their argument lists — `rg --pre`, and git's configured
external-diff and textconv helpers — and a textual flag guard cannot stop that, because shell
quoting spells one argument in unboundedly many ways: `rg --p''re CMD` slipped past a guard written
for `--pre` while the shell handed ripgrep `--pre` regardless. `sed` and `perl` are out for the same
kind of reason. All four reach you as ordinary approval requests instead, and
`policy.WITHDRAWN_HEADS` records each with what was measured.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `ASK_OPENCODE_STATE_DIR` | under `$HOME` | Conversation storage |
| `ASK_OPENCODE_BIN` | `opencode` on `PATH` | opencode binary |
| `ASK_OPENCODE_URL` | `http://127.0.0.1:$PORT` | Server to drive; set it to disable auto-start |
| `ASK_OPENCODE_PORT` | 47399 | Port for the managed server |
| `ASK_OPENCODE_WAIT_S` | 3500 | Default wait budget |
| `ASK_OPENCODE_POLL_S` | 0.4 | Floor between two fetches |
| `ASK_OPENCODE_POLL_MAX_S` | 30 | Ceiling: fetch anyway after this, whatever the stream says |
| `ASK_OPENCODE_POLL_FAIL_S` | 30 | How long the poll may keep failing before the wait gives up |
| `ASK_OPENCODE_EVENT_RETRY_S` | 2 | How long a dropped event stream waits before reconnecting |
| `ASK_OPENCODE_REPLY_GRACE_S` | 20 | How long a turn may produce no reply at all before it counts as dropped |
| `ASK_OPENCODE_PREAPPROVED` | `preapproved.json` beside the script | Pre-approval file to load |

`ask_opencode.py --help` covers the rest.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Turn completed, or stopped at a gate awaiting your answer |
| 2 | opencode failed — the reason is in the turn's `.md` |
| 3 | Cancelled |
| 4 | Usage or environment error |
| 10 | Wait budget elapsed; the turn is still running, continue with `wait` |

A gate is exit 0 on purpose: nothing went wrong, the turn is simply waiting on you.

## State

Each conversation gets a directory under `$ASK_OPENCODE_STATE_DIR`:

```
<conv-id>/
├── meta.json          conversation config, session id, turn log
└── turns/
    ├── N.prompt       what was sent
    ├── N.md           the answer, or the pending gate  <- this one
    └── N.json         the full record
```

Nothing is written into the package itself.

**Those files hold whatever the conversation held.** Prompts, opencode's replies, the paths it
touched, the commands it asked to run and part of their output all land there, and the managed
server writes a `server.log` beside them. Treat the state directory as private: do not commit it,
and redact before pasting a turn into an issue or a bug report.

## The server

`start` reuses a server on `ASK_OPENCODE_PORT` if one answers, and starts one if the port is free.
It is detached and outlives the caller, holding a few hundred MB, so collect it when you are done:

```
ask_opencode.py stop              # refuses while any turn is in flight
ask_opencode.py stop --force      # cancels those turns first
ask_opencode.py stop --if-idle 3600
```

It cannot see clients other than its own conversations, so "idle" means idle as far as this tool
knows.

## License

MIT — see `LICENSE`.

## Tests

```console
$ cd tests && python3 -m unittest test_ask_opencode
```

218 tests, standard library only. A fake server stands in for the real one, so the suite makes no
model calls and costs nothing.

The design notes that matter — how opencode resolves a permission ruleset, what a turn looks like on
the wire, and which behaviours were measured rather than assumed — live in the module docstrings of
`policy.py` and `ask_opencode.py`. Read those before changing either.
