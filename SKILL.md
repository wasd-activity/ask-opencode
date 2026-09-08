---
name: ask-opencode
description: Drive a multi-turn conversation with the opencode coding agent through this skill's ask_opencode.py, one turn per tool call, with every tool call opencode wants to make coming back to you for approval. Reach for this whenever the user names opencode as the agent to consult - "ask opencode", "have opencode review this", "what does opencode think", "run it by opencode", "get opencode to look at this", "/ask-opencode" - and for every later turn of a conversation already under way: sending the next message, approving or rejecting a permission gate, answering a question gate, resuming a turn whose wait budget elapsed, cancelling one, and collecting the server afterwards. Use it for casual follow-ups too, like "approve that", "ask it why" or "what did it say", whenever an opencode conversation is open. Do not reach for this when the user has not named opencode - an unqualified request for a second opinion, another model's view, or a cross-check belongs to whatever the session would otherwise do.
---

# ask-opencode

The driver is `ask_opencode.py`, in this skill's own directory. `<skill path>` below stands for wherever it was installed. It talks to an `opencode serve` process over HTTP, starting one if none is listening, and does all the waiting itself.

Four properties are the whole point, and each is easy to destroy by accident:

1. **opencode's text reaches the user unparaphrased.** The reply lands in a `.md` file. Read that file yourself. Never hand it to a subagent to summarize — a relayed summary is the one thing this tool exists to prevent.
2. **You never poll.** The script blocks until the turn lands — it keeps a light watch of its own on the server, so you do not have to — and never loop on a status check or re-issue a turn already in flight.
3. **One call per turn.** The script does the waiting, so a turn never needs more than a single command from you.
4. **You approve every tool call opencode makes**, apart from the pre-approved reads below. opencode stops mid-turn and waits; the turn ends at the gate and hands you the exact command.

## Running a turn

`ask_opencode.py` is synchronous: it blocks until the turn lands, then prints one line. Nothing is lost if your call ends first — the **server** owns the turn, not your process, so a call that is killed, backgrounded away or timed out leaves opencode working, and `wait` reattaches to the same turn.

Pick the highest rung you actually have:

| You have | Use | `--wait` |
|---|---|---|
| `Monitor` | `Monitor(command: "...", timeout_ms: 3600000)` | leave it (3500) |
| background execution only | `Bash` with `run_in_background`, then read the `.md` when the notification arrives | leave it (3500) |
| neither | plain blocking `Bash` | just under the call's own timeout — `--wait 540` under 600 s |

Blocking is the last rung, not the default: it is the only one that makes you sit still through a turn that may run for an hour. Two things the ladder does not excuse you from:

- **Bound `--wait` below your own timeout.** That is what turns a call your harness would kill into an orderly exit 10 with the continuation command printed. Unbounded, the tool call dies mid-turn and you have to work out where you were.
- **A backgrounded turn is only free if you come back for it.** In testing, subagents that backgrounded a turn were observed never to resume — the turn still completes and writes its answer, with nobody left to read it. Treat the notification as an obligation.

**Turn 1:**

```
python3 <skill path>/ask_opencode.py start --repo <repo> [--write] <<'TASK'
<task text>
TASK
```

End the task text with this line, so opencode reaches for the tools that never need approval before it reaches for a shell:

> Use your own `read`, `grep`, `glob` and `list` tools for anything they can do. Fall back to `bash` only for what they cannot express, and say why when you do.

One line comes back:

```
[ask-opencode] <conv-id> · turn 1 · <status> · <duration> · <tokens> · <provider/model:variant> · <path to .md>
```

**Read the `.md` at that path.** The summary line is routing information, not the answer; acting on it alone means reporting a verdict you have not read.

Tell the user the conv-id early, so they can inspect or continue the conversation from a terminal themselves.

**Every later turn** carries the conv-id from that line:

```
python3 <skill path>/ask_opencode.py send <conv-id> <<'MSG'
<your next message>
MSG
```

Each gate is its own turn, so the turn counter climbs faster than the messages you send.

## Choosing the flags

Settle these from the conversation. If the user has not said what opencode should work on, ask — a vague task wastes a whole conversation.

- **`--repo`** — the only thing that sets opencode's working directory. The server's own cwd never reaches the model. Pass it explicitly; it defaults to `.`, which is rarely what you want.
- **`--write`** — makes `edit` and `write` approval requests instead of denials. Default to read-only for review, diagnosis and research.
- **`--read-root PATH`** — repeatable. Needed whenever the task is about a repository that is not `--repo`: a path outside every root is **denied**, not gated, so without this flag opencode cannot read that repository at all — not with the native `read` tool and not with a shell — and you never see a request you could approve. A root covers everything under it. A path containing `*`, `?` or `[` is refused, because a root is written into the permission patterns verbatim and such a path would also match its siblings.

## Choosing the model

Nothing here picks a model. Without `--model`, opencode resolves it itself, and the last source it consults is whatever was last used in opencode's **TUI** — so the default drifts as the user works. It is never silent: the answering model is on every summary line and in every `.md` header.

Learn the legal values before pinning one:

```
python3 <skill path>/ask_opencode.py models --repo <repo>
```

That lists every `provider/model` this server can reach with the variants each offers, and marks the one that answers today. Then:

- `--model provider/model` pins the model for the conversation.
- `--variant <name>` pins the reasoning effort, sent with every prompt. Only the variants the model actually lists are accepted — a wrong one is exit 4 before the conversation opens, because the server would not fail on it, it would just quietly answer at some other effort.

Leave both alone unless the user asked for a specific model or the task needs a specific effort. Check the summary line on turn 1 either way; if the answer's quality is surprising, the model is the first thing to look at.

`start` refuses a model this server does not have, with exit 4, before spending a turn on it. If that happens, do not retry with the same settings: report it, and offer `--model` from the `models` listing or a fix to the opencode config — the server caches config per directory, so it needs a restart after an edit.

## Statuses

The summary line's third field says what to do next.

| Status | What it means | What to do |
|---|---|---|
| `converged` | opencode considers the question answered | Read the `.md`, judge whether it answers what the user asked, and push back with another `send` if it does not |
| `needs_input` | It needs a decision from you | Read the open questions and answer them with `send` |
| `blocked` | It cannot proceed | Read the `.md` and tell the user what it is stuck on |
| `unknown` | The structured envelope did not parse, so the file holds raw prose | Treat it as prose; nothing is wrong. A status the schema does not name lands here too, so a model cannot label its own answer `failed` |
| `needs_permission` | It wants to run a tool call and is waiting | See below |
| `needs_answer` | It is asking you a question and is suspended | See below |
| `failed` / `cancelled` | The `.md` carries the reason | Report it; do not silently retry |
| `running` | Your wait budget elapsed and opencode is still working | Nothing was lost. Continue with `wait <conv-id>` as another turn |

## `needs_permission`

opencode wants to run a tool call. The `.md` holds the exact command, the patterns it matched, and what the model was doing when it asked.

```
python3 <skill path>/ask_opencode.py approve <conv-id>
python3 <skill path>/ask_opencode.py reject <conv-id> --reason "why not"
```

Either answers the gate and keeps waiting, so it returns the next summary line — which may well be another `needs_permission`. **One turn asks once per tool call**, and a turn that works through a handful of commands asks a handful of times, so treat repeated gates as normal, not as a loop.

`--reason` is recorded in the resumed turn's record and rendered in its `.md`, for whoever reads the conversation later. **opencode never sees it** — if the model needs to know why, say so in your next `send`.

`approve` always answers `once`. There is deliberately no way to say "always": a persistent grant would stop opencode asking, which removes the checkpoint this tool exists to provide.

### Judging a permission request

You are the approval step; nothing else asks the user. Approve when the command plainly serves the task you handed opencode and its effect is contained. Reject when it reaches outside the repo, touches credentials, writes somewhere the task never mentioned, or you cannot tell what it would do — and say why in `--reason`.

The ruleset denies the commonest irreversible spellings outright, so `rm -rf x`, `sudo x`, `git push`, `git reset --hard` and `git clean -f` never reach you. **It is a literal-prefix blacklist and it does not close the class**: `git -C . push`, `/usr/bin/git push` and `rm -fr x` all miss those patterns and arrive as ordinary approval requests. So read every command you are asked about rather than trusting that the dangerous ones were filtered. If you find yourself wanting to approve something in that class, stop and ask the user.

A pipe into a shell also reaches you: opencode splits on `|`, so `curl … | sh` arrives as a gate on the `sh` half rather than a deny. Reject it.

## `needs_answer`

opencode is asking you a question and is suspended until it is answered. The `.md` lists each question with its options. Reply with **one label per question, in order**:

```
python3 <skill path>/ask_opencode.py answer <conv-id> <label> [<label> ...]
```

A label the question does not offer, or the wrong number of labels, is refused with exit 4 that names the valid options — nothing is sent.

## Reads that never reach you

`preapproved.json` names a few read-only commands — `ls`, `wc`, `cat`, `head`, `grep`, plus `echo` and `<tool> --version` — and the roots they may be aimed at. That layer also makes those roots readable at all, since a session rooted elsewhere cannot otherwise read a file under them by any means. Each turn's `.md` header names exactly what was pre-approved for that conversation, and `--read-root` extends the same layer to another directory.

**`rg` and `git` are not on that list, and were deliberately removed from it.** In any conversation you start now, both reach you as ordinary approval requests, including plain reads like `rg pattern file` and `git diff`. A conversation started *before* they were withdrawn keeps the ruleset the server fixed when it began, and its turn header says so — read the header rather than assuming this paragraph applies to a conversation you are resuming.

The reason is worth carrying into your judgement: their argument lists can execute a program of their own — `rg --pre CMD`, and git's configured external-diff and textconv helpers — and no pattern can filter that reliably, because shell quoting spells the same argument in unboundedly many ways (`rg --p''re CMD` was measured slipping past a guard written for `--pre`). So **read the flags**; a familiar command name does not mean a read.

Two things are denied rather than gated, and a rejection you did not issue is the ruleset working: redirecting a pre-approved command's output (`cat … > file`), and `$(…)` substitution inside one.

`--no-preapproved` on `start` drops that layer for one conversation, so those reads come back as approval requests too. It does not change the rest of the ruleset: the native `read`, `grep`, `glob` and `list` tools stay allowed, and edits and out-of-root paths stay denied.

## Inspecting and abandoning a conversation

```
python3 <skill path>/ask_opencode.py list                     # every conversation, with its last status
python3 <skill path>/ask_opencode.py show <conv-id> [--turn N]  # reprint a stored turn
python3 <skill path>/ask_opencode.py cancel <conv-id>          # abort the turn in flight
```

`cancel` is for a turn that is running and no longer wanted — the user changed their mind, or the task was wrong. It aborts the turn and closes it out; the conversation itself survives, so you can still `send` to it afterwards.

## The server outlives you

The first turn starts a detached `opencode serve` that nothing ever stops — it survives this session and holds a few hundred MB. When the user is done with opencode, collect it:

```
python3 <skill path>/ask_opencode.py stop
```

It refuses while any conversation holds a turn open, and it is shared: other sessions may be using the same server, so stop it when the user asks or when the work is finished, not at the end of every conversation.

## Long conversations get compacted

A long session gets summarised mid-turn by the server, which then carries on. Where that happens is opencode's business and follows the model's declared context window — with one 500k-context model it was observed at roughly 468k tokens, which is an observation rather than a setting, so do not plan around the number. What matters to you is the effect: the turn still converges, and its `.md` header says the session was compacted — read that as "the model answered from a summary of everything before this turn". opencode also runs a short turn of its own right after compacting, so token counts move without you asking for anything.

## Reference

`<skill path>/README.md` has the full CLI surface, exit codes, environment variables, the state layout, how the permission ruleset is built, and the measured notes on opencode's HTTP API. Read it when you need a flag this file does not cover, or before changing the driver.
