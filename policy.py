"""Permission rulesets handed to opencode when a conversation's session is created.

opencode resolves a ruleset **last-match-wins**: the last rule whose permission
and pattern match the call decides the action. That is the opposite of the usual
first-match convention, and getting it backwards is silent — the rule is simply
never consulted. Measured on 1.18.21 with one pair of rules, order swapped:

    [{bash, "echo DENIED*", deny}, {bash, "*", ask}]  -> ask won, command ran
    [{bash, "*", ask}, {bash, "echo DENIED*", deny}]  -> deny won, call blocked

So the catch-all goes first and the specific rules go last. `build()` is the only
place that assembles a ruleset, and `test_policy_order` pins the ordering.

A session created without a ruleset inherits opencode's implicit baseline,
`{permission: "*", pattern: "*", action: "allow"}` — everything allowed. Every
session this tool creates therefore carries an explicit ruleset.

`preapproved.json` layers a handful of read-only shell commands on top, so the
caller is not asked to approve the same `cat` of a task note turn after turn. It
holds three lists: `bash`, path-reading commands anchored to the named roots;
`bash_anywhere`, commands that report on whatever directory the session is in
and so have no path to anchor to; and `version_probes`, single executables asked
only for their `--version`. Four measured
facts shape how they are emitted:

  - `external_directory` is checked before the bash ruleset and before the
    native read tools. With the session directory outside a root, no bash rule
    is ever reached for a path under it — so the layer starts with an
    `external_directory` allow, placed after the catch-all deny.
  - `*` matches across spaces and quotes: `cat *<root>/*` covers
    `cat -n "<root>/f"` as well as `cat <root>/f`. Quoted variants are not needed
    *after* a leading `*`; they are needed where the pattern pins a token, which
    is why the `-C` forms below are emitted bare and double-quoted.
  - opencode splits a command on `&&` and matches each segment separately, so a
    trailing `*` cannot swallow a chained command. It does swallow a redirect:
    under a `cat *<root>/*` allow, `cat "<root>/f" > <repo>/x` ran and wrote the
    file, which is a write the `edit`/`write` deny never sees. `unsafe_guards()`
    denies that vector for exactly the commands the file pre-approved.
  - `..` is normalised before the external check, but symlinks are not resolved,
    so each root is emitted both as written and as its real path.

Roots are not fixed by the file: `build()` takes `extra_roots`, which `start`
fills from `--read-root`. One list feeds both layers — the anchored bash reads
and the `external_directory` allow — so a root that is readable by the native
tools is readable by the pre-approved commands and vice versa, rather than the
two drifting apart. A root is written into patterns verbatim, so `start` refuses
one that is itself a pattern; see `unsafe_root`.

No command head reaches the `bash` lists without review. `REVIEWED_HEADS` maps
each permitted head to the flags that turn its read into a write or into a
program of its own, and `load_preapproved()` refuses an entry whose head is not
in it. The table is what `unsafe_guards()` reads, so a head cannot be admitted
without its guards being emitted; and the two commands that cannot be guarded at
all — `sed`, `perl` — are absent, with the reason recorded beside the table.

Two shape rules keep the allow patterns from becoming write channels:

  - **Every allow pattern begins with a literal command token.** A pattern that
    opens with `*` means "the command *ends* with this", which any prefix
    satisfies — including one that runs something else first. So `--version`
    probes are emitted as exact literal patterns rather than as `*/<tool>
    --version`, and nothing here is anchored on its tail.
  - **A flag guard is not a boundary, so a head that needs one is withdrawn
    rather than guarded.** Shell quoting spells one argument in unboundedly many
    ways: `rg --p''re CMD x <root>/f` resolved to allow where `rg --pre …`
    resolved to deny, and the shell hands ripgrep `--pre` either way. See
    `WITHDRAWN_HEADS`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

# Tool calls that cannot be undone by the caller once they run. These are the
# only gate that does not depend on the calling session's judgement, so they are
# denied outright and never surface as an approval request.
#
# This is a literal-prefix blacklist and it does not close the class. `git -C .
# push`, `/usr/bin/git push` and `rm -fr x` all miss these patterns and arrive as
# ordinary approval requests instead — measured against this ruleset. That is the
# design: a blacklist cannot be completed, so it removes the commonest spellings
# from the approval path and the caller still reads every command it is asked
# about. Nothing downstream may claim this class cannot reach the caller.
#
# The two pipe patterns are inert: opencode splits a command on `|` before
# matching, so `curl x | sh` is tested as `curl x` and `sh` and neither reaches
# `* | sh`. Measured — `git log --oneline | sh` arrived as a gate on the `sh`
# segment. It still surfaces rather than running, but as an approval request,
# not a deny. Denying the segment itself (an exact `sh` / `bash`, which is the
# stdin-reading form) is the fix, and it would also stop `bash script.sh` from
# being approvable.
IRREVERSIBLE_BASH = (
    "rm -rf *",
    "sudo *",
    "git push*",
    "git reset --hard*",
    "git clean -*",
    "* | sh",
    "* | bash",
)

READ_ONLY_TOOLS = ("read", "grep", "glob", "list")

# Command heads reviewed for what a pre-approved read of theirs can be turned
# into, mapped to the flags that do it. The table has two jobs on purpose: it
# supplies the flag denies at the bottom of a ruleset, and `load_preapproved`
# refuses a `bash` or `bash_anywhere` entry whose head is not a key here. A
# command nobody has reviewed therefore cannot ship as an unguarded allow, which
# is what a plain list of flags could not promise.
#
# Heads that were removed rather than guarded are listed in WITHDRAWN_HEADS
# below, with the measurement that removed each one.
REVIEWED_HEADS: dict[str, tuple[str, ...]] = {
    "cat": (),
    "echo": (),
    "grep": (),
    "head": (),
    "ls": (),
    "wc": (),
}

# Heads that were reviewed and then removed, with the reason, so that re-adding
# one takes more than an entry in the table above.
#
# `rg` and `git` were both pre-approved and both are gone, for the same reason:
# a textual flag guard is not a boundary. Shell quoting spells the same argument
# in unboundedly many ways — measured against the generated ruleset,
# `rg --p''re CMD x <root>/f` and `git diff --out""put=F` both resolved to allow
# while their unquoted spellings resolved to deny, and the shell hands the
# program `--pre` and `--output=F` either way. Short options bundle, too:
# `rg -iz` slips past a guard written for `-z`. And `git` needs no flag at all:
# `--ext-diff` and textconv run programs named by configuration this tool never
# sees. Both now reach the caller as ordinary approval requests, which is a
# decision a person makes rather than a pattern nobody can complete.
WITHDRAWN_HEADS = ("rg", "git", "sed", "perl")

# Shell forms that run a second command inside the first. An allow pattern's
# *internal* `*` absorbs them the same way it absorbs a redirect — the argument
# sitting between the command word and the root is still matched by `*`, while
# the shell evaluates it before the read ever happens. Denied per command head
# rather than per full prefix: there is no pre-approved read for which a
# substituted argument is worth approving, and one rule per head keeps the
# ruleset from growing a copy for every root.
SUBSTITUTION_FORMS = ("$(", "`", "<(")

# A pre-approved command is a literal prefix; build() supplies the anchor.
# Anything that could turn the entry into a glob, a second command or a redirect
# is refused, so the file cannot widen its own grant.
FORBIDDEN_IN_ENTRY = ("*", "?", "[", "]", ">", "<", "|", ";", "&", "`", "$", "\n")

# A root is written into permission patterns verbatim, so a path that is itself
# a pattern is interpreted twice: once as a filesystem name and once as a glob.
# A real directory named `review*` emits `<parent>/review*/*`, which also matches
# `<parent>/review-secret/notes.txt`. Being a directory does not make the second
# reading safe, so a root carrying one of these is refused rather than escaped —
# escaping would have to match opencode's dialect exactly, which is not measured.
PATTERN_METACHARACTERS = ("*", "?", "[", "]")


def unsafe_root(root: Path) -> str | None:
    """The pattern character that makes this root unusable, or None.

    Both spellings are checked, because `root_forms` emits the path as written
    and as its real path, and only one of them may carry the character.
    """
    for form in (root, root.resolve()):
        for char in PATTERN_METACHARACTERS:
            if char in str(form):
                return char
    return None


DEFAULT_PREAPPROVED_PATH = Path(__file__).resolve().parent / "preapproved.json"


class PreapprovedError(ValueError):
    """The pre-approval file exists but cannot be trusted."""


def root_forms(roots: tuple[Path, ...]) -> tuple[str, ...]:
    """Each root as written and as its real path — the permission check does not
    resolve symlinks, so a session may name a root either way."""
    forms: list[str] = []
    for root in roots:
        for form in (root, root.resolve()):
            text = str(form).rstrip("/")
            if text and text not in forms:
                forms.append(text)
    return tuple(forms)


def covered_by(roots: tuple[Path, ...], repo: Path | None) -> bool:
    """True when the session directory is itself inside one of the roots."""
    if repo is None:
        return False
    target = repo.resolve()
    for root in roots:
        resolved = root.resolve()
        if target == resolved or resolved in target.parents:
            return True
    return False


@dataclass(frozen=True)
class Preapproved:
    """What `preapproved.json` grants.

    Three kinds of entry, because they are scoped differently. `commands` read a
    path, so they are anchored to a root. `anywhere` read the state of whatever
    directory the session is in — `echo` names no path to anchor — so they carry
    no anchor and are bounded by the session directory itself. `version_probes`
    are single executables asked only whether they exist, and are emitted as
    exact literal patterns.
    """

    roots: tuple[Path, ...]
    commands: tuple[str, ...]
    anywhere: tuple[str, ...] = ()
    version_probes: tuple[str, ...] = ()

    def root_patterns(self) -> tuple[str, ...]:
        return root_forms(self.roots)

    def covers(self, repo: Path | None) -> bool:
        return covered_by(self.roots, repo)


def _string_list(source: Path, loaded: dict[str, object], key: str) -> tuple[str, ...]:
    raw = loaded.get(key)
    if not isinstance(raw, list):
        raise PreapprovedError(f"{source}: '{key}' must be a list of strings")
    items: list[str] = []
    for entry in cast("list[object]", raw):
        if not isinstance(entry, str) or not entry.strip():
            raise PreapprovedError(f"{source}: '{key}' must hold non-empty strings")
        text = entry.strip()
        bad = [c for c in FORBIDDEN_IN_ENTRY if c in text]
        if bad:
            raise PreapprovedError(
                f"{source}: '{key}' entry {text!r} may not contain {bad[0]!r} — "
                "entries are literal, and build() supplies the anchor"
            )
        if text not in items:
            items.append(text)
    return tuple(items)


def load_preapproved(path: Path | None = None) -> Preapproved | None:
    """Read the pre-approval file.

    A missing file means no pre-approval at all. A malformed one is an error:
    falling back to a ruleset the caller did not ask for — in either direction —
    would be exactly the kind of silent change this module exists to prevent.
    """
    target = (
        path
        or Path(os.environ.get("ASK_OPENCODE_PREAPPROVED", DEFAULT_PREAPPROVED_PATH)).expanduser()
    )
    if not target.exists():
        return None
    try:
        decoded = json.loads(target.read_text())
    except (OSError, ValueError) as exc:
        raise PreapprovedError(f"{target}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise PreapprovedError(f"{target}: the top level must be an object")
    # A JSON object's keys are strings by construction, which is what the cast asserts;
    # `isinstance` alone narrows the value to a mapping whose contents stay untyped.
    loaded = cast("dict[str, object]", decoded)
    known = {"roots", "bash", "bash_anywhere", "version_probes"}
    unknown = sorted(set(loaded) - known)
    if unknown:
        raise PreapprovedError(f"{target}: unknown key(s) {unknown}")
    roots = _string_list(target, loaded, "roots") if "roots" in loaded else ()
    commands = _string_list(target, loaded, "bash") if "bash" in loaded else ()
    anywhere = _string_list(target, loaded, "bash_anywhere") if "bash_anywhere" in loaded else ()
    probes = _string_list(target, loaded, "version_probes") if "version_probes" in loaded else ()
    for key, entries in (("bash", commands), ("bash_anywhere", anywhere)):
        for entry in entries:
            head = entry.split()[0]
            if head in WITHDRAWN_HEADS:
                raise PreapprovedError(
                    f"{target}: '{key}' entry {entry!r} names {head!r}, which was "
                    "reviewed and withdrawn — see policy.WITHDRAWN_HEADS for what "
                    "was measured. It reaches the caller as an approval request "
                    "instead."
                )
            if head not in REVIEWED_HEADS:
                raise PreapprovedError(
                    f"{target}: '{key}' entry {entry!r} has an unreviewed command "
                    f"head {head!r}. A head has to be reviewed for what a "
                    "pre-approved read of it can be turned into — a write, or a "
                    "program of its own — and added to policy.REVIEWED_HEADS with "
                    "the flags that do it."
                )
    for entry in probes:
        # One executable and nothing else. `sh -c id` is a literal string with no
        # forbidden character in it, and appending `--version` to it does not
        # make it a version probe — it runs `id`. The field promises a single
        # executable to which build() appends the flag, so enforce that.
        if len(entry.split()) != 1:
            raise PreapprovedError(
                f"{target}: 'version_probes' entry {entry!r} must be a single "
                "executable name or path — build() appends the `--version`"
            )
    if commands and not roots:
        raise PreapprovedError(
            f"{target}: 'bash' needs at least one root to anchor its patterns to"
        )
    expanded = tuple(Path(root).expanduser() for root in roots)
    for root in expanded:
        if not root.is_absolute():
            raise PreapprovedError(f"{target}: root {str(root)!r} is not absolute")
        if str(root.resolve()) == "/":
            # Resolved, not as written: a path ending in `..` that normalises
            # to `/`, and a symlink to `/`, are the same request spelled
            # differently, and `root_forms` drops the resolved `/` either way —
            # so it would grant nothing while every turn header advertised it.
            raise PreapprovedError(
                f"{target}: root {str(root)!r} resolves to '/', which is not "
                "supported — it would grant nothing while every turn header "
                "advertised it"
            )
        # After expansion *and* resolution. Rejecting the JSON string is not
        # enough: a root written as an innocuous name can resolve through a
        # symlink onto a directory whose own name is a pattern, and it is the
        # resolved spelling that `root_forms` emits into the ruleset.
        bad = unsafe_root(root)
        if bad:
            raise PreapprovedError(
                f"{target}: root {str(root)!r} resolves to a path containing "
                f"{bad!r}, a pattern character in the permission ruleset: it "
                "would also match sibling paths"
            )
    if not (expanded or anywhere or probes):
        return None
    return Preapproved(
        roots=expanded,
        commands=commands,
        anywhere=anywhere,
        version_probes=probes,
    )


def unsafe_guards(prefixes: tuple[str, ...]) -> tuple[str, ...]:
    """The write and exec vectors the pre-approved allows would otherwise open.

    Takes the literal prefixes that were actually emitted as allows, not the
    entries the file listed, so an allow shape cannot be added without its guard
    coming along.

    An allow pattern that ends in `*` also matches the same command with a
    redirect appended — measured: under a `cat *<root>/*` allow,
    `cat "<root>/f" > <repo>/x` executed and wrote the file. A substituted
    argument rides in on the pattern's internal `*`. Both are denied, last, and
    only for what was allowed.

    A head may also declare flags in `REVIEWED_HEADS` that change what it does,
    and those are denied the same way. Every head currently on that table
    declares none: the heads that needed flag guards are withdrawn instead,
    because quoting spells one argument in unboundedly many ways.
    """
    patterns: list[str] = []
    heads: list[str] = []
    for prefix in prefixes:
        patterns.append(f"{prefix} *>*")
        head = prefix.split()[0]
        if head not in heads:
            heads.append(head)
    for head in heads:
        patterns.extend(f"{head} *{form}*" for form in SUBSTITUTION_FORMS)
        for flag in REVIEWED_HEADS.get(head, ()):
            if flag.startswith("--"):
                patterns.append(f"{head} *{flag}*")
            else:
                # `rg -z …` and `rg pattern -z …`. The leading space is what
                # keeps a task directory called `fix-import-bug` from matching;
                # a long flag is distinctive enough not to need it.
                patterns += [f"{head} {flag}*", f"{head} * {flag}*"]
    return tuple(dict.fromkeys(patterns))


def rule(permission: str, pattern: str, action: str) -> dict[str, str]:
    return {"permission": permission, "pattern": pattern, "action": action}


def unanchored(command: str) -> list[dict[str, str]]:
    """A command allowed with no path anchor: bare, and with arguments."""
    return [rule("bash", command, "allow"), rule("bash", f"{command} *", "allow")]


def version_patterns(entry: str) -> list[str]:
    """The exact patterns a `--version` probe is allowed under.

    Literal, with no wildcard anywhere: an extra argument, a redirect or a
    substitution all make the command longer than the pattern and so gate. A `~`
    entry is emitted as written and expanded, because the pattern is matched
    against the command text the model wrote, before the shell expands it.
    """
    forms = [entry]
    expanded = str(Path(entry).expanduser())
    if expanded != entry:
        forms.append(expanded)
    return [f"{form} --version" for form in forms]


def build(
    write: bool,
    preapproved: Preapproved | None = None,
    repo: Path | None = None,
    extra_roots: tuple[Path, ...] = (),
) -> list[dict[str, str]]:
    """The ruleset for a conversation. Order is load-bearing; see module docstring.

    read-only (default): the model may inspect the repo freely and propose shell
    commands, which the caller approves one at a time. It cannot edit or write.

    --write: edit and write become approval requests too, so the caller decides
    per file operation rather than granting the whole conversation up front.

    With a `preapproved` file the named roots become reachable and its commands
    stop surfacing when aimed at one. Passing none reproduces the ruleset this
    tool carried before the file existed.

    `extra_roots` are the conversation's own `--read-root` paths. They join the
    file's roots for every layer that takes a root. They also work on their own:
    with no pre-approval there are no pre-approved commands, but the
    `external_directory` allow still lands, because reaching a directory at all
    and not being asked about it are different grants.
    """
    rules = [rule("*", "*", "ask")]

    rules += [rule(tool, "*", "allow") for tool in READ_ONLY_TOOLS]

    # A separate gate from `bash`: it fires for paths outside the session
    # directory before the bash rule is ever consulted, so a bash ruleset alone
    # does not cover it.
    rules.append(rule("external_directory", "*", "deny"))

    all_roots = (preapproved.roots if preapproved else ()) + tuple(extra_roots)
    roots = root_forms(all_roots)
    # After that deny, so last-match-wins re-opens the roots. This is what makes
    # a root reachable at all — for the native read tools as much as for bash.
    # The root itself as well as everything under it: a `list` of the root and an
    # `ls <root>` with no trailing slash both name the root exactly.
    for root in roots:
        rules.append(rule("external_directory", root, "allow"))
        rules.append(rule("external_directory", f"{root}/*", "allow"))

    mutating = "ask" if write else "deny"
    rules.append(rule("edit", "*", mutating))
    rules.append(rule("write", "*", mutating))

    commands = preapproved.commands if preapproved else ()
    anywhere = preapproved.anywhere if preapproved else ()
    probes = preapproved.version_probes if preapproved else ()
    inside_root = covered_by(all_roots, repo)
    # Every literal prefix allowed below, so the guards at the bottom are derived
    # from what was granted rather than from what the file asked for.
    prefixes: list[str] = []

    for command in commands:
        for root in roots:
            # The root itself and everything under it. `ls *<root>` ends on the
            # root, so a sibling directory whose name merely starts with it does
            # not match; `ls *<root>/*` needs the separator and cannot either.
            rules.append(rule("bash", f"{command} *{root}", "allow"))
            rules.append(rule("bash", f"{command} *{root}/*", "allow"))
        if inside_root:
            # The session directory is itself inside a root, so a relative path
            # is already in scope even though no pattern can tell. What such a
            # command can reach is still bounded by external_directory.
            rules += unanchored(command)
        prefixes.append(command)
    for command in anywhere:
        # No path to anchor to: these report on the directory the session is
        # already working in. Both forms, so a bare `echo` matches as well as
        # `echo hello`.
        rules += unanchored(command)
        prefixes.append(command)
    for entry in probes:
        # Exact patterns, so no trailing `*` and nothing to guard.
        rules += [rule("bash", pattern, "allow") for pattern in version_patterns(entry)]

    # Last, so nothing above can re-open them.
    rules += [rule("bash", pattern, "deny") for pattern in IRREVERSIBLE_BASH]
    guarded = tuple(dict.fromkeys(prefixes))
    rules += [rule("bash", pattern, "deny") for pattern in unsafe_guards(guarded)]
    return rules
