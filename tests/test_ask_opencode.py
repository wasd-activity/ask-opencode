"""Test suite for ask_opencode.py. Stdlib unittest only, no real opencode calls.

Run: cd tests && python3 -m unittest test_ask_opencode -v
"""
import argparse
import contextlib
import fnmatch
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

PKG_DIR = Path(__file__).resolve().parents[1]
ASK = PKG_DIR / "ask_opencode.py"
FAKE = Path(__file__).resolve().parent / "fake_opencode.py"
SCHEMA = PKG_DIR / "schemas" / "turn.schema.json"
# The repository ships an example; `preapproved.json` is a local file and is not
# tracked, so nothing here may assume it exists.
EXAMPLE_PREAPPROVED = PKG_DIR / "preapproved.example.json"

sys.path.insert(0, str(PKG_DIR))
import ask_opencode  # noqa: E402
import policy  # noqa: E402


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class AskOpencodeCase(unittest.TestCase):
    """Each test gets its own fake server, state dir and request log."""

    script: dict = {"turns": []}
    server_env: dict = {}

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ask-opencode-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        self.log = self.tmp / "requests.log"
        self.script_path = self.tmp / "script.json"
        self.script_path.write_text(json.dumps(self.script))
        self.port = free_port()
        self.server = subprocess.Popen(
            [sys.executable, str(FAKE)],
            env={**os.environ,
                 "FAKE_OPENCODE_PORT": str(self.port),
                 "FAKE_OPENCODE_SCRIPT": str(self.script_path),
                 "FAKE_OPENCODE_LOG": str(self.log),
                 **self.server_env},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.addCleanup(self._stop_server)
        self.env = {
            **os.environ,
            "ASK_OPENCODE_URL": f"http://127.0.0.1:{self.port}",
            "ASK_OPENCODE_STATE_DIR": str(self.tmp / "state"),
            "ASK_OPENCODE_POLL_S": "0.05",
            "ASK_OPENCODE_GATE_CLEAR_S": "5",
            # Neither the machine's pre-approval file nor its opencode state may
            # leak into a test: both would change the ruleset under assertion.
            "ASK_OPENCODE_PREAPPROVED": str(self.tmp / "absent-preapproved.json"),
            "ASK_OPENCODE_MODEL_STATE": str(self.tmp / "absent-model-state.json"),
        }
        self._await_server()

    def _stop_server(self):
        self.server.terminate()
        try:
            self.server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.server.kill()

    def _await_server(self):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.05)
        self.fail("fake opencode server did not start")

    def run_ask(self, *args, stdin="", env=None, timeout=60):
        return subprocess.run(
            [sys.executable, str(ASK), *args],
            input=stdin, capture_output=True, text=True, timeout=timeout,
            env={**self.env, **(env or {})},
        )

    def start(self, task="do the thing", extra=(), env=None):
        return self.run_ask("start", "--repo", str(self.repo), *extra, stdin=task, env=env)

    def conv_of(self, proc):
        for line in proc.stdout.splitlines():
            if line.startswith("[ask-opencode] "):
                return line.split(" · ")[0].removeprefix("[ask-opencode] ").strip()
        self.fail(f"no summary line in output:\n{proc.stdout}\n{proc.stderr}")

    def status_of(self, proc):
        for line in proc.stdout.splitlines():
            if line.startswith("[ask-opencode] "):
                return line.split(" · ")[2].strip()
        self.fail(f"no summary line in output:\n{proc.stdout}\n{proc.stderr}")

    def requests(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines() if line.strip()]

    def session_create_body(self):
        for entry in self.requests():
            if entry["method"] == "POST" and entry["path"] == "/session":
                return entry["body"]
        self.fail("no session create request was made")


PERMISSION_GATE = {
    "_kind": "permission",
    "id": "per_bash_0",
    "permission": "bash",
    "patterns": ["date +%Y"],
    "metadata": {"command": "date +%Y"},
    "always": ["date *"],
    "tool": {"messageID": "msg_x", "callID": "bash_0"},
}
EXTERNAL_GATE = {
    "_kind": "permission",
    "id": "per_ext_0",
    "permission": "external_directory",
    "patterns": ["/tmp/*"],
    "metadata": {"command": "cat /tmp/x", "directories": ["/tmp"]},
    "always": ["/tmp/*"],
}
QUESTION_GATE = {
    "_kind": "question",
    "id": "que_0",
    "questions": [{"message": "Which database?",
                   "options": [{"label": "postgres"}, {"label": "sqlite"}]}],
}
TWO_QUESTION_GATE = {
    "_kind": "question",
    "id": "que_two",
    "questions": [
        {"message": "Which database?",
         "options": [{"label": "postgres"}, {"label": "sqlite"}]},
        {"message": "Which runtime?",
         "options": [{"label": "node"}, {"label": "bun"}]},
    ],
}


class TestPolicyOrder(unittest.TestCase):
    """opencode resolves rules last-match-wins, so ordering is the whole contract."""

    def test_catch_all_is_first(self):
        rules = policy.build(write=False)
        self.assertEqual(rules[0], {"permission": "*", "pattern": "*", "action": "ask"})

    def test_irreversible_bash_rules_come_after_every_other_rule(self):
        rules = policy.build(write=False)
        last_deny = max(i for i, r in enumerate(rules)
                        if r["permission"] == "bash" and r["action"] == "deny")
        others = [i for i, r in enumerate(rules)
                  if not (r["permission"] == "bash" and r["action"] == "deny")]
        self.assertTrue(all(i < last_deny for i in others),
                        "a non-deny rule follows the blacklist and would override it")

    def test_every_irreversible_pattern_is_present_and_denied(self):
        rules = policy.build(write=False)
        denied = {r["pattern"] for r in rules
                  if r["permission"] == "bash" and r["action"] == "deny"}
        self.assertEqual(denied, set(policy.IRREVERSIBLE_BASH))

    def test_blacklist_is_not_reopened_by_a_later_wildcard(self):
        """Sensitivity check: this is exactly the mistake the ordering prevents."""
        rules = policy.build(write=True)
        first_deny = min(i for i, r in enumerate(rules)
                         if r["permission"] == "bash" and r["action"] == "deny")
        later_bash_wildcards = [r for r in rules[first_deny:]
                                if r["pattern"] == "*" and r["permission"] in ("*", "bash")]
        self.assertEqual(later_bash_wildcards, [])

    def test_write_flag_only_moves_edit_and_write(self):
        read_only = policy.build(write=False)
        writable = policy.build(write=True)
        self.assertEqual(len(read_only), len(writable))
        differing = [(a, b) for a, b in zip(read_only, writable) if a != b]
        self.assertEqual({a["permission"] for a, _ in differing}, {"edit", "write"})
        self.assertEqual({b["action"] for _, b in differing}, {"ask"})
        self.assertEqual({a["action"] for a, _ in differing}, {"deny"})

    def test_external_directory_has_its_own_rule(self):
        """It is a separate gate from bash and fires before it."""
        rules = policy.build(write=True)
        entries = [r for r in rules if r["permission"] == "external_directory"]
        self.assertEqual(entries, [{"permission": "external_directory",
                                    "pattern": "*", "action": "deny"}])


class TestPreapprovedLoader(unittest.TestCase):
    """The file is the only way to widen a session's ruleset, so anything it
    cannot express must be refused rather than half-applied."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ask-opencode-pre-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.root = self.tmp / "tasks"
        self.root.mkdir()

    def write(self, payload) -> Path:
        path = self.tmp / "preapproved.json"
        path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
        return path

    def load(self, payload):
        return policy.load_preapproved(self.write(payload))

    def test_a_missing_file_means_no_preapproval(self):
        self.assertIsNone(policy.load_preapproved(self.tmp / "absent.json"))

    def test_roots_and_commands_are_parsed(self):
        loaded = self.load({"roots": [str(self.root)], "bash": ["cat", "head"]})
        assert loaded is not None
        self.assertEqual(loaded.commands, ("cat", "head"))
        self.assertEqual(loaded.roots, (self.root,))

    def test_a_root_is_emitted_as_written_and_resolved(self):
        """The permission check normalises `..` but does not follow symlinks, so
        a session may name a root either way."""
        link = self.tmp / "link"
        link.symlink_to(self.root)
        loaded = self.load({"roots": [str(link)], "bash": ["cat"]})
        assert loaded is not None
        self.assertEqual(loaded.root_patterns(), (str(link), str(link.resolve())))

    def test_a_tilde_root_is_expanded(self):
        loaded = self.load({"roots": ["~/notes/tasks"], "bash": ["cat"]})
        assert loaded is not None
        self.assertEqual(loaded.roots, (Path.home() / "notes/tasks",))

    def test_malformed_json_is_an_error(self):
        with self.assertRaises(policy.PreapprovedError):
            self.load("{not json")

    def test_an_unknown_key_is_an_error(self):
        with self.assertRaises(policy.PreapprovedError):
            self.load({"roots": [str(self.root)], "bash": [], "allow": ["rm"]})

    def test_a_glob_in_a_command_is_refused(self):
        """`cat *` would grant cat of anything; the anchor is build()'s to add."""
        with self.assertRaises(policy.PreapprovedError):
            self.load({"roots": [str(self.root)], "bash": ["cat *"]})

    def test_a_redirect_or_chain_in_a_command_is_refused(self):
        for entry in ("cat > x", "cat; rm -rf /", "cat | sh", "cat `id`"):
            with self.assertRaises(policy.PreapprovedError):
                self.load({"roots": [str(self.root)], "bash": [entry]})

    def test_commands_without_a_root_are_refused(self):
        with self.assertRaises(policy.PreapprovedError):
            self.load({"roots": [], "bash": ["cat"]})

    def test_a_relative_root_is_refused(self):
        with self.assertRaises(policy.PreapprovedError):
            self.load({"roots": ["tasks"], "bash": ["cat"]})

    def test_path_free_commands_are_kept_apart_from_anchored_ones(self):
        loaded = self.load({"roots": [str(self.root)], "bash": ["cat"],
                            "bash_anywhere": ["echo"]})
        assert loaded is not None
        self.assertEqual(loaded.commands, ("cat",))
        self.assertEqual(loaded.anywhere, ("echo",))

    def test_path_free_commands_need_no_root(self):
        loaded = self.load({"bash_anywhere": ["echo"]})
        assert loaded is not None
        self.assertEqual(loaded.roots, ())
        self.assertEqual(loaded.anywhere, ("echo",))

    def test_a_path_free_entry_is_validated_the_same_way(self):
        with self.assertRaises(policy.PreapprovedError):
            self.load({"bash_anywhere": ["echo hello > out"]})

    def test_the_example_file_parses(self):
        loaded = policy.load_preapproved(EXAMPLE_PREAPPROVED)
        assert loaded is not None
        self.assertIn("cat", loaded.commands)
        self.assertIn("echo", loaded.anywhere)
        self.assertTrue(loaded.roots)

    def test_the_new_lists_are_validated_the_same_way(self):
        for payload in ({"version_probes": ["cargo; rm -rf /"]},
                        {"version_probes": ["cargo *"]},
                        {"bash_anywhere": ["echo > out"]}):
            with self.assertRaises(policy.PreapprovedError):
                self.load(payload)

    def test_a_version_probe_must_be_one_executable(self):
        """`sh -c id` holds no forbidden character, and appending `--version`
        to it does not make it a version probe — it runs `id`."""
        with self.assertRaises(policy.PreapprovedError) as caught:
            self.load({"version_probes": ["sh -c id"]})
        self.assertIn("single", str(caught.exception))
        # The control: an ordinary probe still loads.
        loaded = self.load({"version_probes": ["cargo"]})
        assert loaded is not None
        self.assertEqual(loaded.version_probes, ("cargo",))

    def test_a_withdrawn_head_is_named_as_withdrawn(self):
        for head in policy.WITHDRAWN_HEADS:
            with self.assertRaises(policy.PreapprovedError) as caught:
                self.load({"roots": [str(self.root)], "bash": [head]})
            self.assertIn("withdrawn", str(caught.exception))


class TestPreapprovedOrder(unittest.TestCase):
    """Where the pre-approved rules land decides whether they work at all, and
    whether the blacklist above them survives."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ask-opencode-order-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.root = self.tmp / "tasks"
        (self.root / "inside").mkdir(parents=True)
        self.pre = policy.Preapproved(roots=(self.root,), commands=("cat", "head"),
                                      anywhere=("echo",))

    def rules(self, repo=None):
        return policy.build(False, self.pre, repo)

    def index(self, rules, predicate, last=False):
        hits = [i for i, r in enumerate(rules) if predicate(r)]
        self.assertTrue(hits, "no rule matched")
        return hits[-1] if last else hits[0]

    def test_no_preapproval_reproduces_the_previous_ruleset(self):
        self.assertEqual(policy.build(False), policy.build(False, None, self.root))

    def test_bash_allows_sit_after_the_catch_all_and_before_every_deny(self):
        rules = self.rules()
        allow = self.index(rules, lambda r: r["permission"] == "bash" and r["action"] == "allow", last=True)
        first_deny = self.index(rules, lambda r: r["permission"] == "bash" and r["action"] == "deny")
        self.assertEqual(rules[0]["action"], "ask")
        self.assertLess(0, allow)
        self.assertLess(allow, first_deny)

    def test_the_external_allow_comes_after_the_external_deny(self):
        """Reversed, the deny would win and no bash rule would ever be reached."""
        rules = self.rules()
        deny = self.index(rules, lambda r: r["permission"] == "external_directory" and r["action"] == "deny")
        allow = self.index(rules, lambda r: r["permission"] == "external_directory" and r["action"] == "allow")
        self.assertLess(deny, allow)
        self.assertIn(f"{self.root}/*", [r["pattern"] for r in rules
                                         if r["permission"] == "external_directory"])

    def test_every_command_is_anchored_to_every_root(self):
        patterns = {r["pattern"] for r in self.rules()
                    if r["permission"] == "bash" and r["action"] == "allow"}
        for command in self.pre.commands:
            for root in self.pre.root_patterns():
                self.assertIn(f"{command} *{root}/*", patterns)

    def test_write_guards_are_last_and_cover_every_command(self):
        """An allow ending in `*` also matches the same command with a redirect
        appended — measured, and it wrote the file."""
        rules = self.rules()
        denied = [r["pattern"] for r in rules
                  if r["permission"] == "bash" and r["action"] == "deny"]
        for command in self.pre.commands:
            self.assertIn(f"{command} *>*", denied)
        for command in self.pre.commands + self.pre.anywhere:
            self.assertIn(f"{command} *$(*", denied)
        last_allow = self.index(rules, lambda r: r["action"] == "allow", last=True)
        first_guard = self.index(rules, lambda r: r["pattern"].endswith("*>*"))
        self.assertLess(last_allow, first_guard)

    def test_the_irreversible_blacklist_is_still_unreachable_from_above(self):
        rules = self.rules(self.root / "inside")
        first_deny = min(i for i, r in enumerate(rules)
                         if r["permission"] == "bash" and r["action"] == "deny")
        self.assertEqual([r for r in rules[first_deny:] if r["action"] == "allow"], [])

    def test_path_free_commands_are_allowed_bare_and_with_arguments(self):
        """`git log` names no path to anchor to, and a bare `git status` has no
        trailing space for a `<cmd> *` pattern to match."""
        for repo in (self.tmp / "elsewhere", self.root / "inside"):
            patterns = {r["pattern"] for r in self.rules(repo)
                        if r["permission"] == "bash" and r["action"] == "allow"}
            self.assertIn("echo", patterns)
            self.assertIn("echo *", patterns)

    def test_path_free_commands_carry_their_own_write_guards(self):
        denied = [r["pattern"] for r in self.rules()
                  if r["permission"] == "bash" and r["action"] == "deny"]
        self.assertIn("echo *>*", denied)
        self.assertIn("echo *$(*", denied)

    def test_unanchored_allows_appear_only_when_the_repo_is_inside_a_root(self):
        outside = {r["pattern"] for r in self.rules(self.tmp / "elsewhere")
                   if r["permission"] == "bash" and r["action"] == "allow"}
        inside = {r["pattern"] for r in self.rules(self.root / "inside")
                  if r["permission"] == "bash" and r["action"] == "allow"}
        self.assertNotIn("cat *", outside)
        self.assertIn("cat *", inside)
        self.assertIn("cat", inside)


GIT_MUTATING = (
    "checkout", "gc", "reset", "clean", "commit", "add", "rm", "mv", "merge",
    "rebase", "apply", "am", "stash", "push", "pull", "fetch", "branch", "tag",
    "config", "remote", "worktree", "update-ref", "prune", "repack", "restore",
    "switch", "submodule", "init", "clone", "filter-branch", "cherry-pick",
    "revert", "sparse-checkout", "maintenance", "reflog", "symbolic-ref",
)


class TestReadRoots(unittest.TestCase):
    """`--read-root` has to reach all three layers at once, and the patterns it
    generates must not become write channels."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ask-opencode-roots-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.root = self.tmp / "tasks"
        (self.root / "inside").mkdir(parents=True)
        self.extra = self.tmp / "review-target"
        (self.extra / "crates").mkdir(parents=True)
        self.pre = policy.Preapproved(
            roots=(self.root,),
            commands=("ls", "cat", "head"),
            anywhere=("echo",),
            version_probes=("cargo", "~/.cargo/bin/cargo"),
        )

    def rules(self, repo=None, extra=None, pre=True):
        return policy.build(False, self.pre if pre else None, repo,
                            (self.extra,) if extra is None else extra)

    def allows(self, **kw) -> set[str]:
        return {r["pattern"] for r in self.rules(**kw)
                if r["permission"] == "bash" and r["action"] == "allow"}

    def denies(self, **kw) -> set[str]:
        return {r["pattern"] for r in self.rules(**kw)
                if r["permission"] == "bash" and r["action"] == "deny"}

    def externals(self, **kw) -> list[dict]:
        return [r for r in self.rules(**kw) if r["permission"] == "external_directory"]

    def matched(self, command: str, **kw) -> bool:
        return any(fnmatch.fnmatchcase(command, p) for p in self.allows(**kw))

    # --- the three layers -------------------------------------------------

    def test_an_extra_root_reaches_the_native_read_tools(self):
        """Without this the `read` tool cannot open the target at all — the deny
        fires before any bash rule is consulted."""
        patterns = [r["pattern"] for r in self.externals() if r["action"] == "allow"]
        self.assertIn(f"{self.extra}/*", patterns)
        self.assertIn(str(self.extra), patterns)

    def test_an_extra_root_anchors_the_preapproved_commands(self):
        self.assertTrue(self.matched(f"cat {self.extra}/crates/x.rs"))
        self.assertTrue(self.matched(f'head -n 40 "{self.extra}/crates/x.rs"'))
        # Sensitivity: only the commands the file listed, not any read command.
        self.assertFalse(self.matched(f"grep -n foo {self.extra}/crates/x.rs"))

    def test_the_root_itself_is_readable_without_a_trailing_slash(self):
        """`ls <root>` was gated because every pattern demanded a `/` after the
        root; it is the first thing a review session runs."""
        self.assertTrue(self.matched(f"ls {self.extra}"))
        self.assertTrue(self.matched(f"ls -la {self.extra}"))
        self.assertTrue(self.matched(f"ls {self.extra}/crates"))

    def test_a_sibling_directory_sharing_the_prefix_is_not_reachable(self):
        """Sensitivity: the root-itself pattern ends on the root, so it must not
        also open `<root>-secret`."""
        self.assertFalse(self.matched(f"ls {self.extra}-secret"))
        self.assertFalse(self.matched(f"cat {self.extra}-secret/f"))

    def test_the_file_roots_and_the_extra_roots_both_apply(self):
        allowed = self.allows()
        self.assertIn(f"cat *{self.root}/*", allowed)
        self.assertIn(f"cat *{self.extra}/*", allowed)

    def test_an_extra_root_works_with_no_preapproval_file(self):
        """Reaching a directory and not being asked about it are different
        grants: --read-root gives the first even under --no-preapproved."""
        rules = self.rules(pre=False)
        patterns = [r["pattern"] for r in rules
                    if r["permission"] == "external_directory" and r["action"] == "allow"]
        self.assertIn(f"{self.extra}/*", patterns)
        self.assertEqual([r for r in rules
                          if r["permission"] == "bash" and r["action"] == "allow"], [])

    def test_no_extra_roots_and_no_file_reproduces_the_original_ruleset(self):
        self.assertEqual(policy.build(False), policy.build(False, None, self.root, ()))

    # --- git read subcommands --------------------------------------------

    # --- version probes ---------------------------------------------------

    def test_a_version_probe_is_an_exact_pattern(self):
        allowed = self.allows()
        self.assertIn("cargo --version", allowed)
        self.assertIn(str(Path("~/.cargo/bin/cargo").expanduser()) + " --version",
                      allowed)
        self.assertIn("~/.cargo/bin/cargo --version", allowed)
        for pattern in allowed:
            if pattern.endswith("--version"):
                self.assertNotIn("*", pattern)

    def test_a_version_probe_does_not_open_the_tool_generally(self):
        self.assertFalse(self.matched("cargo test"))
        self.assertFalse(self.matched("cargo --version --config x=y"))
        self.assertFalse(self.matched("cargo --version > out"))

    # --- shape invariants -------------------------------------------------

    def test_every_allow_pattern_starts_with_a_literal_token(self):
        """A leading `*` turns a pattern into "the command ends with this",
        which any prefix satisfies — including one that runs something first."""
        for pattern in self.allows():
            self.assertFalse(pattern.startswith("*"),
                             f"{pattern!r} is anchored on its tail")

    def test_every_open_ended_allow_prefix_has_a_redirect_guard(self):
        """Derived from what was emitted, so a new allow shape cannot ship
        without its guard: `git -C <root> log` needs its own."""
        denied = self.denies()
        for pattern in self.allows():
            if not pattern.endswith(" *"):
                continue
            prefix = pattern[:-2]
            if "*" in prefix:  # anchored form; guarded by its command head
                prefix = prefix.split(" *", 1)[0]
            self.assertIn(f"{prefix} *>*", denied,
                          f"no redirect guard for {pattern!r}")

    def test_substitution_is_denied_for_every_pre_approved_head(self):
        """An internal `*` absorbs a substituted argument the same way it
        absorbs a redirect."""
        denied = self.denies()
        for head in ("cat", "ls", "head", "echo"):
            for form in policy.SUBSTITUTION_FORMS:
                self.assertIn(f"{head} *{form}*", denied)

    def test_the_guards_still_come_after_every_allow(self):
        rules = self.rules()
        last_allow = max(i for i, r in enumerate(rules) if r["action"] == "allow")
        first_guard = min(i for i, r in enumerate(rules)
                          if r["permission"] == "bash" and r["action"] == "deny")
        self.assertLess(last_allow, first_guard)

    def test_the_irreversible_blacklist_is_still_unreachable_from_above(self):
        rules = self.rules(repo=self.root / "inside")
        first_deny = min(i for i, r in enumerate(rules)
                         if r["permission"] == "bash" and r["action"] == "deny")
        self.assertEqual([r for r in rules[first_deny:] if r["action"] == "allow"], [])


class TestReadRootSession(AskOpencodeCase):
    """The flag has to survive the trip through the CLI into the session body."""

    def preapproved_file(self, payload=None) -> dict:
        path = self.tmp / "preapproved.json"
        path.write_text(json.dumps(payload or {
            "roots": [str(self.tmp / "notes")],
            "bash": ["cat"],
            "version_probes": ["cargo"],
        }))
        (self.tmp / "notes").mkdir(exist_ok=True)
        return {"ASK_OPENCODE_PREAPPROVED": str(path)}

    def target(self) -> Path:
        path = self.tmp / "review-target"
        path.mkdir(exist_ok=True)
        return path

    def test_read_root_reaches_the_session_ruleset(self):
        target = self.target()
        proc = self.start(extra=("--read-root", str(target)), env=self.preapproved_file())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        rules = self.session_create_body()["permission"]
        self.assertIn({"permission": "external_directory",
                       "pattern": f"{target}/*", "action": "allow"}, rules)
        self.assertIn({"permission": "bash",
                       "pattern": f"cat *{target}/*", "action": "allow"}, rules)

    def test_the_turn_header_names_the_read_root(self):
        """A widening the caller cannot see is the failure this line prevents."""
        target = self.target()
        proc = self.start(extra=("--read-root", str(target)), env=self.preapproved_file())
        conv = self.conv_of(proc)
        body = (Path(self.env["ASK_OPENCODE_STATE_DIR"]) / conv / "turns" / "1.md").read_text()
        self.assertIn("--read-root", body)
        self.assertIn(str(target), body)

    def test_read_root_is_repeatable(self):
        first, second = self.tmp / "one", self.tmp / "two"
        first.mkdir()
        second.mkdir()
        self.start(extra=("--read-root", str(first), "--read-root", str(second)),
                   env=self.preapproved_file())
        patterns = [r["pattern"] for r in self.session_create_body()["permission"]
                    if r["permission"] == "external_directory" and r["action"] == "allow"]
        self.assertIn(f"{first}/*", patterns)
        self.assertIn(f"{second}/*", patterns)

    def test_a_missing_read_root_is_a_usage_error(self):
        proc = self.start(extra=("--read-root", str(self.tmp / "absent")))
        self.assertEqual(proc.returncode, 4)
        self.assertIn("--read-root", proc.stderr)

    def test_read_root_still_opens_the_root_under_no_preapproved(self):
        target = self.target()
        self.start(extra=("--read-root", str(target), "--no-preapproved"),
                   env=self.preapproved_file())
        rules = self.session_create_body()["permission"]
        self.assertIn({"permission": "external_directory",
                       "pattern": f"{target}/*", "action": "allow"}, rules)
        self.assertEqual([r for r in rules
                          if r["permission"] == "bash" and r["action"] == "allow"], [])


class TestPreapprovedSession(AskOpencodeCase):
    def preapproved_file(self, payload) -> dict:
        path = self.tmp / "preapproved.json"
        path.write_text(json.dumps(payload))
        return {"ASK_OPENCODE_PREAPPROVED": str(path)}

    def test_the_file_reaches_the_session_ruleset(self):
        notes = self.tmp / "notes"
        notes.mkdir()
        env = self.preapproved_file({"roots": [str(notes)], "bash": ["cat"]})
        proc = self.start(env=env)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        rules = self.session_create_body()["permission"]
        self.assertIn({"permission": "external_directory",
                       "pattern": f"{notes}/*", "action": "allow"}, rules)
        self.assertIn({"permission": "bash",
                       "pattern": f"cat *{notes}/*", "action": "allow"}, rules)
        self.assertIn({"permission": "bash", "pattern": "cat *>*", "action": "deny"}, rules)

    def test_every_turn_file_says_what_was_pre_approved(self):
        """A ruleset loosened where the caller cannot see it is the failure this
        line exists to prevent."""
        notes = self.tmp / "notes"
        notes.mkdir()
        env = self.preapproved_file({"roots": [str(notes)], "bash": ["cat"]})
        proc = self.start(env=env)
        conv = self.conv_of(proc)
        body = (Path(self.env["ASK_OPENCODE_STATE_DIR"]) / conv / "turns" / "1.md").read_text()
        self.assertIn("pre-approved", body)
        self.assertIn(str(notes), body)

    def test_no_preapproved_restores_the_fully_gated_ruleset(self):
        notes = self.tmp / "notes"
        notes.mkdir()
        env = self.preapproved_file({"roots": [str(notes)], "bash": ["cat"]})
        self.start(extra=("--no-preapproved",), env=env)
        self.assertEqual(self.session_create_body()["permission"], policy.build(write=False))

    def test_a_malformed_file_stops_the_conversation(self):
        path = self.tmp / "preapproved.json"
        path.write_text("{not json")
        proc = self.start(env={"ASK_OPENCODE_PREAPPROVED": str(path)})
        self.assertEqual(proc.returncode, 4)
        self.assertIn("preapproved.json", proc.stderr)


class TestDroppedPrompt(AskOpencodeCase):
    """A prompt the server cannot serve is accepted with 204 and then never
    answered. Measured: a healthy turn creates its assistant message in 0.27s,
    so its absence is a rejection, not slowness."""

    script = {"turns": [{"no_reply": True}]}

    def test_a_dropped_prompt_fails_instead_of_running_out_the_budget(self):
        proc = self.start(env={"ASK_OPENCODE_REPLY_GRACE_S": "1"})
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(self.status_of(proc), "failed")
        conv = self.conv_of(proc)
        body = (Path(self.env["ASK_OPENCODE_STATE_DIR"]) / conv / "turns" / "1.md").read_text()
        self.assertIn("no reply", body)

    def test_a_turn_still_being_written_is_not_mistaken_for_a_dropped_one(self):
        """Sensitivity: `hang` leaves the assistant message open, which must
        still report `running`, not `failed`."""
        self.script_path.write_text(json.dumps({"turns": [{"hang": True}]}))
        proc = self.start(env={"ASK_OPENCODE_REPLY_GRACE_S": "1"}, extra=("--wait", "2"))
        self.assertEqual(proc.returncode, 10)
        self.assertIn("running", proc.stdout)


class TestModelPreflight(AskOpencodeCase):
    script = {"turns": [], "config": {"model": "test-provider/not-a-real-model"}}

    def test_a_model_the_server_lacks_is_refused_before_the_turn(self):
        proc = self.start()
        self.assertEqual(proc.returncode, 4)
        self.assertIn("not-a-real-model", proc.stderr)
        self.assertEqual([e for e in self.requests()
                          if e["path"].endswith("/prompt_async")], [])

    def test_the_error_names_where_the_model_came_from(self):
        proc = self.start()
        self.assertIn("opencode config", proc.stderr)

    def test_an_explicit_model_is_checked_too(self):
        proc = self.start(extra=("--model", "test-provider/nope"))
        self.assertEqual(proc.returncode, 4)
        self.assertIn("--model", proc.stderr)

    def test_a_model_the_server_has_is_allowed_through(self):
        self.script_path.write_text(json.dumps(
            {"turns": [{"text": "ok"}], "config": {"model": "test-provider/model-a"}}))
        proc = self.start()
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_a_server_that_will_not_list_providers_is_not_second_guessed(self):
        self.script_path.write_text(json.dumps(
            {"turns": [{"text": "ok"}],
             "config": {"model": "test-provider/not-a-real-model"},
             "providers": {}}))
        proc = self.start()
        self.assertEqual(proc.returncode, 0, proc.stderr)


class TestVariantPreflight(AskOpencodeCase):
    script = {
        "turns": [{"text": "ok"}],
        "providers": {"providers": [{"id": "test-provider", "models": {
            "model-a": {"variants": {"high": {}, "xhigh": {}}}}}]},
    }

    def test_the_variant_travels_in_every_prompt_body(self):
        proc = self.start(
            extra=("--model", "test-provider/model-a", "--variant", "xhigh"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        sent = self.run_ask("send", self.conv_of(proc), stdin="and again")
        self.assertEqual(sent.returncode, 0, sent.stderr)
        prompts = [e for e in self.requests()
                   if e["path"].endswith("/prompt_async")]
        self.assertEqual([e["body"].get("variant") for e in prompts],
                         ["xhigh", "xhigh"])

    def test_a_variant_the_model_does_not_list_is_refused_before_the_turn(self):
        proc = self.start(
            extra=("--model", "test-provider/model-a", "--variant", "nope"))
        self.assertEqual(proc.returncode, 4)
        self.assertIn("nope", proc.stderr)
        self.assertIn("xhigh", proc.stderr)
        self.assertEqual([e for e in self.requests()
                          if e["path"].endswith("/prompt_async")], [])

    def test_a_model_that_lists_no_variants_is_not_second_guessed(self):
        self.script_path.write_text(json.dumps({"turns": [{"text": "ok"}]}))
        proc = self.start(
            extra=("--model", "test-provider/model-a", "--variant", "xhigh"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        prompts = [e for e in self.requests()
                   if e["path"].endswith("/prompt_async")]
        self.assertEqual([e["body"].get("variant") for e in prompts], ["xhigh"])


class TestStart(AskOpencodeCase):
    def test_start_reports_summary_and_writes_the_answer_file(self):
        proc = self.start()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("[ask-opencode]", proc.stdout)
        self.assertEqual(self.status_of(proc), "converged")
        conv = self.conv_of(proc)
        body = (Path(self.env["ASK_OPENCODE_STATE_DIR"]) / conv / "turns" / "1.md").read_text()
        self.assertIn("here is the answer", body)

    def test_the_answering_model_is_reported_on_every_turn(self):
        """Nothing here picks the model: without --model opencode resolves the
        last one used in its TUI, so which one answered must never be silent."""
        proc = self.start()
        self.assertIn("test-provider/model-a", proc.stdout)
        conv = self.conv_of(proc)
        body = (Path(self.env["ASK_OPENCODE_STATE_DIR"]) / conv / "turns" / "1.md").read_text()
        self.assertIn("test-provider/model-a", body.splitlines()[0])

    def test_a_non_default_variant_is_reported_too(self):
        """The TUI's variant is not inherited by API-created sessions, so a
        session here can run at a different reasoning effort than the same
        model does in the TUI."""
        self.script_path.write_text(json.dumps(
            {"turns": [{"text": "hi", "variant": "max"}]}))
        proc = self.start()
        self.assertIn("test-provider/model-a:max", proc.stdout)

    def test_session_is_created_with_an_explicit_ruleset(self):
        """Never rely on opencode's implicit baseline, which allows everything."""
        self.start()
        body = self.session_create_body()
        self.assertEqual(body["permission"], policy.build(write=False))

    def test_write_flag_reaches_the_ruleset(self):
        self.start(extra=("--write",))
        self.assertEqual(self.session_create_body()["permission"], policy.build(write=True))

    def test_prompt_carries_a_client_minted_message_id_and_the_schema(self):
        self.start()
        prompts = [e for e in self.requests() if e["path"].endswith("/prompt_async")]
        self.assertEqual(len(prompts), 1)
        self.assertTrue(str(prompts[0]["body"]["messageID"]).startswith("msg_"))
        sent = prompts[0]["body"]["parts"][0]["text"]
        self.assertIn("do the thing", sent)
        self.assertIn('"open_questions"', sent)

    def test_the_format_field_is_never_sent(self):
        """Storing any value in prompt_async's `format` — `{"type":"text"}`
        included — makes every later read of the session's messages fail with
        `Expected OutputFormat…` on 1.18.21, which bricks the conversation."""
        self.start()
        self.run_ask("send", self.conv_of(self.start()), stdin="again")
        prompts = [e for e in self.requests() if e["path"].endswith("/prompt_async")]
        self.assertTrue(prompts)
        for prompt in prompts:
            self.assertNotIn("format", prompt["body"])

    def test_schema_none_sends_the_bare_task(self):
        self.start(extra=("--schema", "none"))
        prompts = [e for e in self.requests() if e["path"].endswith("/prompt_async")]
        self.assertEqual(prompts[0]["body"]["parts"][0]["text"], "do the thing")

    def test_missing_repo_is_a_usage_error(self):
        proc = self.run_ask("start", "--repo", str(self.tmp / "nope"), stdin="x")
        self.assertEqual(proc.returncode, 4)

    def test_empty_task_is_refused(self):
        proc = self.start(task="   ")
        self.assertEqual(proc.returncode, 4)


class TestCompletionDetection(AskOpencodeCase):
    script = {"turns": [{"text": "first answer"}, {"text": "second answer"}]}

    def test_completion_never_consults_session_status(self):
        """Session status reports idle by omission, so it cannot tell 'finished'
        from 'not started yet'. The reply is claimed by parentID instead."""
        self.start()
        entries = self.requests()
        prompt_at = next(i for i, e in enumerate(entries) if e["path"].endswith("/prompt_async"))
        after = [e["path"] for e in entries[prompt_at + 1:]]
        self.assertNotIn("/session/status", after)
        self.assertTrue(any(p.endswith("/message") for p in after))

    def test_second_turn_is_not_served_the_first_turns_reply(self):
        proc = self.start()
        conv = self.conv_of(proc)
        first = (Path(self.env["ASK_OPENCODE_STATE_DIR"]) / conv / "turns" / "1.md").read_text()
        follow = self.run_ask("send", conv, stdin="and now?")
        self.assertEqual(follow.returncode, 0, follow.stderr)
        second = (Path(self.env["ASK_OPENCODE_STATE_DIR"]) / conv / "turns" / "2.md").read_text()
        self.assertIn("first answer", first)
        self.assertIn("second answer", second)
        self.assertNotEqual(first, second)

    def test_each_turn_uses_a_fresh_message_id(self):
        proc = self.start()
        self.run_ask("send", self.conv_of(proc), stdin="again")
        mids = [e["body"]["messageID"] for e in self.requests()
                if e["path"].endswith("/prompt_async")]
        self.assertEqual(len(mids), 2)
        self.assertNotEqual(mids[0], mids[1])


class TestPermissionGate(AskOpencodeCase):
    script = {"turns": [{"gates": [PERMISSION_GATE], "text": "2026"}]}

    def test_turn_stops_at_the_gate_with_the_command_in_the_file(self):
        proc = self.start()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.status_of(proc), "needs_permission")
        conv = self.conv_of(proc)
        body = (Path(self.env["ASK_OPENCODE_STATE_DIR"]) / conv / "turns" / "1.md").read_text()
        self.assertIn("date +%Y", body)
        self.assertIn("bash", body)

    def test_approve_answers_once_and_never_always(self):
        """`always` would persist opencode's generalised pattern and stop asking,
        removing the approval point this tool exists to provide."""
        proc = self.start()
        self.run_ask("approve", self.conv_of(proc))
        replies = [e for e in self.requests() if "/permissions/" in e["path"]]
        self.assertEqual(len(replies), 1)
        self.assertEqual(replies[0]["body"], {"response": "once"})

    def test_approve_resumes_the_same_message_id(self):
        proc = self.start()
        conv = self.conv_of(proc)
        resumed = self.run_ask("approve", conv)
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertEqual(self.status_of(resumed), "converged")
        meta = json.loads(
            (Path(self.env["ASK_OPENCODE_STATE_DIR"]) / conv / "meta.json").read_text())
        self.assertEqual({t["mid"] for t in meta["turns"]}, {meta["turns"][0]["mid"]})

    def test_reject_answers_reject(self):
        proc = self.start()
        self.run_ask("reject", self.conv_of(proc), "--reason", "no")
        replies = [e for e in self.requests() if "/permissions/" in e["path"]]
        self.assertEqual(replies[0]["body"], {"response": "reject"})

    def test_a_reply_with_no_prose_still_reports_what_the_tools_did(self):
        """A rejected tool call ends the turn with no final message; the caller
        must not be handed an empty file."""
        proc = self.start()
        conv = self.conv_of(proc)
        self.run_ask("reject", conv)
        turns = json.loads(
            (Path(self.env["ASK_OPENCODE_STATE_DIR"]) / conv / "meta.json").read_text())["turns"]
        body = (Path(self.env["ASK_OPENCODE_STATE_DIR"]) / conv / "turns"
                / f"{turns[-1]['n']}.md").read_text()
        self.assertIn("without a final message", body)
        self.assertIn("rejected permission", body)

    def test_send_is_refused_while_a_gate_is_open(self):
        proc = self.start()
        blocked = self.run_ask("send", self.conv_of(proc), stdin="next")
        self.assertEqual(blocked.returncode, 4)
        self.assertIn("gate", blocked.stderr)

    def test_answer_is_refused_on_a_permission_gate(self):
        proc = self.start()
        wrong = self.run_ask("answer", self.conv_of(proc), "postgres")
        self.assertEqual(wrong.returncode, 4)


class TestMultipleGatesInOneTurn(AskOpencodeCase):
    """One tool call can trip more than one gate: external_directory fires
    before bash and is not covered by the bash rules."""

    script = {"turns": [{"gates": [EXTERNAL_GATE, PERMISSION_GATE], "text": "done"}]}

    def test_each_gate_is_its_own_turn_and_all_are_answerable(self):
        proc = self.start()
        conv = self.conv_of(proc)
        self.assertEqual(self.status_of(proc), "needs_permission")

        second = self.run_ask("approve", conv)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(self.status_of(second), "needs_permission")

        third = self.run_ask("approve", conv)
        self.assertEqual(third.returncode, 0, third.stderr)
        self.assertEqual(self.status_of(third), "converged")

        gates = [e["path"] for e in self.requests() if "/permissions/" in e["path"]]
        self.assertEqual(len(gates), 2)
        self.assertIn("per_ext_0", gates[0])
        self.assertIn("per_bash_0", gates[1])


class TestQuestionGate(AskOpencodeCase):
    script = {"turns": [{"gates": [QUESTION_GATE], "text": "postgres it is"}]}

    def test_question_is_surfaced_with_its_options(self):
        proc = self.start()
        self.assertEqual(self.status_of(proc), "needs_answer")
        conv = self.conv_of(proc)
        body = (Path(self.env["ASK_OPENCODE_STATE_DIR"]) / conv / "turns" / "1.md").read_text()
        self.assertIn("Which database?", body)
        self.assertIn("postgres", body)

    def test_answer_replies_with_the_labels_and_resumes(self):
        proc = self.start()
        resumed = self.run_ask("answer", self.conv_of(proc), "postgres")
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertEqual(self.status_of(resumed), "converged")
        replies = [e for e in self.requests() if e["path"].endswith("/reply")]
        self.assertEqual(replies[0]["body"], {"answers": [["postgres"]]})

    def test_approve_is_refused_on_a_question_gate(self):
        proc = self.start()
        wrong = self.run_ask("approve", self.conv_of(proc))
        self.assertEqual(wrong.returncode, 4)


class TestEmitModes(AskOpencodeCase):
    script = {"turns": [{"text": "short answer"}]}

    def test_summary_mode_prints_one_line_without_body(self):
        proc = self.start(extra=("--emit", "summary"))
        self.assertEqual(len(proc.stdout.strip().splitlines()), 1)
        self.assertNotIn("short answer", proc.stdout)

    def test_result_mode_prints_the_body_without_summary(self):
        proc = self.start(extra=("--emit", "result"))
        self.assertNotIn("[ask-opencode]", proc.stdout)
        self.assertIn("short answer", proc.stdout)

    def test_auto_inlines_a_small_body(self):
        proc = self.start()
        self.assertIn("[ask-opencode]", proc.stdout)
        self.assertIn("short answer", proc.stdout)


class TestEmitLargeBody(AskOpencodeCase):
    script = {"turns": [{"text": "x" * 4000}]}

    def test_auto_omits_a_body_over_the_inline_limit(self):
        """Monitor truncates a large notification, so a long body degrades to a
        path rather than arriving cut in half."""
        proc = self.start()
        self.assertIn("[ask-opencode]", proc.stdout)
        self.assertNotIn("x" * 4000, proc.stdout)

    def test_the_limit_is_configurable(self):
        proc = self.start(env={"ASK_OPENCODE_INLINE_BYTES": "100000"})
        self.assertIn("x" * 4000, proc.stdout)


class TestFailureHandling(AskOpencodeCase):
    script = {"turns": [{"error": {"name": "UnknownError",
                                   "data": {"message": "provider exploded"}}}]}

    def test_message_error_is_exit_2_and_the_reason_is_in_the_file(self):
        proc = self.start()
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(self.status_of(proc), "failed")
        conv = self.conv_of(proc)
        body = (Path(self.env["ASK_OPENCODE_STATE_DIR"]) / conv / "turns" / "1.md").read_text()
        self.assertIn("provider exploded", body)


class TestUnstructuredReply(AskOpencodeCase):
    script = {"turns": [{"unstructured": True, "text": "I ignored the schema."}]}

    def test_unparsed_structured_output_degrades_instead_of_failing(self):
        proc = self.start()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.status_of(proc), "unknown")
        conv = self.conv_of(proc)
        body = (Path(self.env["ASK_OPENCODE_STATE_DIR"]) / conv / "turns" / "1.md").read_text()
        self.assertIn("I ignored the schema.", body)


class TestWaitingAndCancel(AskOpencodeCase):
    script = {"turns": [{"text": "slow answer", "delay_s": 3}]}

    def test_wait_budget_elapsed_reports_exit_10(self):
        proc = self.start(extra=("--wait", "0.5"))
        self.assertEqual(proc.returncode, 10)
        self.assertIn("running", proc.stdout)

    def test_wait_reattaches_and_delivers_the_result(self):
        proc = self.start(extra=("--wait", "0.5"))
        conv = self.conv_of(proc)
        resumed = self.run_ask("wait", conv, "--wait", "30")
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertEqual(self.status_of(resumed), "converged")
        self.assertIn("slow answer", resumed.stdout)

    def test_send_is_refused_while_a_turn_is_in_flight(self):
        proc = self.start(extra=("--wait", "0.5"))
        blocked = self.run_ask("send", self.conv_of(proc), stdin="next")
        self.assertEqual(blocked.returncode, 4)
        self.assertIn("still running", blocked.stderr)


class TestCancel(AskOpencodeCase):
    script = {"turns": [{"hang": True}]}

    def test_cancel_aborts_the_session_and_reports_exit_3(self):
        proc = self.start(extra=("--wait", "0.5"))
        self.assertEqual(proc.returncode, 10)
        conv = self.conv_of(proc)
        cancelled = self.run_ask("cancel", conv)
        self.assertEqual(cancelled.returncode, 3)
        self.assertTrue(any(e["path"].endswith("/abort") for e in self.requests()))

    def test_cancel_on_an_idle_conversation_is_a_no_op(self):
        proc = self.start(extra=("--wait", "0.5"))
        conv = self.conv_of(proc)
        self.run_ask("cancel", conv)
        again = self.run_ask("cancel", conv)
        self.assertEqual(again.returncode, 0)
        self.assertIn("no turn in flight", again.stdout)


class TestInspection(AskOpencodeCase):
    def test_list_shows_conversations(self):
        proc = self.start()
        listing = self.run_ask("list")
        self.assertIn(self.conv_of(proc), listing.stdout)

    def test_show_prints_a_stored_turn(self):
        proc = self.start()
        shown = self.run_ask("show", self.conv_of(proc))
        self.assertIn("here is the answer", shown.stdout)

    def test_unknown_conversation_is_a_usage_error(self):
        proc = self.run_ask("show", "nope-000000")
        self.assertEqual(proc.returncode, 4)
        self.assertIn("no such conversation", proc.stderr)


class TestServerGuard(unittest.TestCase):
    def test_explicit_url_that_is_not_opencode_is_a_usage_error(self):
        """A foreign service on the port must fail loudly, not be driven blindly."""
        tmp = Path(tempfile.mkdtemp(prefix="ask-opencode-guard-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        port = free_port()
        server = subprocess.Popen(
            [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
            cwd=tmp, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.addCleanup(server.wait)
        self.addCleanup(server.kill)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.05)
        proc = subprocess.run(
            [sys.executable, str(ASK), "start", "--repo", str(tmp)],
            input="hi", capture_output=True, text=True, timeout=60,
            env={**os.environ,
                 "ASK_OPENCODE_URL": f"http://127.0.0.1:{port}",
                 "ASK_OPENCODE_STATE_DIR": str(tmp / "state")},
        )
        self.assertEqual(proc.returncode, 4)
        self.assertIn("not answering as an opencode server", proc.stderr)


class TestUnits(unittest.TestCase):
    def test_schema_satisfies_strict_structured_output(self):
        schema = json.loads(SCHEMA.read_text())
        self.assertEqual(schema["type"], "object")
        self.assertIs(schema["additionalProperties"], False)
        self.assertEqual(set(schema["required"]), set(schema["properties"]))

    def test_message_id_is_minted_with_the_expected_prefix(self):
        mid = ask_opencode.mint_message_id()
        self.assertTrue(mid.startswith("msg_"))
        self.assertNotEqual(mid, ask_opencode.mint_message_id())

    def test_token_formatting(self):
        self.assertEqual(ask_opencode.fmt_usage({}), "tok n/a")
        self.assertEqual(ask_opencode.fmt_usage({"input": 300, "output": 200}), "500 tok")
        self.assertEqual(ask_opencode.fmt_usage({"total": 2500}), "2.5k tok")

    def test_duration_formatting(self):
        self.assertEqual(ask_opencode.fmt_duration(45), "45s")
        self.assertEqual(ask_opencode.fmt_duration(125), "2m05s")

    def test_model_spec_must_be_provider_slash_model(self):
        self.assertEqual(ask_opencode.parse_model("test-provider/model-a"),
                         {"providerID": "test-provider", "id": "model-a"})
        self.assertIsNone(ask_opencode.parse_model(None))
        with self.assertRaises(ask_opencode.UsageError):
            ask_opencode.parse_model("model-a")

    def test_structured_envelope_is_read_from_the_last_fenced_block(self):
        """A reply that quotes the schema before answering must not be read as
        its own answer."""
        text = (
            'Here is the schema I was given:\n\n'
            '```json\n{"status": "blocked", "answer": "SCHEMA ECHO", "open_questions": []}\n```\n\n'
            'And here is my reply:\n\n'
            '```json\n{"status": "converged", "answer": "REAL", "open_questions": []}\n```\n'
        )
        parsed = ask_opencode.parse_structured(text)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["answer"], "REAL")

    def test_structured_envelope_survives_fences_inside_the_answer(self):
        """Observed against a live model: quoting command output puts a closing
        fence inside `answer`, which fence-matching truncates the object."""
        answer = "I ran it. Raw output:\n\n```\nsome ``` output\n```\n\nThat is all."
        text = "```json\n" + json.dumps(
            {"status": "converged", "answer": answer, "open_questions": []}, indent=2
        ) + "\n```\n"
        parsed = ask_opencode.parse_structured(text)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["answer"], answer)

    def test_a_status_object_quoted_inside_the_answer_does_not_win(self):
        answer = 'The tool replied with {"status": "denied", "answer": "nope"} verbatim.'
        text = json.dumps({"status": "converged", "answer": answer, "open_questions": []})
        parsed = ask_opencode.parse_structured(text)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["status"], "converged")

    def test_structured_envelope_accepts_a_bare_json_reply(self):
        parsed = ask_opencode.parse_structured(
            '{"status": "converged", "answer": "hi", "open_questions": []}')
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["status"], "converged")

    def test_prose_has_no_structured_envelope(self):
        self.assertIsNone(ask_opencode.parse_structured("just some prose"))
        self.assertIsNone(ask_opencode.parse_structured(""))
        self.assertIsNone(ask_opencode.parse_structured('```json\n{"answer": "no status"}\n```'))

    def test_gate_command_prefers_the_raw_command(self):
        self.assertEqual(
            ask_opencode.gate_command({"metadata": {"command": "date +%Y"},
                                       "patterns": ["date *"]}),
            "date +%Y")
        self.assertEqual(ask_opencode.gate_command({"patterns": ["date *"]}), "date *")


def message(mid, role, parent=None, *, summary=None, parts=None, finish="stop"):
    """A message the way the server stores one.

    A completed assistant message always carries a `finish`, and the default is
    the terminal one: a fixture without it would be classified `unresolved` and
    would test reply detection against a shape the server never produces.
    """
    info = {"id": mid, "role": role, "time": {"created": 1, "completed": 2}}
    if role == "assistant" and finish is not None:
        info["finish"] = finish
    if parent is not None:
        info["parentID"] = parent
    if summary is not None:
        info["summary"] = summary
    return {"info": info, "parts": parts if parts is not None else [{"type": "text", "text": mid}]}


class TestReplyAfterCompaction(unittest.TestCase):
    """Reply detection over the message shape a real compaction leaves behind.

    Transcribed from a compaction captured on 1.18.21: the server inserts its
    own compaction message, answers it with a summary, appends a synthetic
    `Continue …` user message, and parents the real answer to that one.
    """

    OURS = {"msg_one", "msg_two", "msg_three"}

    def setUp(self):
        self.messages = [
            message("msg_one", "user", summary={"diffs": []}),
            message("reply_one", "assistant", "msg_one", parts=[{"type": "text", "text": "ANSWER ONE"}]),
            message("cmp_a", "user", summary={"diffs": []}, parts=[{"type": "compaction", "auto": True}]),
            message("sum_a", "assistant", "cmp_a", summary=True,
                    parts=[{"type": "text", "text": "SUMMARY"}]),
            message("cont_a", "user", summary={"diffs": []},
                    parts=[{"type": "text", "text": "Continue if you have next steps",
                            "synthetic": True, "metadata": {"compaction_continue": True}}]),
            message("cont_reply", "assistant", "cont_a", parts=[{"type": "text", "text": "CONTINUATION"}]),
            message("msg_two", "user"),
            message("cmp_b", "user", parts=[{"type": "compaction", "auto": True}]),
            message("sum_b", "assistant", "cmp_b", summary=True,
                    parts=[{"type": "text", "text": "SUMMARY"}]),
            message("cont_b", "user",
                    parts=[{"type": "text", "text": "Continue if you have next steps",
                            "synthetic": True, "metadata": {"compaction_continue": True}}]),
            message("reply_two", "assistant", "cont_b", parts=[{"type": "text", "text": "ANSWER TWO"}]),
            # A turn that used a tool and then compacted: the prompt's own child
            # is a completed intermediate step, and the answer is on the chain.
            message("msg_three", "user"),
            message("step_three", "assistant", "msg_three", finish="tool-calls",
                    parts=[{"type": "text", "text": "CALLING A TOOL"}]),
            message("cmp_c", "user", parts=[{"type": "compaction", "auto": True}]),
            message("sum_c", "assistant", "cmp_c", summary=True,
                    parts=[{"type": "text", "text": "SUMMARY"}]),
            message("cont_c", "user",
                    parts=[{"type": "text", "text": "Continue if you have next steps",
                            "synthetic": True, "metadata": {"compaction_continue": True}}]),
            message("reply_three", "assistant", "cont_c",
                    parts=[{"type": "text", "text": "ANSWER THREE"}]),
        ]

    def test_a_completed_tool_step_does_not_hide_the_compaction_chain(self):
        """The prompt's own child is completed, but it is a step. Letting a
        completed child win here would pin the turn on it for good: the chain
        holds the answer and nothing else is ever parented to the prompt."""
        found = ask_opencode.find_reply(self.messages, "msg_three", self.OURS)
        assert found is not None
        self.assertEqual(found["info"]["id"], "reply_three")

    def test_a_reply_parented_to_the_continue_message_is_still_the_turns_reply(self):
        found = ask_opencode.find_reply(self.messages, "msg_two", self.OURS)
        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found["info"]["id"], "reply_two")

    def test_a_compaction_summary_is_never_the_reply(self):
        for mid in ("msg_one", "msg_two"):
            found = ask_opencode.find_reply(self.messages, mid, self.OURS)
            assert found is not None
            self.assertFalse(ask_opencode.is_summary(found))

    def test_a_compaction_after_a_finished_turn_does_not_rewrite_its_answer(self):
        found = ask_opencode.find_reply(self.messages, "msg_one", self.OURS)
        assert found is not None
        # Turn one finished before the server compacted, so its own reply stands:
        # not the continuation, and certainly not the next turn's answer.
        self.assertEqual(found["info"]["id"], "reply_one")

    def test_a_turn_compacted_before_its_reply_finished_follows_the_chain(self):
        messages = list(self.messages[:2])
        messages[1]["info"]["time"] = {"created": 1}  # started, never finished
        messages += self.messages[2:6]
        found = ask_opencode.find_reply(messages, "msg_one", {"msg_one"})
        assert found is not None
        self.assertEqual(found["info"]["id"], "cont_reply")

    def test_the_ordinary_case_is_unchanged(self):
        plain = [message("msg_one", "user"), message("reply_one", "assistant", "msg_one")]
        found = ask_opencode.find_reply(plain, "msg_one", {"msg_one"})
        assert found is not None
        self.assertEqual(found["info"]["id"], "reply_one")

    def test_a_prompt_with_no_reply_yet_finds_nothing(self):
        found = ask_opencode.find_reply([message("msg_one", "user")], "msg_one", {"msg_one"})
        self.assertIsNone(found)

    def test_a_prompt_that_is_not_in_the_session_finds_nothing(self):
        self.assertIsNone(ask_opencode.find_reply(self.messages, "msg_absent", self.OURS))

    def test_compacting_is_bounded_by_the_next_prompt_we_sent(self):
        opening = self.messages[:8]  # through our second prompt and its compaction
        self.assertTrue(ask_opencode.compacting(opening, "msg_two", self.OURS))
        # Turn two's compaction is not visible to a turn that ended before it.
        self.assertFalse(ask_opencode.compacting(self.messages[:2], "msg_one", self.OURS))

    def test_a_turn_with_no_compaction_reports_none(self):
        plain = [message("msg_one", "user"), message("reply_one", "assistant", "msg_one")]
        self.assertFalse(ask_opencode.compacting(plain, "msg_one", {"msg_one"}))

    def test_own_message_ids_reads_every_prompt_sent(self):
        meta = {"turns": [{"n": 1, "mid": "msg_one"}, {"n": 2, "mid": "msg_two"}, {"n": 3}]}
        self.assertEqual(ask_opencode.own_message_ids(meta), {"msg_one", "msg_two"})


class TestCompactedTurn(AskOpencodeCase):
    """End to end: a turn the server compacts mid-flight still converges."""

    script = {"turns": [{"compaction": True, "text": "COMPACTED ANSWER"}]}

    def test_a_compacted_turn_returns_the_answer_not_a_failure(self):
        proc = self.start()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.status_of(proc), "converged")
        body = self._turn_md(self.conv_of(proc), 1)
        self.assertIn("COMPACTED ANSWER", body)

    def test_the_summary_is_not_served_as_the_answer(self):
        proc = self.start()
        body = self._turn_md(self.conv_of(proc), 1)
        self.assertNotIn("summary of the session so far", body)

    def test_the_caller_is_told_the_session_was_compacted(self):
        proc = self.start()
        body = self._turn_md(self.conv_of(proc), 1)
        self.assertIn("compacted this session", body)
        record = json.loads(
            (Path(self.env["ASK_OPENCODE_STATE_DIR"]) / self.conv_of(proc)
             / "turns" / "1.json").read_text())
        self.assertTrue(record["compacted"])

    def _turn_md(self, conv, n):
        return (Path(self.env["ASK_OPENCODE_STATE_DIR"]) / conv / "turns" / f"{n}.md").read_text()


class TestCompactionInProgress(AskOpencodeCase):
    """A compaction still running is not a dropped prompt."""

    script = {"turns": [{"compaction": True, "delay_s": 4, "text": "LATE ANSWER"}]}

    def test_a_compacting_turn_is_reported_running_not_failed(self):
        proc = self.start(extra=("--wait", "1.5"),
                          env={"ASK_OPENCODE_REPLY_GRACE_S": "0.5"})
        self.assertEqual(proc.returncode, 10, proc.stdout + proc.stderr)
        self.assertIn("running", proc.stdout)
        self.assertIn("compacting", proc.stdout)

    def test_the_answer_is_there_once_the_compaction_finishes(self):
        proc = self.start(extra=("--wait", "1.5"),
                          env={"ASK_OPENCODE_REPLY_GRACE_S": "0.5"})
        conv = self.conv_of(proc)
        done = self.run_ask("wait", conv, "--wait", "20",
                            env={"ASK_OPENCODE_REPLY_GRACE_S": "0.5"})
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(self.status_of(done), "converged")
        body = (Path(self.env["ASK_OPENCODE_STATE_DIR"]) / conv / "turns" / "1.md").read_text()
        self.assertIn("LATE ANSWER", body)


class TestTwoTurnsAcrossACompaction(AskOpencodeCase):
    """Turn 2 answers turn 2, even when the server compacted in between."""

    script = {"turns": [{"text": "ANSWER ONE"}, {"compaction": True, "text": "ANSWER TWO"}]}

    def test_the_second_turn_does_not_serve_the_first_turns_reply(self):
        first = self.start()
        conv = self.conv_of(first)
        second = self.run_ask("send", conv, stdin="and now the second thing")
        self.assertEqual(second.returncode, 0, second.stderr)
        body = (Path(self.env["ASK_OPENCODE_STATE_DIR"]) / conv / "turns" / "2.md").read_text()
        self.assertIn("ANSWER TWO", body)
        self.assertNotIn("ANSWER ONE", body)


class StopCase(AskOpencodeCase):
    """`stop` resolves its target from the port, so the URL override is dropped."""

    def setUp(self):
        super().setUp()
        self.env = {k: v for k, v in self.env.items() if k != "ASK_OPENCODE_URL"}
        self.env["ASK_OPENCODE_PORT"] = str(self.port)

    def server_alive(self):
        return self.server.poll() is None

    def meta_of(self, conv):
        return json.loads(
            (Path(self.env["ASK_OPENCODE_STATE_DIR"]) / conv / "meta.json").read_text())


class TestStop(StopCase):
    script = {"turns": [{"text": "done"}]}

    def test_stopping_a_running_server_stops_it(self):
        proc = self.run_ask("stop")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("server stopped", proc.stdout)
        self.assertIn("SIGTERM", proc.stdout)
        self.assertFalse(self.server_alive())

    def test_stopping_when_nothing_listens_is_a_no_op(self):
        self.run_ask("stop")
        again = self.run_ask("stop")
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertIn("no server on port", again.stdout)

    def test_an_explicit_url_is_refused(self):
        proc = self.run_ask("stop", env={"ASK_OPENCODE_URL": f"http://127.0.0.1:{self.port}"})
        self.assertEqual(proc.returncode, 4)
        self.assertIn("did not start", proc.stderr)
        self.assertTrue(self.server_alive())

    def test_without_lsof_it_says_so_instead_of_guessing_a_pid(self):
        proc = self.run_ask("stop", env={"PATH": ""})
        self.assertEqual(proc.returncode, 4)
        self.assertIn("lsof", proc.stderr)
        self.assertTrue(self.server_alive())

    def test_a_finished_conversation_does_not_block_the_stop(self):
        started = self.start()
        self.assertEqual(self.status_of(started), "converged")
        proc = self.run_ask("stop")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertFalse(self.server_alive())


class TestStopRefusesLiveTurns(StopCase):
    script = {"turns": [{"hang": True}]}

    def _wedge(self):
        proc = self.start(extra=("--wait", "0.5"))
        self.assertEqual(proc.returncode, 10, proc.stdout + proc.stderr)
        return self.conv_of(proc)

    def test_a_turn_in_flight_refuses_the_stop(self):
        conv = self._wedge()
        proc = self.run_ask("stop")
        self.assertEqual(proc.returncode, 4)
        self.assertIn(conv, proc.stderr)
        self.assertTrue(self.server_alive())

    def test_force_cancels_the_turn_and_stops(self):
        conv = self._wedge()
        proc = self.run_ask("stop", "--force")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("cancelled", proc.stdout)
        self.assertFalse(self.server_alive())
        meta = self.meta_of(conv)
        self.assertIsNone(meta["current"])
        self.assertEqual(meta["turns"][-1]["status"], "cancelled")

    def test_a_conversation_on_another_port_is_not_this_servers_problem(self):
        conv = self._wedge()
        meta_path = Path(self.env["ASK_OPENCODE_STATE_DIR"]) / conv / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["port"] = self.port + 1
        meta_path.write_text(json.dumps(meta))
        proc = self.run_ask("stop")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertFalse(self.server_alive())

    def test_if_idle_leaves_a_busy_server_alone_and_never_cancels(self):
        conv = self._wedge()
        proc = self.run_ask("stop", "--if-idle", "0")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("busy", proc.stdout)
        self.assertTrue(self.server_alive())
        self.assertEqual(self.meta_of(conv)["current"], 1)


class TestStopIfIdle(StopCase):
    script = {"turns": [{"text": "done"}]}

    def test_recent_activity_keeps_the_server_alive(self):
        self.start()
        proc = self.run_ask("stop", "--if-idle", "600")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("was active", proc.stdout)
        self.assertTrue(self.server_alive())

    def test_an_idle_server_is_collected(self):
        self.start()
        proc = self.run_ask("stop", "--if-idle", "0")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("server stopped", proc.stdout)
        self.assertFalse(self.server_alive())

    def test_a_server_no_conversation_has_used_counts_as_idle(self):
        proc = self.run_ask("stop", "--if-idle", "600")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("server stopped", proc.stdout)
        self.assertFalse(self.server_alive())


class TestStopEscalates(StopCase):
    server_env = {"FAKE_OPENCODE_IGNORE_SIGTERM": "1"}

    def test_a_server_that_ignores_sigterm_is_killed(self):
        proc = self.run_ask("stop", "--timeout", "1")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("SIGKILL", proc.stdout)
        self.assertFalse(self.server_alive())


class TestStopRefusesAForeignServer(unittest.TestCase):
    def test_a_plain_http_server_on_the_port_is_not_signalled(self):
        port = free_port()
        server = subprocess.Popen(
            [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(server.wait)
        self.addCleanup(server.kill)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.05)
        env = {k: v for k, v in os.environ.items() if k != "ASK_OPENCODE_URL"}
        env["ASK_OPENCODE_PORT"] = str(port)
        proc = subprocess.run([sys.executable, str(ASK), "stop"],
                              capture_output=True, text=True, timeout=60, env=env)
        self.assertEqual(proc.returncode, 4)
        self.assertIn("not an opencode server", proc.stderr)
        self.assertIsNone(server.poll())


class TestIdleSeconds(unittest.TestCase):
    def test_never_used_is_unknown_rather_than_zero(self):
        self.assertIsNone(ask_opencode.idle_seconds([{"conv": "a", "turns": []}]))

    def test_idle_is_measured_from_the_end_of_the_last_turn(self):
        started = ask_opencode.datetime.now(ask_opencode.timezone.utc)
        metas = [{"conv": "a", "turns": [
            {"n": 1, "started_at": started.isoformat(), "duration_s": 0}]}]
        idle = ask_opencode.idle_seconds(metas)
        self.assertIsNotNone(idle)
        assert idle is not None
        self.assertLess(idle, 5)

    def test_the_newest_turn_wins(self):
        now = ask_opencode.datetime.now(ask_opencode.timezone.utc)
        old = (now - ask_opencode.datetime.resolution * 0).isoformat()
        metas = [
            {"conv": "a", "turns": [{"n": 1, "started_at": "2020-01-01T00:00:00+00:00",
                                     "duration_s": 1}]},
            {"conv": "b", "turns": [{"n": 1, "started_at": old, "duration_s": 0}]},
        ]
        idle = ask_opencode.idle_seconds(metas)
        assert idle is not None
        self.assertLess(idle, 5)

    def test_a_malformed_timestamp_is_ignored_rather_than_fatal(self):
        metas = [{"conv": "a", "turns": [{"n": 1, "started_at": "not-a-time"}]}]
        self.assertIsNone(ask_opencode.idle_seconds(metas))



class TestReviewedHeads(unittest.TestCase):
    """A command head reaches the bash lists only after someone has decided what
    a pre-approved read of it can be turned into."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ask-opencode-heads-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.root = self.tmp / "tasks"
        self.root.mkdir()

    def load(self, payload):
        path = self.tmp / "preapproved.json"
        path.write_text(json.dumps(payload))
        return policy.load_preapproved(path)

    def test_a_reviewed_head_is_accepted(self):
        loaded = self.load({"roots": [str(self.root)], "bash": ["cat"]})
        assert loaded is not None
        self.assertEqual(loaded.commands, ("cat",))

    def test_an_unreviewed_head_is_refused(self):
        with self.assertRaises(policy.PreapprovedError) as caught:
            self.load({"roots": [str(self.root)], "bash": ["awk"]})
        self.assertIn("awk", str(caught.exception))
        self.assertIn("REVIEWED_HEADS", str(caught.exception))

    def test_an_unreviewed_head_is_refused_in_the_anchorless_list_too(self):
        with self.assertRaises(policy.PreapprovedError):
            self.load({"bash_anywhere": ["awk"]})

    def test_every_head_the_example_file_uses_is_reviewed(self):
        shipped = policy.load_preapproved(EXAMPLE_PREAPPROVED)
        assert shipped is not None
        for entry in shipped.commands + shipped.anywhere:
            self.assertIn(entry.split()[0], policy.REVIEWED_HEADS)

    def test_the_two_heads_that_cannot_be_guarded_are_absent(self):
        """sed writes with its `w` command from inside the quoted script, where
        no pattern reaches it; perl's -e is the same shape."""
        self.assertNotIn("sed", policy.REVIEWED_HEADS)
        self.assertNotIn("perl", policy.REVIEWED_HEADS)


class TestFlagGuardMechanism(unittest.TestCase):
    """Every head on the table now carries an empty flag tuple, so the machinery
    that turns a head's flags into denies has no user in the shipped ruleset.
    It is kept because the table is what admits a head, and admitting one has to
    emit its guards; this pins the mechanism so it cannot rot unnoticed."""

    def test_a_heads_flags_become_denies_for_exactly_that_head(self):
        table = dict(policy.REVIEWED_HEADS, cat=("--danger", "-x"))
        with mock.patch.object(policy, "REVIEWED_HEADS", table):
            guards = policy.unsafe_guards(("cat", "ls"))
        self.assertIn("cat *--danger*", guards)
        self.assertIn("cat -x*", guards)
        self.assertIn("cat * -x*", guards)
        self.assertNotIn("ls *--danger*", guards)

    def test_with_no_flags_only_the_shape_guards_are_emitted(self):
        guards = policy.unsafe_guards(("cat",))
        self.assertEqual(
            sorted(guards), sorted(["cat *>*", "cat *$(*", "cat *`*", "cat *<(*"])
        )


class TestShippedRulesetVectors(unittest.TestCase):
    """Decisions the shipped pre-approval actually produces, controls included."""

    def setUp(self):
        self.pre = policy.load_preapproved(EXAMPLE_PREAPPROVED)
        assert self.pre is not None
        self.root = str(self.pre.roots[-1])
        self.rules = policy.build(False, self.pre, Path("/tmp/somewhere-else"), ())

    def decide(self, command):
        action = None
        for rule in self.rules:
            if rule["permission"] in ("bash", "*") and fnmatch.fnmatchcase(
                command, rule["pattern"]
            ):
                action = rule["action"]
        return action

    def test_the_controls_still_hold(self):
        """A verdict identical across every input would indict the matcher."""
        self.assertEqual(self.decide(f"cat {self.root}/f"), "allow")
        self.assertEqual(self.decide(f"cat {self.root}/f > /tmp/x"), "deny")

    def test_sed_is_no_longer_pre_approved(self):
        """It was, and `sed -n '1w /tmp/x' <root>/f` resolved to allow."""
        self.assertNotIn("sed -n", self.pre.commands)
        self.assertNotEqual(
            self.decide(f"sed -n '1w /tmp/pwned.txt' {self.root}/f"), "allow"
        )

    def test_rg_and_git_are_no_longer_pre_approved_at_all(self):
        """Both were withdrawn: a textual flag guard is not a boundary, because
        shell quoting spells the same argument in unboundedly many ways. They
        reach the caller as approval requests now."""
        for command in (
            f"rg hello {self.root}/f",
            f"rg --pre /tmp/evil.sh hello {self.root}/f",
            f"rg --p''re /tmp/evil.sh hello {self.root}/f",
            f"rg -iz hello {self.root}/f",
            f"git -C {self.root} diff",
            f"git -C {self.root} diff --ext-diff a b",
            f'git -C {self.root} diff --out""put=x',
        ):
            self.assertEqual(self.decide(command), "ask", command)

    def test_the_quoted_spellings_that_defeated_the_guards_are_moot(self):
        """The guards they defeated are gone with their allows. Nothing here
        may resolve to allow; gating is the whole point."""
        for command in (
            f"cat --any''thing {self.root}/f > /tmp/x",
            f"grep --colo''r=always x {self.root}/f > /tmp/x",
        ):
            self.assertNotEqual(self.decide(command), "allow", command)


class TestStructuredStatusIsPinned(unittest.TestCase):
    def test_the_accepted_statuses_match_the_schema(self):
        schema = json.loads(SCHEMA.read_text())
        self.assertEqual(
            sorted(ask_opencode.STRUCTURED_STATUSES),
            sorted(schema["properties"]["status"]["enum"]),
        )


class TestInventedStatus(AskOpencodeCase):
    """The envelope is the model's claim, and two of its values are exit codes."""

    script = {"turns": [
        {"text": "a complete and correct answer", "status": "failed"},
        {"text": "the second answer", "status": "converged"},
    ]}

    def md(self, conv, turn):
        state = Path(self.env["ASK_OPENCODE_STATE_DIR"])
        return (state / conv / "turns" / f"{turn}.md").read_text()

    def test_a_status_the_schema_does_not_name_degrades_to_unknown(self):
        proc = self.start()
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(self.status_of(proc), "unknown")
        self.assertIn("a complete and correct answer", self.md(self.conv_of(proc), 1))

    def test_a_status_the_schema_does_name_still_decides(self):
        conv = self.conv_of(self.start())
        second = self.run_ask("send", conv, stdin="next")
        self.assertEqual(self.status_of(second), "converged")


class TestRejectionReasonIsRecorded(AskOpencodeCase):
    script = {"turns": [{"gates": [PERMISSION_GATE], "text": "done"}]}

    def record(self, conv, turn):
        state = Path(self.env["ASK_OPENCODE_STATE_DIR"])
        return json.loads((state / conv / "turns" / f"{turn}.json").read_text())

    def test_the_reason_reaches_the_resumed_turns_record_and_page(self):
        conv = self.conv_of(self.start())
        self.run_ask("reject", conv, "--reason", "it reaches outside the repo")
        self.assertEqual(self.record(conv, 2)["reason"], "it reaches outside the repo")
        state = Path(self.env["ASK_OPENCODE_STATE_DIR"])
        page = (state / conv / "turns" / "2.md").read_text()
        self.assertIn("it reaches outside the repo", page)
        self.assertIn("never sent to opencode", page)

    def test_a_turn_answered_without_a_reason_records_none(self):
        conv = self.conv_of(self.start())
        self.run_ask("approve", conv)
        self.assertIsNone(self.record(conv, 2)["reason"])


class TestMultiQuestionGate(AskOpencodeCase):
    script = {"turns": [{"gates": [TWO_QUESTION_GATE], "text": "done"}]}

    def replies(self):
        return [e for e in self.requests()
                if e["method"] == "POST" and e["path"].endswith("/reply")]

    def test_each_question_gets_its_own_answer(self):
        conv = self.conv_of(self.start())
        proc = self.run_ask("answer", conv, "postgres", "node")
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(self.replies()[0]["body"],
                         {"answers": [["postgres"], ["node"]]})

    def test_one_label_for_two_questions_is_refused_before_anything_is_sent(self):
        conv = self.conv_of(self.start())
        proc = self.run_ask("answer", conv, "postgres")
        self.assertEqual(proc.returncode, 4)
        self.assertIn("one label each", proc.stderr)
        self.assertEqual(self.replies(), [])

    def test_a_label_the_question_does_not_offer_is_refused(self):
        conv = self.conv_of(self.start())
        proc = self.run_ask("answer", conv, "postgres", "deno")
        self.assertEqual(proc.returncode, 4)
        self.assertIn("deno", proc.stderr)
        self.assertIn("node", proc.stderr)
        self.assertEqual(self.replies(), [])


class TestReattachedTurnDuration(AskOpencodeCase):
    """`stop --if-idle` derives a turn's end from its start plus this number."""

    script = {"turns": [{"text": "slow answer", "delay_s": 2}]}

    def test_a_turn_resumed_by_wait_reports_the_whole_turn(self):
        first = self.start(extra=("--wait", "1"))
        self.assertEqual(first.returncode, 10)
        conv = self.conv_of(first)
        second = self.run_ask("wait", conv, "--emit", "summary")
        self.assertEqual(second.returncode, 0)
        state = Path(self.env["ASK_OPENCODE_STATE_DIR"])
        record = json.loads((state / conv / "turns" / "1.json").read_text())
        # The reattaching leg alone is ~1s; only the whole turn clears this.
        self.assertGreaterEqual(record["duration_s"], 1.8)


class TestConversationIds(AskOpencodeCase):
    """The tail was the millisecond clock modulo 0x1000000, which repeats every
    4h39m, and `start` wrote a fresh meta over whatever it landed on."""

    script = {"turns": [{"text": "answer"}]}

    def args(self, name):
        return argparse.Namespace(
            repo=str(self.repo), write=False, model=None, variant=None, agent=None,
            schema="none", name=name, read_root=None, no_preapproved=True,
            wait=10, emit="summary",
        )

    def run_start(self, conv, name="taken"):
        state = Path(self.env["ASK_OPENCODE_STATE_DIR"])
        with mock.patch.dict(
            os.environ, {"ASK_OPENCODE_URL": self.env["ASK_OPENCODE_URL"]}
        ), mock.patch.object(ask_opencode, "STATE_DIR", state), mock.patch.object(
            # In-process, so the module constants have to be isolated the way
            # the subprocess env does it: the machine's own opencode state would
            # otherwise resolve a model this fake server does not have.
            ask_opencode, "MODEL_STATE", self.tmp / "absent-model-state.json"
        ), mock.patch.object(
            ask_opencode, "mint_conv_id", return_value=conv
        ), mock.patch.object(sys, "stdin", io.StringIO("do the thing")), \
                contextlib.redirect_stdout(io.StringIO()):
            return ask_opencode.cmd_start(self.args(name))

    def test_two_ids_minted_in_the_same_millisecond_differ(self):
        with mock.patch.object(ask_opencode.time, "time", return_value=1.7e9):
            first = ask_opencode.mint_conv_id("audit", "task")
            second = ask_opencode.mint_conv_id("audit", "task")
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("audit-"))

    def test_start_refuses_an_id_that_already_names_a_conversation(self):
        state = Path(self.env["ASK_OPENCODE_STATE_DIR"])
        (state / "taken-abc123" / "turns").mkdir(parents=True)
        (state / "taken-abc123" / "meta.json").write_text('{"conv": "taken-abc123"}')
        with self.assertRaises(ask_opencode.UsageError) as caught:
            self.run_start("taken-abc123")
        self.assertIn("already exists", str(caught.exception))
        # The conversation it landed on is untouched.
        self.assertEqual(
            json.loads((state / "taken-abc123" / "meta.json").read_text()),
            {"conv": "taken-abc123"},
        )

    def test_a_fresh_id_still_starts(self):
        self.assertEqual(self.run_start("fresh-abc123"), 0)


class TestModelPayloadShapes(AskOpencodeCase):
    """The two endpoints take different model objects; the fake enforces both."""

    script = {"turns": [{"text": "answer"}]}

    def test_session_create_and_prompt_carry_the_shape_each_endpoint_declares(self):
        proc = self.start(extra=("--model", "test-provider/model-a"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(
            self.session_create_body()["model"],
            {"providerID": "test-provider", "id": "model-a"},
        )
        prompt = next(e for e in self.requests()
                      if e["path"].endswith("/prompt_async"))
        self.assertEqual(
            prompt["body"]["model"],
            {"providerID": "test-provider", "modelID": "model-a"},
        )


class TestModelsCommand(AskOpencodeCase):
    """--model is only usable if the caller can learn a legal value."""

    script = {
        "turns": [],
        "config": {"model": "test-provider/model-a"},
        "providers": {"providers": [{"id": "test-provider", "models": {
            "model-a": {"variants": {"max": {}}},
            "model-b": {"variants": {"high": {}, "medium": {}}},
            "plain-model": {},
        }}]},
    }

    def test_it_lists_every_model_with_its_variants(self):
        proc = self.run_ask("models", "--repo", str(self.repo))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("test-provider/model-b", proc.stdout)
        self.assertIn("high, medium", proc.stdout)
        self.assertIn("none listed", proc.stdout)

    def test_it_marks_the_model_that_answers_without_the_flag(self):
        proc = self.run_ask("models", "--repo", str(self.repo))
        marked = [line for line in proc.stdout.splitlines()
                  if "answers without --model" in line]
        self.assertEqual(len(marked), 1)
        self.assertIn("test-provider/model-a", marked[0])

    def test_it_says_where_the_default_came_from(self):
        proc = self.run_ask("models", "--repo", str(self.repo))
        self.assertIn("the opencode config for this directory", proc.stdout)


class TestTransientApiFailure(AskOpencodeCase):
    """A turn belongs to the server; one dropped request is not its end."""

    script = {"turns": [{"text": "answer"}]}
    server_env = {"FAKE_OPENCODE_FLAKY": "3"}

    def test_a_blip_does_not_end_the_wait(self):
        proc = self.start()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.status_of(proc), "converged")

    def test_a_failure_that_persists_still_surfaces(self):
        proc = self.start(env={"ASK_OPENCODE_POLL_FAIL_S": "0"})
        self.assertEqual(proc.returncode, 4)


class TestFailureBodyFencing(AskOpencodeCase):
    """A failed turn quotes the raw body, which may hold fences of its own."""

    script = {"turns": [{
        "unstructured": True,
        "text": 'before\n```json\n{"a": 1}\n```\nafter',
        "error": {"name": "ProviderError", "data": {"message": "upstream refused"}},
    }]}

    def test_the_quote_is_longer_than_anything_inside_it(self):
        proc = self.start()
        self.assertEqual(proc.returncode, 2)
        state = Path(self.env["ASK_OPENCODE_STATE_DIR"])
        page = (state / self.conv_of(proc) / "turns" / "1.md").read_text()
        self.assertIn("upstream refused", page)
        self.assertIn("````", page)
        # The body survives whole rather than being cut off at its own fence.
        self.assertIn("after", page.split("````")[1])



# Every class below pins ASK_OPENCODE_POLL_MAX_S to an hour, so the safety poll
# cannot be what finishes the turn. Whatever lands, landed because of the stream.
STREAM_ONLY = {"ASK_OPENCODE_POLL_MAX_S": "3600", "ASK_OPENCODE_POLL_S": "0.05"}


class TestEventDrivenCompletion(AskOpencodeCase):
    script = {"turns": [{"text": "the answer", "delay_s": 1}]}

    def test_a_turn_completes_on_the_stream(self):
        started = time.monotonic()
        proc = self.start(env=STREAM_ONLY)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.status_of(proc), "converged")
        # The reply lands a second in; the safety poll is an hour away.
        self.assertLess(time.monotonic() - started, 10)


class TestEventDrivenGate(AskOpencodeCase):
    """The gate is raised a second in, so it is not already pending when the
    driver first looks — otherwise the first fetch finds it and the stream is
    not what is under test."""

    script = {"turns": [
        {"gates": [PERMISSION_GATE], "text": "done", "gate_delay_s": 1},
    ]}

    def test_a_gate_arrives_on_the_stream(self):
        started = time.monotonic()
        proc = self.start(env=STREAM_ONLY)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.status_of(proc), "needs_permission")
        self.assertLess(time.monotonic() - started, 10)


class TestSafetyPollCarriesASilentStream(AskOpencodeCase):
    """A stream that connects and then says nothing must not strand the turn."""

    script = {"turns": [{"text": "the answer", "delay_s": 1}]}
    server_env = {"FAKE_OPENCODE_EVENT_SILENT": "1"}

    def test_the_turn_still_lands_and_waits_for_the_poll_to_do_it(self):
        started = time.monotonic()
        proc = self.start(env={"ASK_OPENCODE_POLL_MAX_S": "2"})
        elapsed = time.monotonic() - started
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.status_of(proc), "converged")
        # Nothing woke it, so it cannot have finished before the ceiling.
        self.assertGreaterEqual(elapsed, 1.5)


class TestServerWithoutTheEndpoint(AskOpencodeCase):
    """A server too old to stream degrades to the cadence this driver had
    before the stream existed, not to the safety-poll ceiling."""

    script = {"turns": [{"text": "the answer", "delay_s": 1}]}
    server_env = {"FAKE_OPENCODE_NO_EVENT": "1"}

    def test_a_404_on_the_stream_falls_back_to_polling_at_the_floor(self):
        started = time.monotonic()
        proc = self.start(env=STREAM_ONLY)
        elapsed = time.monotonic() - started
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.status_of(proc), "converged")
        # The ceiling is an hour away, so only the POLL_S floor can explain this.
        self.assertLess(elapsed, 10)


class TestDroppedStreamIsRecovered(AskOpencodeCase):
    """An SSE stream replays nothing, so a reconnect must re-read rather than
    assume it missed nothing."""

    # The server holds the reply until a subscriber has actually been dropped,
    # so the completion is inside the disconnected gap by construction rather
    # than by timing. Timing alone left a legal ordering — a late subscription
    # whose first event is the completion itself — in which the test passed with
    # no reconnect wake at all.
    script = {"turns": [{"text": "the answer"}]}
    server_env = {"FAKE_OPENCODE_EVENT_DROP": "1",
                  "FAKE_OPENCODE_COMPLETE_ON_DROP": "1"}

    def test_the_turn_lands_after_the_stream_dies_under_it(self):
        started = time.monotonic()
        proc = self.start(env=STREAM_ONLY)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.status_of(proc), "converged")
        # Only the reconnect can explain this: the ceiling is an hour out, and
        # the reply grace stops applying the moment the message exists.
        self.assertLess(time.monotonic() - started, 10)
        subscriptions = [e for e in self.requests()
                         if e["method"] == "GET" and e["path"] == "/event"]
        self.assertGreaterEqual(len(subscriptions), 2, "no reconnect happened")


class TestColdStartAfterDetachedCompletion(AskOpencodeCase):
    """The turn finishes while no driver is attached at all — the case the
    ladder in SKILL.md creates every time a caller is killed or times out.

    Nothing that happened while nobody was subscribed is replayed, so no event
    can announce this reply to the new process. It is found by a fetch the loop
    performs without being asked to — the one at entry, or the one the first
    connect wakes — which is what makes reattachment work at all.
    """

    script = {"turns": [{"text": "the answer", "delay_s": 3}]}

    def test_wait_delivers_a_reply_that_landed_while_nothing_was_attached(self):
        first = self.start(extra=("--wait", "0.3"), env=STREAM_ONLY)
        self.assertEqual(first.returncode, 10)
        conv = self.conv_of(first)
        time.sleep(4)
        started = time.monotonic()
        second = self.run_ask(
            "wait", conv, "--emit", "summary", env=STREAM_ONLY
        )
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(self.status_of(second), "converged")
        self.assertLess(time.monotonic() - started, 10)



class TestReplyState(unittest.TestCase):
    """The terminal decision, as its own table.

    opencode completes every intermediate step of a tool-using turn: in the
    local store, 196 assistant messages carry `finish: "tool-calls"` with a
    completion timestamp against 61 carrying `stop`.
    """

    def state(self, **info):
        base = {"id": "m", "role": "assistant", "time": {"created": 1, "completed": 2}}
        base.update(info)
        return ask_opencode.reply_state({"info": base, "parts": []})

    def test_an_unfinished_message_is_running(self):
        self.assertEqual(self.state(time={"created": 1}), ask_opencode.REPLY_RUNNING)

    def test_a_completed_tool_step_is_running_not_terminal(self):
        self.assertEqual(self.state(finish="tool-calls"), ask_opencode.REPLY_RUNNING)

    def test_a_completed_stop_is_terminal(self):
        self.assertEqual(self.state(finish="stop"), ask_opencode.REPLY_TERMINAL)

    def test_an_error_is_terminal_however_it_finished(self):
        self.assertEqual(
            self.state(finish="tool-calls", error={"name": "APIError"}),
            ask_opencode.REPLY_TERMINAL,
        )

    def test_an_unknown_or_absent_finish_is_unresolved(self):
        """Positive predicate: only values known to end a turn end one. Reading
        an unknown continuation marker as terminal would publish a partial
        answer and close the turn, which nothing recovers."""
        self.assertEqual(self.state(finish="something-new"), ask_opencode.REPLY_UNRESOLVED)
        self.assertEqual(self.state(), ask_opencode.REPLY_UNRESOLVED)


class TestIntermediateToolStep(AskOpencodeCase):
    """The window between one step completing and the next being created.

    The prompt's newest child is a completed `tool-calls` message for two
    seconds before the real answer exists, and the stream announces it — which
    is exactly when an event-driven driver looks.
    """

    script = {"turns": [{
        "tool_steps": 1,
        "delay_s": 2,
        "text": "the real answer",
    }]}

    def test_the_answer_is_the_final_message_not_the_step(self):
        proc = self.start(env={"ASK_OPENCODE_POLL_MAX_S": "3600"})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.status_of(proc), "converged")
        state = Path(self.env["ASK_OPENCODE_STATE_DIR"])
        page = (state / self.conv_of(proc) / "turns" / "1.md").read_text()
        self.assertIn("the real answer", page)
        self.assertNotIn("calling a tool", page)

    def test_the_turn_is_not_closed_while_only_the_step_exists(self):
        """A short budget lands inside the window, and must report the turn as
        still running rather than publishing the step."""
        proc = self.start(extra=("--wait", "0.5"), env={"ASK_OPENCODE_POLL_MAX_S": "3600"})
        self.assertEqual(proc.returncode, 10)
        self.assertIn("running", proc.stdout)
        meta = json.loads(
            (Path(self.env["ASK_OPENCODE_STATE_DIR"]) / self.conv_of(proc)
             / "meta.json").read_text()
        )
        self.assertEqual(meta["current"], 1)


class TestUnresolvedFinish(AskOpencodeCase):
    """A completed, error-free reply whose finish this tool does not know."""

    script = {"turns": [{"text": "a partial answer", "finish": "something-new"}]}

    def test_it_is_held_rather_than_published(self):
        proc = self.start(extra=("--wait", "1"), env={"ASK_OPENCODE_POLL_MAX_S": "3600"})
        self.assertEqual(proc.returncode, 10)
        # Named, so a change in opencode's vocabulary is diagnosable in one line
        # instead of looking like a slow turn.
        self.assertIn("unresolved finish 'something-new'", proc.stdout)

    def test_the_turn_stays_recoverable(self):
        conv = self.conv_of(self.start(extra=("--wait", "1"),
                                       env={"ASK_OPENCODE_POLL_MAX_S": "3600"}))
        meta = json.loads(
            (Path(self.env["ASK_OPENCODE_STATE_DIR"]) / conv / "meta.json").read_text()
        )
        self.assertEqual(meta["current"], 1)
        self.assertIsInstance(meta["current_mid"], str)



class FetchCountingCase(AskOpencodeCase):
    def fetches(self):
        return [e for e in self.requests()
                if e["method"] == "GET" and e["path"].endswith("/message")]


class TestCompactionCadence(FetchCountingCase):
    """The wait must not pin at its floor for the length of a compaction.

    The reply-grace deadline cannot fire while a compaction is in flight — the
    interpretation branch suppresses it — so leaving it in the sleep calculation
    only made `min()` negative, and the loop then re-read the whole message
    history every POLL_S on the largest history this tool ever handles.
    """

    script = {"turns": [{"compaction": True, "delay_s": 3, "text": "the answer"}]}

    def test_a_long_compaction_does_not_re_read_the_history_at_the_floor(self):
        proc = self.start(env={
            "ASK_OPENCODE_REPLY_GRACE_S": "0.3",
            "ASK_OPENCODE_POLL_S": "0.05",
            "ASK_OPENCODE_POLL_MAX_S": "3600",
        })
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.status_of(proc), "converged")
        # Pinned at the floor this is roughly sixty fetches over three seconds.
        self.assertLess(len(self.fetches()), 15)


class TestEventStorm(FetchCountingCase):
    """A turn writing its answer updates its message continuously, so events
    arrive in bursts. The floor is what keeps a burst from becoming a burst of
    whole-history fetches."""

    script = {"turns": [{"text": "the answer", "delay_s": 3}]}
    server_env = {"FAKE_OPENCODE_EVENT_STORM": "120"}

    def test_a_stream_of_events_does_not_become_a_stream_of_fetches(self):
        proc = self.start(env={
            "ASK_OPENCODE_POLL_S": "0.5",
            "ASK_OPENCODE_POLL_MAX_S": "3600",
        })
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertLess(len(self.fetches()), 10)


class TestReadRootValidation(AskOpencodeCase):
    """A root is written into permission patterns verbatim, so a path that is
    itself a pattern is read twice: as a filename, then as a glob."""

    script = {"turns": [{"text": "the answer"}]}

    def test_a_root_that_is_itself_a_pattern_is_refused(self):
        star = self.tmp / "review*"
        star.mkdir()
        (self.tmp / "review-secret").mkdir()
        proc = self.start(extra=("--read-root", str(star)))
        self.assertEqual(proc.returncode, 4)
        self.assertIn("pattern character", proc.stderr)

    def test_the_filesystem_root_is_refused_rather_than_silently_dropped(self):
        """`root_forms` strips the trailing slash to an empty string and drops
        it, so `/` granted nothing while the turn header advertised it."""
        proc = self.start(extra=("--read-root", "/"))
        self.assertEqual(proc.returncode, 4)
        self.assertIn("not supported", proc.stderr)

    def test_an_ordinary_root_still_works(self):
        ordinary = self.tmp / "notes"
        ordinary.mkdir()
        proc = self.start(extra=("--read-root", str(ordinary)))
        self.assertEqual(proc.returncode, 0, proc.stderr)


class TestCancelledTurnDuration(AskOpencodeCase):
    """`stop --if-idle` derives a turn's end from its start plus its duration,
    so a cancellation that reports its own elapsed time instead of the turn's
    makes a long turn look as though it ended near its beginning."""

    script = {"turns": [{"hang": True}]}

    def test_a_cancelled_turn_reports_the_time_it_actually_ran(self):
        first = self.start(extra=("--wait", "1.5"))
        self.assertEqual(first.returncode, 10)
        conv = self.conv_of(first)
        cancelled = self.run_ask("cancel", conv)
        self.assertEqual(cancelled.returncode, 3)
        record = json.loads(
            (Path(self.env["ASK_OPENCODE_STATE_DIR"]) / conv / "turns" / "1.json").read_text()
        )
        # The cancellation itself takes milliseconds; only the turn's own span
        # clears this.
        self.assertGreaterEqual(record["duration_s"], 1.3)



class TestBlankError(unittest.TestCase):
    """An error is terminal however it is worded, including not at all.

    `message_error` formats; `has_error` detects. They were one function, and a
    blank `data.message` came back as the empty string — falsey — so a completed
    reply carrying a real error read as still running, and its record lost the
    failure it was reporting.
    """

    def message(self, error):
        return {"info": {"id": "m", "role": "assistant", "finish": "tool-calls",
                         "time": {"created": 1, "completed": 2}, "error": error},
                "parts": []}

    def test_an_error_with_a_blank_message_is_still_terminal(self):
        for blank in ({"name": "APIError", "data": {"message": ""}},
                      {"name": "APIError", "data": {"message": "   "}},
                      {"name": "APIError", "message": ""}):
            message = self.message(blank)
            self.assertTrue(ask_opencode.has_error(message), blank)
            self.assertEqual(ask_opencode.reply_state(message),
                             ask_opencode.REPLY_TERMINAL, blank)

    def test_the_reason_never_comes_back_empty(self):
        for blank in ({"name": "APIError", "data": {"message": ""}},
                      {"name": "", "data": {"message": "  "}}):
            reason = ask_opencode.message_error(self.message(blank))
            self.assertTrue(reason and reason.strip(), blank)

    def test_a_worded_error_still_reports_its_wording(self):
        reason = ask_opencode.message_error(
            self.message({"name": "APIError", "data": {"message": "upstream refused"}})
        )
        self.assertEqual(reason, "upstream refused")

    def test_no_error_is_no_error(self):
        message = {"info": {"id": "m", "role": "assistant", "finish": "stop",
                            "time": {"created": 1, "completed": 2}}, "parts": []}
        self.assertFalse(ask_opencode.has_error(message))
        self.assertIsNone(ask_opencode.message_error(message))


class TestBlankErrorTurn(AskOpencodeCase):
    script = {"turns": [{
        "text": "a partial answer",
        "finish": "tool-calls",
        "error": {"name": "APIError", "data": {"message": ""}},
    }]}

    def test_a_turn_whose_error_has_no_text_still_fails(self):
        proc = self.start(env={"ASK_OPENCODE_POLL_MAX_S": "3600"})
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(self.status_of(proc), "failed")
        page = (Path(self.env["ASK_OPENCODE_STATE_DIR"]) / self.conv_of(proc)
                / "turns" / "1.md").read_text()
        self.assertIn("APIError", page)


class TestStandingRootValidation(unittest.TestCase):
    """The CLI roots and the file's roots reach the same patterns, so they need
    the same gate. Rejecting the JSON string is not enough: a root written as an
    innocuous name can resolve through a symlink onto a pattern, and it is the
    resolved spelling that lands in the ruleset."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ask-opencode-roots-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def load(self, payload):
        path = self.tmp / "preapproved.json"
        path.write_text(json.dumps(payload))
        return policy.load_preapproved(path)

    def test_a_root_resolving_onto_a_pattern_is_refused(self):
        target = self.tmp / "review*"
        target.mkdir()
        (self.tmp / "review-secret").mkdir()
        link = self.tmp / "innocuous"
        link.symlink_to(target)
        with self.assertRaises(policy.PreapprovedError) as caught:
            self.load({"roots": [str(link)], "bash": ["cat"]})
        self.assertIn("pattern character", str(caught.exception))

    def test_the_filesystem_root_is_refused(self):
        with self.assertRaises(policy.PreapprovedError) as caught:
            self.load({"roots": ["/"], "bash": ["cat"]})
        self.assertIn("not supported", str(caught.exception))

    def test_an_ordinary_root_still_loads(self):
        ordinary = self.tmp / "notes"
        ordinary.mkdir()
        loaded = self.load({"roots": [str(ordinary)], "bash": ["cat"]})
        assert loaded is not None
        self.assertEqual(loaded.roots, (ordinary,))


class TestWithdrawalOutranksReview(unittest.TestCase):
    def test_a_withdrawn_head_is_refused_even_if_it_is_also_reviewed(self):
        """Withdrawal is checked first on purpose: putting the head back in the
        reviewed table must not be enough to re-admit it."""
        tmp = Path(tempfile.mkdtemp(prefix="ask-opencode-order-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        root = tmp / "tasks"
        root.mkdir()
        path = tmp / "preapproved.json"
        path.write_text(json.dumps({"roots": [str(root)], "bash": ["rg"]}))
        table = dict(policy.REVIEWED_HEADS, rg=())
        with mock.patch.object(policy, "REVIEWED_HEADS", table):
            with self.assertRaises(policy.PreapprovedError) as caught:
                policy.load_preapproved(path)
        self.assertIn("withdrawn", str(caught.exception))


class TestLegacyGrantIsStillDisclosed(AskOpencodeCase):
    """A conversation started before git reads were withdrawn keeps the ruleset
    the server fixed at session creation. Its turn headers have to keep saying
    so — dropping the rendering with the feature would have left those sessions
    running git without approval and no longer disclosing it."""

    script = {"turns": [{"text": "the answer"}]}

    def test_a_stored_git_grant_is_named_as_a_standing_one(self):
        note = ask_opencode.preapproved_note({"preapproved": {
            "commands": ["cat"],
            "anywhere": [],
            "version_probes": [],
            "git_read": ["log", "diff"],
            "roots": ["/tmp/notes"],
            "read_roots": [],
            "unanchored": False,
        }})
        self.assertIn("git `log`, `diff`", note)
        self.assertIn("withdrawn", note)

    def test_a_summary_that_holds_only_the_legacy_grant_still_renders(self):
        note = ask_opencode.preapproved_note({"preapproved": {
            "commands": [], "anywhere": [], "version_probes": [],
            "git_read": ["diff"], "roots": ["/tmp/notes"], "read_roots": [],
            "unanchored": False,
        }})
        self.assertNotEqual(note, "")
        self.assertIn("diff", note)

    def test_a_modern_summary_names_no_git(self):
        note = ask_opencode.preapproved_note({"preapproved": {
            "commands": ["cat"], "anywhere": [], "version_probes": [],
            "roots": ["/tmp/notes"], "read_roots": [], "unanchored": False,
        }})
        self.assertNotIn("git", note)


class TestToolStepsWithoutDelay(AskOpencodeCase):
    """The fake has to finish the message that is still open. Completing the
    first child of the prompt re-stamped the intermediate step as the turn's
    `stop` and left the real answer unfinished for good."""

    script = {"turns": [{"tool_steps": 2, "text": "the real answer"}]}

    def test_the_answer_is_published_and_the_step_is_not(self):
        proc = self.start(env={"ASK_OPENCODE_POLL_MAX_S": "3600"})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.status_of(proc), "converged")
        page = (Path(self.env["ASK_OPENCODE_STATE_DIR"]) / self.conv_of(proc)
                / "turns" / "1.md").read_text()
        self.assertIn("the real answer", page)
        self.assertNotIn("calling a tool", page)


class TestGateAfterToolSteps(AskOpencodeCase):
    """A gate reached after an intermediate step is still a gate.

    The deferred answer used to be completed unconditionally, so a fixture
    combining tool steps with a gate silently exercised plain success."""

    script = {"turns": [
        {"tool_steps": 1, "delay_s": 1, "gates": [PERMISSION_GATE], "text": "done"},
    ]}

    def test_the_turn_stops_at_the_gate(self):
        proc = self.start(env={"ASK_OPENCODE_POLL_MAX_S": "3600"})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.status_of(proc), "needs_permission")

    def test_the_answer_arrives_after_the_approval(self):
        """The rest of the path, since a gate is only useful if answering it
        resumes the turn.

        What this cannot show is the answer being *held*, because `_await_turn`
        checks the gate queues before the reply: a fixture that raised the gate
        and completed the answer looks identical from out here. That is a limit
        of observing through the driver, not of testing — an assertion on the
        message's own `time.completed`, read from the fake, would separate them.
        It is a fixture invariant rather than a driver one, and no test here
        claims it."""
        conv = self.conv_of(self.start(env={"ASK_OPENCODE_POLL_MAX_S": "3600"}))
        resumed = self.run_ask("approve", conv, "--emit", "summary",
                               env={"ASK_OPENCODE_POLL_MAX_S": "3600"})
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertEqual(self.status_of(resumed), "converged")
        answer = (Path(self.env["ASK_OPENCODE_STATE_DIR"]) / conv / "turns"
                  / "2.md").read_text()
        self.assertIn("done", answer)


class TestHangAfterToolSteps(AskOpencodeCase):
    """Nor does a hanging turn finish itself once it has taken a step."""

    script = {"turns": [{"tool_steps": 1, "delay_s": 1, "hang": True}]}

    def test_the_turn_stays_open(self):
        proc = self.start(extra=("--wait", "3"),
                          env={"ASK_OPENCODE_POLL_MAX_S": "3600"})
        self.assertEqual(proc.returncode, 10)
        self.assertIn("running", proc.stdout)


if __name__ == "__main__":
    unittest.main()
