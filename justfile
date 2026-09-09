# Everything that has to be green before a change is done. `just check` is the gate.
#
# The gate is offline: it starts a fake server, never the real opencode binary, so it makes
# no model calls and costs nothing. Exercising the real thing is a manual step, kept out
# of every recipe here on purpose — a lane that spends money should not be reachable by
# running the default target.
#
# Test flags are not repeated here. They live in `pyproject.toml`, so a bare
# `uv run pytest` and this gate cannot drift apart.

# Recipe arguments reach the shell as positional parameters, so `"$@"` below keeps each
# one whole. Interpolating the argument list as text splits it on whitespace instead,
# which turns `just test -k 'a or b'` into three arguments, two of which pytest reads as
# paths.
set positional-arguments

default: check

# Format, lint, type-check, test.
check: fmt lint types test

# Applied rather than checked, and `lint` re-reads the result: an autofix that introduces
# a new violation has to fail the gate rather than ride along in the commit.

# Apply formatting and autofixes.
fmt:
    @printf '\n\033[1m==> ruff check --fix\033[0m\n'
    uv run ruff check . --fix
    @printf '\n\033[1m==> ruff format\033[0m\n'
    uv run ruff format .

# Lint the formatted tree.
lint:
    @printf '\n\033[1m==> ruff check\033[0m\n'
    uv run ruff check .

# Type-check under all three configurations, concurrently.
types:
    #!/usr/bin/env bash
    # Three runs, not one. `pyproject.toml` holds `standard` over the whole tree so an
    # editor stays usable; `config/pyright/strict.json` holds `strict` over the driver
    # modules; and `config/pyright/tests.json` holds `strict` over the suite with
    # `reportPrivateUsage` off, because these tests reach for internals by name on
    # purpose and the alternative is widening the driver's API to satisfy a linter.
    #
    # One configuration cannot do the first two: `executionEnvironments` has no
    # `typeCheckingMode`, so writing one there is ignored and still exits 0.
    #
    # Concurrent because they share nothing but the source they read. Each keeps its own
    # output so a failure is still readable in order.
    set -uo pipefail
    printf '\n\033[1m==> pyright (standard tree, strict driver, strict tests)\033[0m\n'
    tree=$(mktemp) && driver=$(mktemp) && suite=$(mktemp)
    uv run pyright >"$tree" 2>&1 &
    tree_pid=$!
    uv run pyright --project config/pyright/strict.json >"$driver" 2>&1 &
    driver_pid=$!
    uv run pyright --project config/pyright/tests.json >"$suite" 2>&1 &
    suite_pid=$!
    status=0
    wait $tree_pid || status=1
    wait $driver_pid || status=1
    wait $suite_pid || status=1
    for report in "$tree" "$driver" "$suite"; do
        cat "$report"
        rm -f "$report"
    done
    exit $status

# The offline suite.
test *ARGS:
    @printf '\n\033[1m==> pytest\033[0m\n'
    uv run pytest -q "$@"
