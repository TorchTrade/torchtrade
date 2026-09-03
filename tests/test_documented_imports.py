"""The docs are executed here, because prose cannot be verified by review.

Scope: `from torchtrade... import X` (flat AND parenthesized), plus a syntax check on the
python blocks in the in-package READMEs. Config kwargs are covered below; observation keys still are not.
"""

import ast
import builtins
import functools
import dataclasses
import importlib
import pathlib
import pkgutil
import re
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
# Parenthesized too: a flat-only pattern skipped 30 of 225 documented symbols, and the
# skipped set contained the exact phantoms the sweep that added this test had missed.
IMPORT_LINE = re.compile(r"^from (torchtrade[\w.]*) import (?:\(([^)]*)\)|([^\n(]+))$", re.M)
PY_BLOCK = re.compile(r"```python\n(.*?)```", re.S)


@functools.lru_cache(maxsize=None)
def _git_ls_files(pattern=None):
    """Tracked paths from the index, not the filesystem. Anyone's untracked local notes
    must not fail the suite, and an empty directory left behind by a rename is not
    something a reader can clone."""
    argv = ["git", "ls-files"] + ([pattern] if pattern else [])
    listing = subprocess.run(argv, cwd=REPO, capture_output=True, text=True, check=True)
    return tuple(line for line in listing.stdout.splitlines() if line.strip())


@functools.lru_cache(maxsize=None)
def _doc_sources():
    return tuple(REPO / line for line in _git_ls_files("*.md"))


def _documented_imports():
    for path in _doc_sources():
        for match in IMPORT_LINE.finditer(path.read_text()):
            module = match.group(1)
            for name in (n.strip() for n in (match.group(2) or match.group(3)).split(",")):
                if name.isidentifier():
                    yield pytest.param(module, name,
                                       id=f"{path.relative_to(REPO)}::{module}.{name}")


CASES = list(_documented_imports())
_README_SOURCES = [p for p in _doc_sources()
                   if p.name == "README.md" and p.is_relative_to(REPO / "torchtrade")]
# Read each README once. The guard needs a block's predecessors, so blocks are carried
# per-file and the flat list is derived from that rather than re-globbing.
_BLOCKS_BY_FILE = [(p, PY_BLOCK.findall(p.read_text())) for p in _README_SOURCES]
README_BLOCKS = [
    pytest.param(block, id=f"{p.relative_to(REPO)}::block{i}")
    for p, blocks in _BLOCKS_BY_FILE for i, block in enumerate(blocks)
]
README_BLOCKS_IN_CONTEXT = [
    pytest.param(blocks[:i], block, id=f"{p.relative_to(REPO)}::block{i}")
    for p, blocks in _BLOCKS_BY_FILE for i, block in enumerate(blocks)
]


@pytest.mark.parametrize("module,name", CASES)
def test_a_documented_import_resolves(module, name):
    assert hasattr(importlib.import_module(module), name), f"{module} has no {name!r}"


@pytest.mark.parametrize("block", README_BLOCKS)
def test_a_documented_code_block_parses(block):
    """Syntax only -- most blocks need credentials or data to run. Cheap, and it catches
    what review misses: deleting the line that opened a call, leaving its arguments.

    compile(), not ast.parse(): `break` outside a loop is legal to PARSE and illegal to
    COMPILE, and converting the gym-style `obs, reward, done, info = env.step(action)`
    loops to the TensorDict contract stranded four `break` statements at module level in
    live/README.md. An ast.parse() sweep declared all three files clean while they were
    not. Anything CPython refuses to compile, a reader cannot run."""
    compile(block, "<doc>", "exec")


def test_the_sweep_still_covers_every_source():
    """Floors set just under the real counts, because a generous one hides exactly what it
    is for: >140 against 190 still passed with the largest source (49) removed, and >20
    against 66 tolerated losing two thirds. Raise these when the docs grow."""
    assert len(CASES) > 185, f"only {len(CASES)} documented imports discovered"
    assert len(README_BLOCKS) > 60, f"only {len(README_BLOCKS)} code blocks discovered"


# ── Config kwargs ────────────────────────────────────────────────────────────
# The scope note above used to say config kwargs "live in prose and are not covered".
# They were not: `margin_call_threshold`, `rollout_steps` and `trading_mode` sat in the
# docs with ZERO occurrences in the package while all 257 import cases here passed
# (#287). An import check validates that a NAME resolves, not that a CALL is
# constructible -- the boundary was the import line, and every defect lived below it.
# Indentation-tolerant, and single-line calls too. A 4-space-only pattern silently
# skipped both flagship SequentialTradingEnvConfig examples -- they sit inside mkdocs
# `=== "Spot Trading"` tabs, so the whole fence is indented -- plus every Polymarket
# example and a live api_key bug in torchtrade/envs/live/README.md that this check's own
# premise says it should catch.
CONFIG_CALL = re.compile(r"\b(\w*Config)\(([^()]*(?:\([^()]*\)[^()]*)*)\)", re.S)
KWARG = re.compile(r"(?:^|[\s,(])([A-Za-z_]\w*)\s*=(?!=)", re.M)
# Comments are stripped first: a value comment like `# 1 = spot` parses as a kwarg
# named "1" otherwise, and an identifier cannot start with a digit anyway.
COMMENT = re.compile(r"#[^\n]*")
# Nested calls are blanked before kwargs are read: matching "at any depth" pulls the
# INNER call's kwargs out as if they belonged to the config, so
# `reward_function=partial(fn, alpha=0.5)` yields a phantom `alpha`. That either cries
# wolf or -- worse -- passes silently when the inner name collides with a real field.
# No doc does this today. Handles arbitrary nesting depth INSIDE a captured call;
# the depth CONFIG_CALL itself can capture is pinned separately below.
NESTED = re.compile(r"\([^()]*\)")


def _documented_config_kwargs():
    """Every (config class, kwarg) pair written in a tracked markdown code block."""
    for path in _doc_sources():
        for block in PY_BLOCK.findall(path.read_text()):
            for cls_name, body in CONFIG_CALL.findall(COMMENT.sub("", block)):
                flat = body
                while NESTED.search(flat):
                    flat = NESTED.sub("", flat)
                for kwarg in KWARG.findall(flat):
                    yield str(path.relative_to(REPO)), cls_name, kwarg


CONFIG_KWARGS = sorted(set(_documented_config_kwargs()))


def _resolve_config(cls_name):
    """The config class by name, or None when the docs name one we do not ship."""
    for module in ("torchtrade.envs.offline", "torchtrade.envs.live.alpaca",
                   "torchtrade.envs.live.binance", "torchtrade.envs.live.bitget",
                   "torchtrade.envs.live.bybit", "torchtrade.envs.live.okx",
                   "torchtrade.envs.live.polymarket"):
        try:
            found = getattr(importlib.import_module(module), cls_name, None)
        except Exception:
            continue
        if found is not None:
            return found
    return None


# The #287 ratchet reached empty and the scaffolding went with it: with no exemptions
# left, the bare assert below IS the ratchet, and it is strictly stronger than the
# xfail machinery it replaces. A newly broken kwarg fails immediately.


@pytest.mark.parametrize("source,cls_name,kwarg", CONFIG_KWARGS,
                         ids=[f"{c}.{k}" for _, c, k in CONFIG_KWARGS])
def test_a_documented_config_kwarg_exists(source, cls_name, kwarg):
    """A kwarg the class does not accept is a TypeError for anyone copying the block."""
    config_cls = _resolve_config(cls_name)
    if config_cls is None or not dataclasses.is_dataclass(config_cls):
        pytest.skip(f"{cls_name} is not a resolvable dataclass config")
    fields = {f.name for f in dataclasses.fields(config_cls)}
    assert kwarg in fields, (
        f"{source} documents {cls_name}({kwarg}=...), which raises TypeError -- "
        f"the class accepts {sorted(fields)}"
    )


CONFIG_OPEN = re.compile(r"\b\w*Config\(")


def test_every_documented_config_call_is_actually_captured():
    """A call the pattern cannot parse vanishes silently -- no failure, no xfail, nothing.

    CONFIG_CALL tolerates one level of nesting, so a doubly-nested callable
    (`Config(reward_function=partial(f, w=partial(g, x=1)), initial_cash=1)`) makes the
    WHOLE call unmatchable, taking its legitimate kwargs with it. That is a worse failure
    than the phantom kwarg it replaced, because it produces no signal at all.

    Counting opens against matches is the cheap invariant: it does not care how the
    pattern is written, only that every documented call reached the check.
    """
    missed = []
    for path in _doc_sources():
        for block in PY_BLOCK.findall(path.read_text()):
            block = COMMENT.sub("", block)
            opens = len(CONFIG_OPEN.findall(block))
            matched = len(CONFIG_CALL.findall(block))
            if opens != matched:
                missed.append(f"{path.relative_to(REPO)}: {opens} calls, {matched} parsed")
    assert not missed, (
        "documented config calls the pattern could not parse, so they are unchecked:\n"
        + "\n".join(missed)
    )


def test_the_kwarg_sweep_found_configs_to_check():
    """A regex that silently matched nothing would make the check above vacuous."""
    assert len(CONFIG_KWARGS) > 40, f"only {len(CONFIG_KWARGS)} documented kwargs found"


def test_no_readme_redefines_a_package_config_as_a_dataclass():
    """A `@dataclass class SomeConfig:` block shadowing a REAL config is two copies.

    Both rot independently, and the doc copy rots worse: written without annotations --
    `symbol = "BTCUSDT"` rather than `symbol: str = "BTCUSDT"` -- the decorator sees
    zero fields, so the class it builds rejects every kwarg the same page documents.
    Copying it raises TypeError on construction. compile() cannot see this; the block is
    valid Python that builds a useless class, which is why it survived three rounds.

    Only names that resolve to a real package config count. A `MyEnvConfig` in a
    subclassing example is the reader's own class, not a shadow.
    """
    offenders = []
    for path in _doc_sources():
        if path.name != "README.md" or not path.is_relative_to(REPO / "torchtrade"):
            continue
        for block in PY_BLOCK.findall(path.read_text()):
            for name in re.findall(r"@dataclass\s*\nclass\s+(\w+)\b", block):
                if _resolve_config(name) is not None:
                    offenders.append(f"{path.relative_to(REPO)}::{name}")
    assert not offenders, (
        f"{len(offenders)} README blocks redefine a real config as a dataclass instead "
        f"of importing it: {offenders}"
    )


@functools.lru_cache(maxsize=None)
def _package_names():
    """Every public name any torchtrade module exports, collected DETERMINISTICALLY.

    The first version scanned `sys.modules`, which made the verdict depend on which
    other tests had run first: selected alone it saw 71 torchtrade modules, after the
    import tests 102, and `ReplayObserver` -- a real exported class -- resolved False in
    the first case and True in the second. Identical file, opposite results, so the test
    failed on real API under `-k`, under xdist sharding, or under any reordering. An
    lru_cache on top froze whichever answer came first.

    Walking the package is slower once and correct always.
    """
    names = set()
    package = importlib.import_module("torchtrade")
    unwalkable = []
    # onerror, because walk_packages routes EVERY exception in a subpackage __init__
    # here when it is set -- with onerror=None an ImportError silently drops that whole
    # subtree and anything else escapes the generator. Either way the name set shrinks,
    # and a shrunken set makes real API read as a phantom: false failures, not false
    # passes. So a package that cannot be WALKED is fatal.
    #
    # A module that cannot be IMPORTED is not: vllm, chronos and openai are optional
    # extras, and every one of them is imported lazily inside a function today, so this
    # currently collects nothing extra -- but the day one moves to module scope, a dev
    # without the extra should not see doc tests fail. The import tests cover what must
    # import.
    for info in pkgutil.walk_packages(package.__path__, prefix="torchtrade.",
                                      onerror=unwalkable.append):
        try:
            module = importlib.import_module(info.name)
        except Exception:
            continue
        names.update(n for n in vars(module) if not n.startswith("_"))
    assert not unwalkable, (
        f"{len(unwalkable)} torchtrade subpackages could not be walked, so this guard "
        f"is checking against a truncated package: {unwalkable}"
    )
    return names


def _resolves_anywhere(name: str) -> bool:
    return name in _package_names()


def _names_bound_by(block, *, before_line=None):
    """Names bound at MODULE level in `block`, optionally only those defined earlier.

    Module level, and line-ordered, because a reader executes the fence top to bottom.
    Walking the whole AST accepted two things Python does not (#373 review):
    `phantom(); def phantom(): ...` -- a forward definition, which raises NameError when
    the fence is actually run -- and a `phantom` local inside an earlier helper, which
    masks a later top-level `phantom()`. Both let a real phantom through.
    """
    try:
        tree = ast.parse(block)
    except SyntaxError:
        return set()

    bound = set()
    for node in tree.body:  # module level only; a nested local binds nothing out here
        if before_line is not None and node.lineno >= before_line:
            break
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bound |= {a.asname or a.name.split(".")[0] for a in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        else:
            bound |= {n.id for n in ast.walk(node)
                      if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}

    return bound


def _params_of(node):
    a = node.args
    return {p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs,
                            *([a.vararg] if a.vararg else []),
                            *([a.kwarg] if a.kwarg else []))}


def _calls_with_scope(node, enclosing=frozenset()):
    """Yield (call_name, lineno, names-in-scope-from-enclosing-functions).

    Parameters are visible only INSIDE the function that declares them. Adding them to
    a block-wide set accepted `def helper(phantom): ...` followed by a top-level
    `phantom()` (#373 review), which raises NameError -- the guard's claim to check
    scope at the call site was false for exactly this case.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            inner = enclosing | _params_of(child) | {child.name}
            yield from _calls_with_scope(child, inner)
            continue
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
            yield child.func.id, child.lineno, enclosing
        yield from _calls_with_scope(child, enclosing)


# Objects a reader is expected to bring; naming one is not a claim about the package.
_READER_SUPPLIED = {"env", "config", "action", "agent", "df", "td", "transition", "obs",
                    "your_dataframe", "policy", "model", "data", "trainer"}


@pytest.mark.parametrize("earlier_blocks,block", README_BLOCKS_IN_CONTEXT)
def test_a_documented_block_calls_nothing_that_does_not_exist(earlier_blocks, block):
    """A CALL to a name the block never binds and the package never defines.

    This is the gap the import test cannot cover: it checks `from x import y` lines, so
    a block whose import line was corrected while its BODY kept calling the old name
    passes it. That is exactly what happened -- utils/README.md imported the real
    calculate_bracket_prices and then called calculate_sltp_prices and check_sltp_hit,
    neither of which has ever existed, three doc PRs in a row.

    Flags a called name that is not bound in the block, not bound by any earlier block
    in the same file, not a builtin, not reader-supplied, and exported nowhere in
    torchtrade. ATTRIBUTE calls (`self.foo()`, `obj.bar()`) are NOT covered -- that gap
    is how `_init_sltp` survived three rounds, and closing it needs the callee's type.
    """
    try:
        tree = ast.parse(block)
    except SyntaxError:
        pytest.skip("covered by the compile test")

    # A reader meets the blocks in order, so anything an EARLIER block defined is in
    # scope here -- that is what makes `MyEnv(...)` in a subclassing walkthrough legal
    # while `calculate_sltp_prices(...)`, defined nowhere at all, is not.
    bound = set()
    for earlier in earlier_blocks:
        # include_params=False: a parameter is local to its own block. Carrying them
        # forward let `def helper(calculate_sltp_prices)` in ANY earlier snippet
        # whitelist that phantom for the whole rest of the file -- and core/README.md
        # really does bind df/config/tensordict/self/kwargs that way in block 0.
        bound |= _names_bound_by(earlier)

    # Each call is judged against what is in scope AT ITS OWN LINE, so a definition
    # further down the fence does not retroactively legitimise a call above it.
    unknown = set()
    for name, lineno, in_scope in _calls_with_scope(tree):
        if name in _READER_SUPPLIED or hasattr(builtins, name):
            continue
        visible = bound | in_scope | _names_bound_by(block, before_line=lineno)
        if name in visible or _resolve_config(name) or _resolves_anywhere(name):
            continue
        unknown.add(name)
    assert not unknown, (
        f"calls names that exist nowhere in torchtrade: {sorted(unknown)}"
    )


# The three ways a doc names a repo file: backticked (`torchtrade/envs/core/live.py`),
# inside a github tree/blob URL, and as a markdown link target. All three have shipped a
# phantom. The prefix alternation is built from git rather than hardcoded, so it cannot
# claim a directory the repo does not have.
_TOP_LEVEL = "|".join(sorted({p.split("/")[0] for p in _git_ls_files() if "/" in p}))
BACKTICKED_PATH = re.compile(rf"`((?:{_TOP_LEVEL})/[\w./-]+)`")
GITHUB_URL = re.compile(
    r"https://github\.com/TorchTrade/torchtrade/(?:tree|blob)/main/([\w./-]+)",
    re.IGNORECASE,
)
RELATIVE_LINK = re.compile(r"\]\((?!https?:|#|mailto:)([^)\s#]+)(?:#[^)\s]*)?\)")


def _documented_paths():
    """Every repo path the docs name, as (source, repo-relative path) pairs.

    Link targets are relative to their own file, so they are resolved against it and then
    made repo-relative. The other two forms are already repo-relative.
    """
    seen = set()
    for path in _doc_sources():
        text = path.read_text()
        for pattern in (BACKTICKED_PATH, GITHUB_URL):
            for match in pattern.finditer(text):
                yield from _one(seen, path, match.group(1))
        for match in RELATIVE_LINK.finditer(text):
            target = (path.parent / match.group(1)).resolve()
            if target.is_relative_to(REPO):
                yield from _one(seen, path, str(target.relative_to(REPO)))


def _one(seen, source, raw):
    """One param per (source, path). A doc that names the same phantom twice is one fix,
    and CONFIG_KWARGS already dedupes for the same reason."""
    raw = raw.rstrip("/")
    key = (source, raw)
    if key not in seen:
        seen.add(key)
        yield pytest.param(raw, id=f"{source.relative_to(REPO)}::{raw}")


DOC_PATHS = list(_documented_paths())


@functools.lru_cache(maxsize=None)
def _tracked_paths():
    """Tracked files plus every directory on the way to one."""
    known = set()
    for line in _git_ls_files():
        path = pathlib.PurePosixPath(line)
        known.update(str(p) for p in (path, *path.parents))
    return known - {"."}


@pytest.mark.parametrize("documented_path", DOC_PATHS)
def test_a_documented_repo_path_is_tracked(documented_path):
    """Kills the mutation that moves a file and leaves the docs naming the old location.
    Nine phantoms shipped this way: offline/base.py, offline/sampler.py,
    examples/offline/iql (twice), a nonexistent offline/README.md linked from three
    READMEs, and a nonexistent docs/examples.md linked from two guides.

    Checked against the index, not the filesystem: the examples/offline/iql phantom still
    existed as an empty directory on the machine that documented it.

    Three forms are still unchecked, all of which have carried a phantom: the ASCII trees
    in the READMEs, bare filenames in backticks, and paths inside shell fences.
    """
    assert documented_path in _tracked_paths(), (
        f"documented path is not in the repo: {documented_path}"
    )


def test_each_path_form_still_finds_paths():
    """Floors just under the real counts. A single floor over the total does not work
    here: dropping the github-URL form entirely still left 31 of 65, which cleared a
    `> 25` guard while half the sweep was dead.

    A fully empty parametrize ERRORS on this pytest rather than passing, so partial
    shrinkage, not silence, is what this catches.
    """
    counts = {}
    for source in _doc_sources():
        text = source.read_text()
        for name, pattern in (("backtick", BACKTICKED_PATH), ("url", GITHUB_URL),
                              ("link", RELATIVE_LINK)):
            counts[name] = counts.get(name, 0) + len(pattern.findall(text))
    assert counts["backtick"] > 28, counts
    assert counts["url"] > 30, counts
    assert counts["link"] > 80, counts


# The class-hierarchy diagram in core/README.md, parsed by indentation: a class's parent
# is the nearest line above it that starts further left.
HIERARCHY_DOC = REPO / "torchtrade" / "envs" / "core" / "README.md"
TREE_BLOCK = re.compile(r"```\n(TorchTradeBaseEnv.*?)```", re.S)
TREE_LINE = re.compile(r"^([\s│├└─]*)([A-Z]\w+)")


def _documented_edges():
    block = TREE_BLOCK.search(HIERARCHY_DOC.read_text()).group(1)
    edges, open_at = [], {}
    for line in block.split("\n"):
        match = TREE_LINE.match(line)
        if not match:
            continue
        column, name = len(match.group(1)), match.group(2)
        outer = [c for c in open_at if c < column]
        if outer:
            edges.append(pytest.param(open_at[max(outer)], name,
                                      id=f"{open_at[max(outer)]}->{name}"))
        open_at = {c: n for c, n in open_at.items() if c < column}
        open_at[column] = name
    return edges


DOCUMENTED_EDGES = _documented_edges()


def _env_classes():
    """Name -> class, over the whole walked package. `_package_names` does the walking;
    without it `__subclasses__` sees only what other tests happened to import."""
    _package_names()
    found, stack = {}, [importlib.import_module("torchtrade.envs.core.base").TorchTradeBaseEnv]
    while stack:
        cls = stack.pop()
        if cls.__name__ not in found:
            found[cls.__name__] = cls
            stack.extend(cls.__subclasses__())
    return found


@pytest.mark.parametrize("parent,child", DOCUMENTED_EDGES)
def test_a_documented_hierarchy_edge_is_a_real_base(parent, child):
    """The diagram drew the offline envs as four siblings when they are a chain, and hung
    the four futures venues off TorchTradeLiveEnv when they route through
    TorchTradeFuturesLiveEnv.

    Direct bases, not issubclass: issubclass(OneStepTradingEnv, TorchTradeOfflineEnv) is
    True, so an issubclass check passes the sibling version unchanged and tests nothing.
    """
    classes = _env_classes()
    assert child in classes, f"diagram names a class that does not exist: {child}"
    bases = [b.__name__ for b in classes[child].__bases__]
    assert parent in bases, f"{child} is drawn under {parent}, real bases are {bases}"


def test_every_intermediate_env_class_is_in_the_diagram():
    """A venue added without updating the diagram is how it drifted the first time. Only
    intermediates are required: the concrete leaves are elided deliberately, which the
    diagram says in words.
    """
    classes = _env_classes()
    drawn = {n for e in DOCUMENTED_EDGES for n in e.values} | {"TorchTradeBaseEnv"}
    intermediate = {n for n, c in classes.items() if c.__subclasses__()}
    assert not intermediate - drawn, f"missing from the diagram: {sorted(intermediate - drawn)}"


@pytest.mark.parametrize("name", ["VectorizedSequentialTradingEnv",
                                  "VectorizedSequentialTradingEnvSLTP",
                                  "PolymarketBetEnv"])
def test_the_envs_documented_as_outside_the_hierarchy_really_are(name):
    """core/README.md tells the reader these do not inherit the shared bases, so a change
    to those bases must be applied by hand. If someone reparents one, that warning turns
    into a lie about money-moving code and nothing else would notice.
    """
    _package_names()
    base = importlib.import_module("torchtrade.envs.core.base").TorchTradeBaseEnv
    cls = _env_classes().get(name) or getattr(
        importlib.import_module("torchtrade.envs"), name, None)
    if cls is None:
        for mod in ["torchtrade.envs.offline.vectorized_sequential",
                    "torchtrade.envs.offline.vectorized_sequential_sltp",
                    "torchtrade.envs.live.polymarket.env"]:
            cls = cls or getattr(importlib.import_module(mod), name, None)
    assert cls is not None, f"{name} does not exist"
    assert not issubclass(cls, base), f"{name} now inherits TorchTradeBaseEnv"
