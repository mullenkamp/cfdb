"""Inventory of the docs' code: nothing in README.md or docs/ may drop out of the docs tests unseen.

Sybil silently ignores what it does not parse (a fence labelled ``python title="x"``, ``py`` or
``Python``, a directive with a typo), and it reports a skip without a reason as PASSED. So this test
compares Sybil's own parse of every docs file against an independent line scan, and checks the
skips against the committed manifest ``docs_skips.txt``. It runs in the unit suite (no docs code is
executed here).
"""

import ast
import collections
import doctest
import pathlib
import re

from sybil.evaluators.skip import Skipper

from cfdb.tests import docs_support

ROOT = docs_support.ROOT
MANIFEST = pathlib.Path(__file__).with_name('docs_skips.txt')
ALLOWED_INFO = {'python', 'bash', 'mermaid', 'text'}
FENCE_RE = re.compile(r'^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})(?P<info>.*)$')
# Anything that looks like a Sybil directive, however misspelt.
DIRECTIVE_RE = re.compile(r'<!--+\s*(invisible-code-block|code-block|skip|clear-namespace)', re.IGNORECASE)
VALUE_COMMENT_SPLITS = (' — ', ' (', ' <-', ' - ', ',  ')
CALLOUT_RE = re.compile(r'^\s*\(\d+\)')  # mkdocs-material annotation markers: '# (1)'
ALLOWED_DOCTEST_OPTIONS = {doctest.ELLIPSIS, doctest.NORMALIZE_WHITESPACE}


def _scan(path):
    """Independent line scan: fences (with their raw info string and body), directive comments,
    and whether each line is inside an HTML comment."""
    lines = path.read_text(encoding='utf-8').split('\n')
    fences, directives = [], []
    in_comment = False
    i = 0
    while i < len(lines):
        line = lines[i]
        m = FENCE_RE.match(line)
        if m and not in_comment:
            fence, start = m['fence'], i
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith(fence):
                j += 1
            fences.append(
                {
                    'line': start + 1,
                    'indent': m['indent'],
                    'info': m['info'],
                    'body': lines[start + 1 : j],
                    'in_comment': False,
                }
            )
            i = j + 1
            continue
        if m and in_comment:
            fences.append({'line': i + 1, 'indent': m['indent'], 'info': m['info'], 'body': [], 'in_comment': True})
        if DIRECTIVE_RE.search(line):
            directives.append({'line': i + 1, 'text': line, 'column0': line.startswith('<!--')})
        if '<!--' in line:
            in_comment = '-->' not in line.split('<!--')[-1]
        elif in_comment and '-->' in line:
            in_comment = False
        i += 1
    return fences, directives


def _sybil_regions(path):
    document = docs_support.make_sybil().parse(path)
    out = []
    for example in document.examples():
        region = example.region
        if isinstance(region.evaluator, Skipper):
            kind = 'skip'
        elif isinstance(region.parsed, str):
            kind = 'code'
        elif 'clear' in type(region.evaluator).__name__.lower():
            kind = 'clear'
        else:
            kind = 'doctest'
        out.append({'line': example.line, 'kind': kind, 'parsed': region.parsed})
    return out


def _value_comment(comment):
    text = comment.strip()
    if CALLOUT_RE.match(text):
        return False
    if text.startswith('->'):
        text = text[2:].strip()
    candidates = [text] + [text.split(s)[0] for s in VALUE_COMMENT_SPLITS if s in text]
    for c in candidates:
        try:
            ast.literal_eval(c.strip())
            return True
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
    return False


def _shown_values(body):
    """Statements in a plain block, at any depth, whose trailing (or next-line) comment states a value."""
    src = '\n'.join(body)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Expr, ast.Assign, ast.AugAssign)):
            continue
        last = body[node.end_lineno - 1]
        comment = last.split('#', 1)[1] if '#' in last and not last.lstrip().startswith('#') else ''
        if not comment and node.end_lineno < len(body) and body[node.end_lineno].lstrip().startswith('#'):
            comment = body[node.end_lineno].lstrip()[1:]
        if comment and _value_comment(comment):
            found.append(body[node.lineno - 1].strip())
    return found


def _doctest_problems(body):
    """A >>> block as stdlib doctest parses it: text after a blank line ends the expected output and is
    then never run, and options other than ELLIPSIS / NORMALIZE_WHITESPACE (e.g. +SKIP) change what
    is checked without the skip manifest seeing it."""
    problems = []
    for part in doctest.DocTestParser().parse('\n'.join(body)):
        if isinstance(part, str):
            if part.strip():
                problems.append(f'text that never runs (after a blank line in a >>> block): {part.strip()[:60]!r}')
        elif set(part.options) - ALLOWED_DOCTEST_OPTIONS:
            problems.append(f'doctest option other than ELLIPSIS/NORMALIZE_WHITESPACE on {part.source.strip()[:60]!r}')
    return problems


def _inventory():
    problems, skips, summary = [], [], []
    for path in docs_support.doc_files():
        rel = str(path.relative_to(ROOT))
        fences, directives = _scan(path)
        regions = _sybil_regions(path)

        for f in fences:
            where = f'{rel}:{f["line"]}'
            if f['in_comment']:
                problems.append(f'{where}: a fence inside an HTML comment (Sybil runs it; the site hides it)')
                continue
            if f['info'] not in ALLOWED_INFO:
                problems.append(
                    f'{where}: fence info string {f["info"]!r} is not exactly one of {sorted(ALLOWED_INFO)}'
                )

        # Every python fence and every directive is claimed by a Sybil region, and vice versa.
        expected = set()
        for f in fences:
            if f['info'] == 'python' and not f['in_comment']:
                if f['body'] and f['body'][0].lstrip().startswith('>>>'):
                    # one region per prompt; a comment-only prompt is not an example
                    expected |= {
                        f['line'] + 1 + k
                        for k, ln in enumerate(f['body'])
                        if ln.lstrip().startswith('>>> ') and not ln.lstrip()[4:].lstrip().startswith('#')
                    }
                else:
                    expected.add(f['line'])
        expected |= {d['line'] for d in directives}
        claimed = {r['line'] for r in regions}
        for ln in sorted(expected - claimed):
            problems.append(f'{rel}:{ln}: code or directive that Sybil does not parse (so it never runs)')
        for ln in sorted(claimed - expected):
            problems.append(f'{rel}:{ln}: Sybil parses something here that the scan does not recognise')

        for d in directives:
            if not d['column0']:
                problems.append(f'{rel}:{d["line"]}: directive not at column 0 (indented comments can render as text)')
            if 'clear-namespace' in d['text']:
                problems.append(f'{rel}:{d["line"]}: clear-namespace is not supported (it drops the per-page module)')

        # Skips: reasons, balance, no "next" before a multi-prompt doctest, and how many examples each
        # covers (recorded in the manifest, so widening a skip shows in review).
        open_start = None
        covered = None  # index into skips of the open "start", counting what it covers
        ordered = sorted(regions, key=lambda r: r['line'])
        for k, r in enumerate(ordered):
            if r['kind'] != 'skip':
                if covered is not None:
                    skips[covered][1] += 1
                continue
            action, reason = r['parsed']
            where = f'{rel}:{r["line"]}'
            if action in ('next', 'start') and not (reason and reason.strip()):
                problems.append(
                    f'{where}: "skip: {action}" without a reason (Sybil reports the skipped code as PASSED)'
                )
            if action == 'start':
                if open_start is not None:
                    problems.append(f'{where}: "skip: start" inside another skip block')
                open_start = r['line']
                skips.append([f'{rel}  start  {(reason or "").strip()}', 0])
                covered = len(skips) - 1
            elif action == 'end':
                if open_start is None:
                    problems.append(f'{where}: "skip: end" without a start')
                open_start = None
                covered = None
            elif action == 'next':
                nxt = ordered[k + 1] if k + 1 < len(ordered) else None
                if nxt is not None and nxt['kind'] == 'doctest':
                    problems.append(
                        f'{where}: "skip: next" before a >>> block skips only its first prompt; use start/end'
                    )
            if action == 'next':
                skips.append([f'{rel}  next  {(reason or "").strip()}', 1])
        if open_start is not None:
            problems.append(f'{rel}:{open_start}: "skip: start" without an end (the rest of the page would pass unrun)')

        for f in fences:
            if f['info'] == 'python' and f['body'] and not f['body'][0].lstrip().startswith('>>>'):
                for stmt in _shown_values(f['body']):
                    problems.append(
                        f'{rel}:{f["line"]}: shown value in a comment, unchecked: {stmt!r} (make it a >>> example)'
                    )
            if f['info'] == 'python' and f['body'] and f['body'][0].lstrip().startswith('>>>'):
                for msg in _doctest_problems(f['body']):
                    problems.append(f'{rel}:{f["line"]}: {msg}')
            if f['info'] == 'python' and any('__name__' in ln for ln in f['body']):
                problems.append(f'{rel}:{f["line"]}: uses __name__ (the page runs under a per-page module name)')

        n_py = sum(f['info'] == 'python' for f in fences)
        n_inv = sum('invisible-code-block' in d['text'] for d in directives)
        summary.append(f'{rel}: {n_py} python fences, {n_inv} invisible blocks, {len(regions)} Sybil regions')

    # Files under docs/ that is_doc_file would not collect (e.g. a non-.md extension used for pages).
    for p in sorted((ROOT / 'docs').rglob('*.md')):
        if not docs_support.is_doc_file(p):
            problems.append(f'{p.relative_to(ROOT)}: a docs page outside the collected set')
    return problems, [f'{text}  covers={n}' for text, n in skips], summary


def test_docs_inventory():
    problems, _, summary = _inventory()
    print('\n'.join(summary))  # noqa: T201 (the per-page inventory, shown with -s)
    assert not problems, '\n' + '\n'.join(problems)


def test_docs_skip_manifest():
    _, skips, _ = _inventory()
    lines = MANIFEST.read_text(encoding='utf-8').splitlines()
    manifest = [ln for ln in lines if ln.strip() and not ln.startswith('#')]
    assert collections.Counter(skips) == collections.Counter(manifest), (
        '\nSkips in the docs differ from docs_skips.txt. If the change is intended, update the manifest.\n'
        'In the docs now:\n  ' + '\n  '.join(skips)
    )
