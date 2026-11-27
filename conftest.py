"""Repo-root pytest configuration: the docs example tests (Sybil) and a scratch HOME.

Docs tests: ``uv run pytest README.md docs``. The machinery is in ``cfdb/tests/docs_support.py``.
"""

import atexit
import os
import shutil
import tempfile

# Every pytest run here gets a throwaway HOME (and XDG dirs), set before anything is imported, so
# no test or docs example can write into the developer's home directory.
_SCRATCH_HOME = tempfile.mkdtemp(prefix='pytest-home-')
atexit.register(shutil.rmtree, _SCRATCH_HOME, ignore_errors=True)
os.environ['HOME'] = _SCRATCH_HOME
for _var in ('XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME'):
    os.environ[_var] = os.path.join(_SCRATCH_HOME, _var.lower())

import pytest  # noqa: E402

from cfdb.tests import docs_support  # noqa: E402

_collect_docs = docs_support.make_sybil().pytest()


def pytest_collect_file(file_path, parent):
    if docs_support.is_doc_file(file_path):
        return _collect_docs(file_path, parent)
    return None


def pytest_sessionfinish(session):
    docs_support.session_finish(session)


def pytest_collection_modifyitems(items):
    for item in items:
        if docs_support.is_docs_item(item):
            item.add_marker(pytest.mark.timeout(docs_support.EXAMPLE_TIMEOUT, method='signal'))


def pytest_runtest_setup(item):
    # Page fail-fast: after one example on a page fails, the rest of the page is skipped (a leaked
    # dataset lock would otherwise cost the full timeout for every later block).
    if docs_support.is_docs_item(item) and docs_support.page_key(item) in docs_support.STATE.failed_pages:
        pytest.skip('an earlier example on this page failed')


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item):
    outcome = yield
    report = outcome.get_result()
    if docs_support.is_docs_item(item) and report.failed:
        docs_support.STATE.failed_pages.add(docs_support.page_key(item))
