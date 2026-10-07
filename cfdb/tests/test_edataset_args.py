"""
Hermetic: open_edataset forwards ebooklet's storage-mode arguments only when
the caller gives them, so ebooklet's own defaults apply otherwise (an existing
remote keeps its mode; a new dataset is grouped). ebooklet.open_ebooklet is
replaced by a recorder - nothing touches a remote.
"""
import pytest

ebooklet = pytest.importorskip('ebooklet')

from cfdb import edataset  # noqa: E402


class _Stop(Exception):
    pass


@pytest.fixture
def recorded(monkeypatch):
    calls = []

    def fake_open_ebooklet(remote_conn, file_path, flag, **kwargs):
        calls.append(kwargs)
        raise _Stop()

    monkeypatch.setattr(edataset.ebooklet, 'open_ebooklet', fake_open_ebooklet)
    return calls


@pytest.mark.parametrize('given, expected', [
    ({}, {}),
    ({'group_bytes': None}, {'group_bytes': None}),
    ({'group_bytes': 1234}, {'group_bytes': 1234}),
    ({'num_groups': None}, {'num_groups': None}),
])
def test_storage_mode_args_forwarded_only_when_given(tmp_path, recorded, given, expected):
    with pytest.raises(_Stop):
        edataset.open_edataset('http://example.invalid/db', tmp_path / 'x.cfdb', flag='r', **given)
    forwarded = {k: v for k, v in recorded[0].items() if k in ('group_bytes', 'num_groups')}
    assert forwarded == expected


def test_group_bytes_is_keyword_only(tmp_path, recorded):
    """group_bytes sits where 0.10's num_groups was: a positional 0.10 call
    (a group COUNT) must fail loudly, not become a byte target."""
    with pytest.raises(TypeError):
        edataset.open_edataset('http://example.invalid/db', tmp_path / 'x.cfdb', 'r', 'grid', 'lz4', 1, 101)
    assert recorded == []
