import numpy as np
import pytest

from openmm_opes.io import BiasSharer


def makeState(value):
    return {"positions": np.full((2, 1), value), "logSumW": float(value)}


def test_save_writes_one_file_named_for_the_walker(tmp_path):
    sharer = BiasSharer(str(tmp_path), walkerId=7)
    sharer.save(makeState(1.0))
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["kde_7_1.npz"]


def test_save_removes_the_previous_index(tmp_path):
    sharer = BiasSharer(str(tmp_path), walkerId=7)
    sharer.save(makeState(1.0))
    sharer.save(makeState(2.0))
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["kde_7_2.npz"]


def test_save_leaves_no_temporary_files(tmp_path):
    sharer = BiasSharer(str(tmp_path), walkerId=7)
    sharer.save(makeState(1.0))
    assert not any(p.name.startswith("temp_") for p in tmp_path.iterdir())


def test_load_ignores_the_walkers_own_files(tmp_path):
    sharer = BiasSharer(str(tmp_path), walkerId=7)
    sharer.save(makeState(1.0))
    assert sharer.load() == {}


def test_load_picks_up_a_peer(tmp_path):
    mine = BiasSharer(str(tmp_path), walkerId=1)
    peer = BiasSharer(str(tmp_path), walkerId=2)
    peer.save(makeState(5.0))
    loaded = mine.load()
    assert set(loaded) == {2}
    assert loaded[2]["logSumW"] == pytest.approx(5.0)
    assert loaded[2]["positions"] == pytest.approx(np.full((2, 1), 5.0))


def test_load_returns_nothing_when_no_peer_advanced(tmp_path):
    mine = BiasSharer(str(tmp_path), walkerId=1)
    peer = BiasSharer(str(tmp_path), walkerId=2)
    peer.save(makeState(5.0))
    assert set(mine.load()) == {2}
    assert mine.load() == {}


def test_load_rereads_a_peer_that_advanced(tmp_path):
    mine = BiasSharer(str(tmp_path), walkerId=1)
    peer = BiasSharer(str(tmp_path), walkerId=2)
    peer.save(makeState(5.0))
    mine.load()
    peer.save(makeState(6.0))
    loaded = mine.load()
    assert loaded[2]["logSumW"] == pytest.approx(6.0)


def test_load_tolerates_a_peer_file_vanishing_mid_scan(tmp_path):
    """A file deleted before os.listdir() runs never appears at all, so this
    corrupts the file's contents in place instead: it still appears in the
    directory listing, but fails when load() tries to open it, exactly like a
    real delete racing between listdir() and open()."""
    mine = BiasSharer(str(tmp_path), walkerId=1)
    peer = BiasSharer(str(tmp_path), walkerId=2)
    peer.save(makeState(5.0))
    for path in tmp_path.glob("kde_2_*.npz"):
        path.write_bytes(b"not a valid npz file")
    with pytest.warns(UserWarning, match="seems to have been deleted"):
        assert mine.load() == {}


def test_walker_ids_are_distinct_when_not_supplied(tmp_path):
    ids = {BiasSharer(str(tmp_path)).walkerId for _ in range(20)}
    assert len(ids) == 20


def test_saved_files_contain_no_pickled_objects(tmp_path):
    """The format must load with allow_pickle=False (spec section 6)."""
    sharer = BiasSharer(str(tmp_path), walkerId=7)
    sharer.save(makeState(1.0))
    path = next(tmp_path.glob("kde_7_*.npz"))
    with np.load(path, allow_pickle=False) as data:
        assert set(data) == {"positions", "logSumW"}


def test_a_restarted_walker_reclaims_its_own_file_slot(tmp_path):
    """Regression for review finding 7.

    A restart that draws a fresh index leaves the previous run's file on
    disk, where every peer reads it back forever as an extra walker,
    double-counting that walker's pre-restart kernels.
    """
    first = BiasSharer(str(tmp_path), walkerId=7)
    first.save(makeState(1.0))
    first.save(makeState(2.0))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["kde_7_2.npz"]

    # same walker comes back after a restart
    resumed = BiasSharer(str(tmp_path), walkerId=7)
    resumed.save(makeState(3.0))
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["kde_7_3.npz"], f"stale file left behind: {files}"


def test_a_peer_does_not_see_a_restarted_walker_twice(tmp_path):
    peer = BiasSharer(str(tmp_path), walkerId=1)
    walker = BiasSharer(str(tmp_path), walkerId=2)
    walker.save(makeState(5.0))
    assert set(peer.load()) == {2}

    restarted = BiasSharer(str(tmp_path), walkerId=2)
    restarted.save(makeState(6.0))
    # still exactly one peer, with the newer state -- not two
    loaded = peer.load()
    assert set(loaded) == {2}
    assert loaded[2]["logSumW"] == pytest.approx(6.0)
    assert len(list(tmp_path.glob("kde_*.npz"))) == 1
