import numpy as np
import pytest

from tests.helpers import makeSpace


def test_grid_shape_is_reversed_widths():
    space = makeSpace((-1.0, 1.0, 11, False), (-2.0, 2.0, 21, False))
    assert space.numDimensions == 2
    assert space.gridShape == (21, 11)


def test_displacement_is_plain_difference_when_not_periodic():
    space = makeSpace((-5.0, 5.0, 11, False))
    disp = space.displacement(np.array([1.0]), np.array([4.0]))
    assert disp == pytest.approx([3.0])


def test_displacement_uses_minimum_image_across_the_periodic_seam():
    # domain length 2*pi; going from 3.0 to -3.0 is +0.2832, not -6.0
    space = makeSpace((-np.pi, np.pi, 11, True))
    disp = space.displacement(np.array([3.0]), np.array([-3.0]))
    assert disp == pytest.approx([2 * np.pi - 6.0])


def test_endpoint_wraps_back_into_the_periodic_domain():
    space = makeSpace((-np.pi, np.pi, 11, True))
    end = space.endpoint(np.array([3.0]), np.array([1.0]))
    assert -np.pi <= end[0] <= np.pi
    assert end == pytest.approx([4.0 - 2 * np.pi])


def test_displacement_and_endpoint_round_trip():
    space = makeSpace((-np.pi, np.pi, 11, True), (-1.0, 1.0, 11, False))
    a = np.array([2.0, -0.3])
    b = np.array([-2.5, 0.8])
    assert space.endpoint(a, space.displacement(a, b)) == pytest.approx(b)


def test_closest_node_clamps_outside_the_domain():
    space = makeSpace((0.0, 1.0, 11, False))
    assert space.closestNode(np.array([0.5])) == (5,)
    assert space.closestNode(np.array([-3.0])) == (0,)
    assert space.closestNode(np.array([9.0])) == (10,)


def test_closest_node_wraps_for_periodic_cvs():
    space = makeSpace((0.0, 1.0, 11, True))
    # raw index is -1 here; periodic wrap gives 10 (the domain's other edge),
    # not clip-to-0 as a non-periodic CV would
    assert space.closestNode(np.array([-0.1])) == (10,)


def test_grid_distances_identify_the_periodic_endpoints():
    space = makeSpace((0.0, 1.0, 11, True))
    dist = space.gridDistances(np.array([0.25]))[0]
    assert dist[-1] == pytest.approx(dist[0])
    assert np.all(np.abs(dist) <= 0.5 + 1e-12)


def test_bounded_grid_triples_then_folds_back():
    space = makeSpace((0.0, 1.0, 11, False), bounded=True)
    values = np.zeros(33)
    folded = space.foldedGrid(values)
    assert folded.shape == (11,)
    # three contributions of log(1) each fold to log(3)
    assert folded == pytest.approx(np.full(11, np.log(3.0)))


def test_folded_grid_is_identity_when_not_bounded():
    space = makeSpace((0.0, 1.0, 11, False))
    values = np.arange(11.0)
    assert space.foldedGrid(values) == pytest.approx(values)


def test_two_periodic_cvs_construct_without_error():
    """Regression: the source implementation cannot construct this at all.

    It indexes self._lengths[self._pdims] with self._pdims a bare tuple.
    Numpy treats a length-1 tuple index as basic indexing (collapses to a
    scalar) and a longer tuple as one index per array dimension (raises
    "too many indices" on a 1-D array). Confirmed against the root
    online_kde.py and both its frozen per-experiment copies, identical in
    all three: constructing a CVSpace with ANY periodic CV crashes, which
    includes the phi/psi (2 periodic CVs) configuration used in both OPES
    papers' own example figures. This port uses list(self._pdims) instead,
    which numpy treats as fancy indexing and returns an array as intended.
    """
    space = makeSpace((-np.pi, np.pi, 41, True), (-np.pi, np.pi, 41, True))
    assert space.gridShape == (41, 41)
    disp = space.displacement(np.array([3.0, 3.0]), np.array([-3.0, -3.0]))
    assert disp == pytest.approx([2 * np.pi - 6.0, 2 * np.pi - 6.0])
