"""
Tests for sublattice grouping in the Lattice class and lattice classification.
"""
import pytest
import numpy as np
from cls_finder.core.lattice import Lattice
from cls_finder.core.lattice_classify import classify_lattice


# ─── Sublattice Auto-Detection Tests ─────────────────────────────────────────

class TestSublatticeGrouping:
    """Test that orbitals are correctly grouped into sublattices."""

    def test_single_orbital_single_sublattice(self):
        """One orbital = one sublattice."""
        lat = Lattice(
            dimension=1,
            primitive_vectors=[[1.0]],
            orbitals=[{"label": "A", "position": [0.0]}]
        )
        assert lat.sublattice_count == 1
        assert lat.orbitals_per_sublattice == [1]
        assert lat.is_multi_orbital is False

    def test_two_orbitals_different_positions(self):
        """Two orbitals at different positions = two sublattices."""
        lat = Lattice(
            dimension=1,
            primitive_vectors=[[1.0]],
            orbitals=[
                {"label": "A", "position": [0.0]},
                {"label": "B", "position": [0.5]}
            ]
        )
        assert lat.sublattice_count == 2
        assert lat.orbitals_per_sublattice == [1, 1]
        assert lat.is_multi_orbital is False

    def test_multi_orbital_same_position(self):
        """Three orbitals at the same position = one sublattice with 3 orbitals."""
        lat = Lattice(
            dimension=3,
            primitive_vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            orbitals=[
                {"label": "px", "position": [0.0, 0.0, 0.0]},
                {"label": "py", "position": [0.0, 0.0, 0.0]},
                {"label": "pz", "position": [0.0, 0.0, 0.0]}
            ]
        )
        assert lat.sublattice_count == 1
        assert lat.orbitals_per_sublattice == [3]
        assert lat.is_multi_orbital is True

    def test_kagome_three_sublattices(self):
        """Kagome lattice has 3 sublattices, each with 1 orbital."""
        lat = Lattice(
            dimension=2,
            primitive_vectors=[[1.0, 0.0], [0.5, 0.8660254037844386]],
            orbitals=[
                {"label": "A", "position": [0.5, 0.0]},
                {"label": "B", "position": [0.0, 0.5]},
                {"label": "C", "position": [0.5, 0.5]}
            ]
        )
        assert lat.sublattice_count == 3
        assert lat.orbitals_per_sublattice == [1, 1, 1]
        assert lat.is_multi_orbital is False

    def test_explicit_sublattice_hints(self):
        """Explicit sublattice hints override position-based detection."""
        lat = Lattice(
            dimension=2,
            primitive_vectors=[[1.0, 0.0], [0.0, 1.0]],
            orbitals=[
                {"label": "px", "position": [0.0, 0.0], "sublattice": 0},
                {"label": "py", "position": [0.0, 0.0], "sublattice": 0},
                {"label": "s",  "position": [0.5, 0.5], "sublattice": 1}
            ]
        )
        assert lat.sublattice_count == 2
        assert lat.orbitals_per_sublattice == [2, 1]
        assert lat.orbitals[0]["sublattice_index"] == 0
        assert lat.orbitals[1]["sublattice_index"] == 0
        assert lat.orbitals[2]["sublattice_index"] == 1

    def test_mixed_sublattice_structure(self):
        """Mix of multi-orbital and single-orbital sublattices."""
        lat = Lattice(
            dimension=2,
            primitive_vectors=[[1.0, 0.0], [0.0, 1.0]],
            orbitals=[
                {"label": "px", "position": [0.0, 0.0]},
                {"label": "py", "position": [0.0, 0.0]},
                {"label": "s",  "position": [0.5, 0.5]}
            ]
        )
        assert lat.sublattice_count == 2
        assert lat.orbitals_per_sublattice == [2, 1]
        assert lat.is_multi_orbital is True
        # First two orbitals share sublattice 0
        assert lat.orbitals[0]["sublattice_index"] == 0
        assert lat.orbitals[1]["sublattice_index"] == 0
        assert lat.orbitals[2]["sublattice_index"] == 1

    def test_orbital_index_in_sublattice(self):
        """Check orbital_index_in_sublattice is correct."""
        lat = Lattice(
            dimension=2,
            primitive_vectors=[[1.0, 0.0], [0.0, 1.0]],
            orbitals=[
                {"label": "px", "position": [0.0, 0.0]},
                {"label": "py", "position": [0.0, 0.0]},
                {"label": "s",  "position": [0.5, 0.5]}
            ]
        )
        assert lat.orbitals[0]["orbital_index_in_sublattice"] == 0
        assert lat.orbitals[1]["orbital_index_in_sublattice"] == 1
        assert lat.orbitals[2]["orbital_index_in_sublattice"] == 0


# ─── Cartesian Position Tests ────────────────────────────────────────────────

class TestCartesianPosition:
    def test_basic_position(self):
        """Check Cartesian position calculation."""
        lat = Lattice(
            dimension=2,
            primitive_vectors=[[1.0, 0.0], [0.0, 1.0]],
            orbitals=[
                {"label": "A", "position": [0.0, 0.0]},
                {"label": "B", "position": [0.5, 0.5]}
            ]
        )
        pos_A = lat.get_cartesian_position([0, 0], 0)
        np.testing.assert_allclose(pos_A, [0.0, 0.0])

        pos_B = lat.get_cartesian_position([0, 0], 1)
        np.testing.assert_allclose(pos_B, [0.5, 0.5])

        pos_A_11 = lat.get_cartesian_position([1, 1], 0)
        np.testing.assert_allclose(pos_A_11, [1.0, 1.0])

    def test_sublattice_position(self):
        """Check sublattice-level position."""
        lat = Lattice(
            dimension=2,
            primitive_vectors=[[1.0, 0.0], [0.0, 1.0]],
            orbitals=[
                {"label": "px", "position": [0.0, 0.0]},
                {"label": "py", "position": [0.0, 0.0]},
                {"label": "s",  "position": [0.5, 0.5]}
            ]
        )
        pos_sub0 = lat.get_cartesian_sublattice_position([0, 0], 0)
        np.testing.assert_allclose(pos_sub0, [0.0, 0.0])

        pos_sub1 = lat.get_cartesian_sublattice_position([0, 0], 1)
        np.testing.assert_allclose(pos_sub1, [0.5, 0.5])


# ─── Lattice Classification Tests ────────────────────────────────────────────

class TestLatticeClassification:
    def test_classify_kagome(self):
        """Kagome should be classified as hexagonal with 3 sublattices."""
        lat = Lattice(
            dimension=2,
            primitive_vectors=[[1.0, 0.0], [0.5, 0.8660254037844386]],
            orbitals=[
                {"label": "A", "position": [0.5, 0.0]},
                {"label": "B", "position": [0.0, 0.5]},
                {"label": "C", "position": [0.5, 0.5]}
            ]
        )
        info = classify_lattice(lat)
        assert info["bravais_type"] == "hexagonal"
        assert info["sublattice_count"] == 3
        assert info["known_lattice"] == "kagome"
        assert info["is_multi_orbital"] is False

    def test_classify_lieb(self):
        """Lieb should be classified as square with 3 sublattices."""
        lat = Lattice(
            dimension=2,
            primitive_vectors=[[1.0, 0.0], [0.0, 1.0]],
            orbitals=[
                {"label": "A", "position": [0.0, 0.0]},
                {"label": "B", "position": [0.5, 0.0]},
                {"label": "C", "position": [0.0, 0.5]}
            ]
        )
        info = classify_lattice(lat)
        assert info["bravais_type"] == "square"
        assert info["sublattice_count"] == 3
        assert info["known_lattice"] == "lieb"

    def test_classify_cubic_3d_multi_orbital(self):
        """Cubic 3D with p-orbitals: simple_cubic, 1 sublattice, multi-orbital."""
        lat = Lattice(
            dimension=3,
            primitive_vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            orbitals=[
                {"label": "px", "position": [0.0, 0.0, 0.0]},
                {"label": "py", "position": [0.0, 0.0, 0.0]},
                {"label": "pz", "position": [0.0, 0.0, 0.0]}
            ]
        )
        info = classify_lattice(lat)
        assert info["bravais_type"] == "simple_cubic"
        assert info["sublattice_count"] == 1
        assert info["is_multi_orbital"] is True
        assert info["orbitals_per_sublattice"] == [3]

    def test_classify_1d_chain(self):
        """Simple 1D chain."""
        lat = Lattice(
            dimension=1,
            primitive_vectors=[[1.0]],
            orbitals=[{"label": "A", "position": [0.0]}]
        )
        info = classify_lattice(lat)
        assert info["bravais_type"] == "1d_chain"
        assert info["sublattice_count"] == 1

    def test_classify_checkerboard(self):
        """Checkerboard = square with 2 sublattices."""
        lat = Lattice(
            dimension=2,
            primitive_vectors=[[1.0, 0.0], [0.0, 1.0]],
            orbitals=[
                {"label": "A", "position": [0.0, 0.0]},
                {"label": "B", "position": [0.5, 0.5]}
            ]
        )
        info = classify_lattice(lat)
        assert info["bravais_type"] == "square"
        assert info["sublattice_count"] == 2
        # Either checkerboard or bilayer_square
        assert info["known_lattice"] in ("checkerboard", "bilayer_square")

    def test_classify_honeycomb(self):
        """Honeycomb = hexagonal with 2 sublattices."""
        lat = Lattice(
            dimension=2,
            primitive_vectors=[[1.0, 0.0], [0.5, 0.8660254037844386]],
            orbitals=[
                {"label": "A", "position": [0.0, 0.0]},
                {"label": "B", "position": [1/3, 1/3]}
            ]
        )
        info = classify_lattice(lat)
        assert info["bravais_type"] == "hexagonal"
        assert info["sublattice_count"] == 2
        assert info["known_lattice"] == "honeycomb"


# ─── Bravais Type Tests ──────────────────────────────────────────────────────

class TestBravaisType:
    def test_hexagonal(self):
        lat = Lattice(
            dimension=2,
            primitive_vectors=[[1.0, 0.0], [0.5, 0.8660254037844386]],
            orbitals=[{"label": "A", "position": [0.0, 0.0]}]
        )
        bravais = lat.get_bravais_type()
        assert bravais["type"] == "hexagonal"

    def test_square(self):
        lat = Lattice(
            dimension=2,
            primitive_vectors=[[1.0, 0.0], [0.0, 1.0]],
            orbitals=[{"label": "A", "position": [0.0, 0.0]}]
        )
        bravais = lat.get_bravais_type()
        assert bravais["type"] == "square"

    def test_rectangular(self):
        lat = Lattice(
            dimension=2,
            primitive_vectors=[[1.0, 0.0], [0.0, 2.0]],
            orbitals=[{"label": "A", "position": [0.0, 0.0]}]
        )
        bravais = lat.get_bravais_type()
        assert bravais["type"] == "rectangular"


# ─── Pipeline Integration Test ───────────────────────────────────────────────

class TestPipelineIntegration:
    """Ensure all existing model library models work with the new Lattice class."""

    def test_all_models_parse(self):
        """All models should construct Lattice objects without error."""
        from cls_finder.models import library
        models = [
            library.zigzag_chain,
            library.kagome_nn,
            library.bilayer_square,
            library.lieb,
            library.modified_lieb,
            library.checkerboard_1,
            library.checkerboard_2,
            library.checkerboard_3,
            library.honeycomb_flat,
            library.cubic_3D,
            library.kagome_3,
        ]
        for model_fn in models:
            spec = model_fn()
            lat_spec = spec["lattice"]
            lat = Lattice(
                dimension=lat_spec["dimension"],
                primitive_vectors=lat_spec["primitive_vectors"],
                orbitals=lat_spec["orbitals"]
            )
            assert lat.num_orbitals > 0
            assert lat.sublattice_count > 0
            assert len(lat.sublattices) == lat.sublattice_count

            # Every orbital should have sublattice_index
            for orb in lat.orbitals:
                assert "sublattice_index" in orb
                assert "orbital_index_in_sublattice" in orb
