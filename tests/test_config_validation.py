"""Tests for MemoryConfig __post_init__ validation."""

import warnings

import pytest

from grafeo_memory import MemoryConfig


class TestMemoryConfigValidation:
    """Validate that MemoryConfig rejects bad values at construction."""

    def test_defaults_are_valid(self):
        config = MemoryConfig()
        assert config.embedding_dimensions == 1536

    def test_embedding_dimensions_zero(self):
        with pytest.raises(ValueError, match="embedding_dimensions must be positive"):
            MemoryConfig(embedding_dimensions=0)

    def test_embedding_dimensions_negative(self):
        with pytest.raises(ValueError, match="embedding_dimensions must be positive"):
            MemoryConfig(embedding_dimensions=-1)

    def test_similarity_threshold_zero(self):
        with pytest.raises(ValueError, match="similarity_threshold must be in"):
            MemoryConfig(similarity_threshold=0.0)

    def test_similarity_threshold_above_one(self):
        with pytest.raises(ValueError, match="similarity_threshold must be in"):
            MemoryConfig(similarity_threshold=1.1)

    def test_similarity_threshold_one_is_valid(self):
        config = MemoryConfig(similarity_threshold=1.0)
        assert config.similarity_threshold == 1.0

    def test_decay_rate_zero(self):
        with pytest.raises(ValueError, match="decay_rate must be positive"):
            MemoryConfig(decay_rate=0.0)

    def test_decay_rate_negative(self):
        with pytest.raises(ValueError, match="decay_rate must be positive"):
            MemoryConfig(decay_rate=-0.5)

    @pytest.mark.parametrize(
        "field_name",
        [
            "weight_similarity",
            "weight_recency",
            "weight_frequency",
            "weight_importance",
            "weight_topology",
            "topology_boost_factor",
            "structural_feedback_gamma",
            "consolidation_protect_threshold",
        ],
    )
    def test_weight_negative(self, field_name):
        with pytest.raises(ValueError, match=f"{field_name} must be in"):
            MemoryConfig(**{field_name: -0.1})

    @pytest.mark.parametrize(
        "field_name",
        [
            "weight_similarity",
            "weight_recency",
            "weight_frequency",
            "weight_importance",
            "weight_topology",
            "topology_boost_factor",
            "structural_feedback_gamma",
            "consolidation_protect_threshold",
        ],
    )
    def test_weight_above_one(self, field_name):
        with pytest.raises(ValueError, match=f"{field_name} must be in"):
            MemoryConfig(**{field_name: 1.1})

    def test_weight_boundaries_valid(self):
        config = MemoryConfig(
            weight_similarity=0.0,
            weight_recency=0.0,
            weight_frequency=0.0,
            weight_importance=0.0,
            weight_topology=1.0,
        )
        assert config.weight_topology == 1.0

    def test_weight_sum_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MemoryConfig(
                weight_similarity=0.5,
                weight_recency=0.5,
                weight_frequency=0.5,
                weight_importance=0.5,
            )
            assert len(w) == 1
            assert "sum to 2.000" in str(w[0].message)

    def test_weight_sum_no_warning_for_defaults(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MemoryConfig()
            assert len(w) == 0

    def test_yolo_still_works(self):
        config = MemoryConfig.yolo()
        assert config.enable_importance is True
        assert config.enable_vision is True
        assert config.usage_callback is not None

    def test_yolo_with_overrides(self):
        config = MemoryConfig.yolo(user_id="alice", embedding_dimensions=384)
        assert config.user_id == "alice"
        assert config.embedding_dimensions == 384
