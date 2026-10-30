from verl.experimental.agent_loop.single_turn_agent_loop import build_single_turn_segment_metadata


def test_single_turn_fixed_token_segments_ignore_non_action_tokens():
    metadata = build_single_turn_segment_metadata([1, 1, 0, 1, 1, 1, 0], configured_m=3)
    assert metadata["segment_ids"] == [0, 0, -1, 1, 1, 2, -1]
    assert metadata["segment_boundaries"] == [(0, 2), (3, 5), (5, 6)]
    assert metadata["segment_action_lengths"] == [2, 2, 1]
    assert metadata["segment_durations"] == [1.0, 1.0, 1.0]


def test_single_turn_default_keeps_one_segment():
    metadata = build_single_turn_segment_metadata([1, 1, 0, 1], configured_m=0)
    assert metadata["segment_ids"] == [0, 0, -1, 0]
    assert metadata["segment_boundaries"] == [(0, 4)]


def test_single_turn_metadata_has_replay_fields():
    metadata = build_single_turn_segment_metadata([1, 1, 0, 1], configured_m=2)
    assert len(metadata["segment_boundaries"]) == 2
    # The loop attaches trajectory/episode identifiers and action text after
    # this pure helper; the helper's boundaries remain the canonical alignment
    # source for those replay fields.
    assert metadata["segment_action_lengths"] == [2, 1]


def test_single_turn_metadata_exports_rollout_owned_successor_features():
    metadata = build_single_turn_segment_metadata([1, 1, 1, 1, 1, 1], configured_m=3, feature_dim=2)
    assert metadata["segment_transition_mask"] == [True, True, False]
    assert len(metadata["segment_next_features"]) == 3
    assert metadata["segment_next_features"][-1] == [0.0, 0.0]
