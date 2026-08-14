from pipeline.modules.search.sag2.candidate_scope import SAG2CandidateSubgraph


def test_scope_maps_are_bounded_and_bidirectional():
    scope = SAG2CandidateSubgraph(
        event_scores={"e1": 0.9, "e2": 0.8},
        event_to_entities={"e1": ["x"], "e2": ["x", "y"]},
        entity_to_events={"x": ["e1", "e2"], "y": ["e2"]},
    )

    assert scope.top_events(1)[0]["event_id"] == "e1"
    mapping, event_ids = scope.events_for_entities(["x"], limit_per_entity=2)
    assert event_ids == ["e1", "e2"]
    assert mapping["e2"] == ["x"]
    assert scope.entities_for_events(["e2"]) == ["x", "y"]
    assert scope.stats() == {"events": 2, "entities": 2, "edges": 3}


def test_scope_config_defaults_and_overrides():
    from pipeline.modules.search.config import SAGConfig

    default = SAGConfig()
    assert default.sag2_scope.enabled is False
    assert default.sag2_scope.event_top_k == 1000

    configured = SAGConfig(
        sag2_scope={"enabled": True, "event_top_k": 500, "include_event_content": False},
    )
    assert configured.sag2_scope.enabled is True
    assert configured.sag2_scope.event_top_k == 500
    assert configured.sag2_scope.include_event_content is False
