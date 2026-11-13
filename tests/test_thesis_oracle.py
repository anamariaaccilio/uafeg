"""Thesis oracle: derives from provenance (operation, regions, asserted classes), never the label."""
from src.schemas.claim import ClaimTransformation
from src.schemas.evidence import DecisionLabel
from uafeg import thesis_oracle
from uafeg.thesis_graph import build_thesis_graph, evidence_view
from tests.test_thesis_loader import thesis_packet


def with_transformation(op, params=None, denied=None):
    p = thesis_packet().model_copy(deep=True)
    params = dict(params or {})
    if denied:
        params["denied_annotated_class"] = denied
    p.claim.transformation = ClaimTransformation(operation=op, parameters=params, seed=0)
    p.claim.metadata = {"asserted_damage_classes": ["dent"]}
    p.claim.image_path = "x.png"
    p.claim.narrative_text = "El vehículo presenta abolladuras."
    return p


def test_oracle_narrative_polarity_from_provenance():
    affirmed = thesis_oracle.oracle_narrative(with_transformation("identity"))
    denied = thesis_oracle.oracle_narrative(with_transformation("narrative_contradiction", denied="dent"))
    assert affirmed.assertions[0].polarity == "affirmed" and affirmed.assertions[0].p == (1.0, 0.0, 0.0)
    assert denied.assertions[0].polarity == "denied" and denied.assertions[0].p == (0.0, 0.0, 1.0)


def test_oracle_forensic_flags_only_local_edits(monkeypatch):
    monkeypatch.setattr(thesis_oracle, "_sources", lambda: {
        "hitl:fixture": ((32, 32), [{"damage_class": "dent",
                                     "polygon": {"points": [[2, 2], [20, 2], [20, 20]]}}])})
    for op, expected in (("identity", 0.0), ("jpeg_recompression", 0.0),
                         ("localized_copy_move", 1.0)):
        p = with_transformation(op, {"destination_box": [0, 0, 8, 8]} if expected else {})
        p.claim.source_image_id = "hitl:fixture"
        visual, forensic = thesis_oracle.oracle_visual_forensic(p)
        assert forensic.score == expected
        assert [d.damage_class for d in visual.detections] == ["dent"]


def test_oracle_ignores_expected_label(monkeypatch):
    p = with_transformation("narrative_contradiction", denied="dent")
    q = p.model_copy(deep=True)
    q.claim.expected_label = DecisionLabel.SUPPORTED
    a = thesis_oracle.build_thesis_oracle_packet(p, True, False)
    b = thesis_oracle.build_thesis_oracle_packet(q, True, False)
    ga, gb = build_thesis_graph(evidence_view(a)), build_thesis_graph(evidence_view(b))
    assert (ga.x == gb.x).all() and ga.edges == gb.edges
