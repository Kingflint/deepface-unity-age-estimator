from deepface_server.analyzers.base import AnalysisOutcome, FaceRegion
from deepface_server.analyzers.ensemble import EnsembleAnalyzer
from deepface_server.analyzers.mock_backend import DeterministicMockAnalyzer
from deepface_server.analyzers.registry import build_default_registry


def test_mock_is_deterministic():
    a = DeterministicMockAnalyzer()
    o1 = a.analyze(b"hello", ["age", "emotion", "gender"])
    o2 = a.analyze(b"hello", ["age", "emotion", "gender"])
    assert o1.age == o2.age
    assert o1.dominant_emotion == o2.dominant_emotion
    assert o1.backend == "mock"


def test_mock_varies_by_input():
    a = DeterministicMockAnalyzer()
    o1 = a.analyze(b"alpha", ["age"])
    o2 = a.analyze(b"beta", ["age"])
    assert o1.age != o2.age


def test_ensemble_merges_two_mocks():
    members = [DeterministicMockAnalyzer(seed_salt="A"), DeterministicMockAnalyzer(seed_salt="B")]
    ensemble = EnsembleAnalyzer(members)
    outcome = ensemble.analyze(b"image", ["age", "emotion", "gender"])
    assert outcome.dominant_emotion is not None
    assert outcome.dominant_gender is not None
    assert "mock" in outcome.backend


def test_ensemble_requires_members():
    import pytest

    with pytest.raises(ValueError):
        EnsembleAnalyzer([])


def test_ensemble_weight_validation():
    import pytest

    members = [DeterministicMockAnalyzer(), DeterministicMockAnalyzer()]
    with pytest.raises(ValueError):
        EnsembleAnalyzer(members, weights=[0.5])
    with pytest.raises(ValueError):
        EnsembleAnalyzer(members, weights=[0.0, 0.0])


def test_outcome_to_dict_round_trip():
    outcome = AnalysisOutcome(
        age=42.0,
        dominant_emotion="happy",
        dominant_gender="Woman",
        emotion_scores={"happy": 0.9, "sad": 0.1},
        gender_scores={"Woman": 0.8, "Man": 0.2},
        region=FaceRegion(x=10, y=20, w=30, h=40),
        backend="test",
    )
    data = outcome.to_dict()
    assert data["age"] == 42.0
    assert data["region"]["w"] == 30
    assert data["emotion_scores"]["happy"] == 0.9


def test_default_registry_lists_known_backends():
    registry = build_default_registry()
    names = registry.names()
    assert "mock" in names
    assert "deepface" in names
    mock = registry.create("mock")
    assert mock.name == "mock"


def test_registry_create_unknown_raises():
    import pytest

    registry = build_default_registry()
    with pytest.raises(KeyError):
        registry.create("bogus")
