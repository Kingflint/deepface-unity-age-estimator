from __future__ import annotations

import pytest

from deepface_server.analyzers import calibration as c


def test_platt_fit_separable():
    scores = [-2, -1, 0, 1, 2]
    labels = [0, 0, 0, 1, 1]
    model = c.fit_platt(scores, labels, iterations=400, learning_rate=0.2)
    assert model.predict(2) > model.predict(-2)
    assert model.predict(-2) < 0.5
    assert model.predict(2) > 0.5


def test_platt_fit_validation():
    with pytest.raises(c.CalibrationError):
        c.fit_platt([1, 2], [0])
    with pytest.raises(c.CalibrationError):
        c.fit_platt([], [])
    with pytest.raises(c.CalibrationError):
        c.fit_platt([1.0], [2])


def test_platt_predict_range():
    model = c.PlattModel(a=1.0, b=0.0)
    p = model.predict(0.0)
    assert 0 < p < 1


def test_temperature_predict_uniform():
    model = c.TemperatureModel(temperature=1.0)
    out = model.predict([1.0, 1.0, 1.0])
    assert pytest.approx(sum(out)) == 1.0
    assert all(abs(v - 1 / 3) < 1e-9 for v in out)


def test_temperature_invalid():
    model = c.TemperatureModel(temperature=0)
    with pytest.raises(c.CalibrationError):
        model.predict([1.0, 2.0])


def test_temperature_higher_smooths():
    sharp = c.TemperatureModel(temperature=0.5).predict([5.0, 0.0])
    smooth = c.TemperatureModel(temperature=5.0).predict([5.0, 0.0])
    assert sharp[0] > smooth[0]


def test_fit_temperature_returns_positive():
    logits = [[2.0, 0.5], [0.1, 3.0], [1.5, 1.0]]
    labels = [0, 1, 0]
    model = c.fit_temperature(logits, labels, iterations=50)
    assert model.temperature > 0


def test_fit_temperature_validation():
    with pytest.raises(c.CalibrationError):
        c.fit_temperature([], [])
    with pytest.raises(c.CalibrationError):
        c.fit_temperature([[1.0]], [0, 1])


def test_ece_zero_for_perfect():
    probs = [1.0, 0.0, 1.0, 0.0]
    labels = [1, 0, 1, 0]
    assert c.expected_calibration_error(probs, labels, bins=2) == pytest.approx(0.0, abs=0.01)


def test_ece_validation():
    with pytest.raises(c.CalibrationError):
        c.expected_calibration_error([0.5], [0, 1])
    with pytest.raises(c.CalibrationError):
        c.expected_calibration_error([0.5], [0], bins=0)


def test_ece_empty_zero():
    assert c.expected_calibration_error([], []) == 0.0
