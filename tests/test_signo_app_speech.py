import pytest

from signo_app.speech import speech_rate_from_percent


@pytest.mark.parametrize(
    ("percent", "rate"),
    [(50, -0.5), (100, 0.0), (150, 0.5)],
)
def test_speech_rate_from_percent(percent, rate):
    assert speech_rate_from_percent(percent) == rate


@pytest.mark.parametrize("percent", [49, 151])
def test_speech_rate_rejects_out_of_range_values(percent):
    with pytest.raises(ValueError, match="Speech rate must be between"):
        speech_rate_from_percent(percent)
