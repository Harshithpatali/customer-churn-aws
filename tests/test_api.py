from src.model_service import risk_band

def test_risk_bands():
    assert risk_band(0.1) == "Low"
    assert risk_band(0.5) == "Medium"
    assert risk_band(0.8) == "High"
