from src.train_model import train_model


def test_train_model():
    model = train_model()
    assert model is not None, "Model training failed"
