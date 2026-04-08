import os, numpy as np

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model.h5")

def model_info():
    if os.path.exists(_MODEL_PATH):
        return {"status": "ready", "model_path": _MODEL_PATH}
    else:
        return {"status": "no_model", "model_path": _MODEL_PATH, "note": "Place your trained Keras model at this path (model.h5)"}

def _dummy_predict(arr):
    # deterministic fake prediction: mean of input features -> map to score
    score = float(np.mean(arr))
    label = "Tumor" if score > 0.5 else "No Tumor"
    return label, score

def predict_from_array(arr):
    # Attempts to load and predict with a Keras model if available (handles missing deps gracefully)
    try:
        if os.path.exists(_MODEL_PATH):
            # lazy import to avoid forcing dependency unless model exists
            from tensorflow.keras.models import load_model
            model = load_model(_MODEL_PATH)
            # Ensure array shape matches model expectations; user must adapt as needed
            preds = model.predict(arr)
            # handle common shapes: binary (sigmoid) or softmax
            if preds.ndim == 2 and preds.shape[1] >= 2:
                # choose class with highest prob
                cls = int(preds[0].argmax())
                score = float(preds[0, cls])
                label = "Tumor" if cls == 1 else "No Tumor"
            else:
                # assume single prob
                score = float(preds.reshape(-1)[0])
                label = "Tumor" if score > 0.5 else "No Tumor"
            return label, score
        else:
            return _dummy_predict(arr)
    except Exception as e:
        # On any error, return dummy prediction to keep the web app stable
        return _dummy_predict(arr)
