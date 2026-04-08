# Glioma Detection — Flask Web App (Scaffold)

This web app is a scaffold that integrates with your notebook at:

```
/mnt/data/Glioma_Detction_With_LSTM_and_BILSTM.ipynb
```

## How it works
- The app expects preprocessed input suitable for your model: a single example as a `.npy` array or a CSV row.
- The app looks for a trained Keras model at `models/model.h5`. If found, it will attempt to load and run predictions.
- If no model is found the app returns a deterministic dummy prediction so it never errors.

## Steps to use with your notebook
1. Open your notebook `/mnt/data/Glioma_Detction_With_LSTM_and_BILSTM.ipynb` and, after training your Keras model, save it:
```python
model.save("models/model.h5")
```
Copy or move the `model.h5` file into the project's `models/` folder.

2. Prepare a single example input as a `.npy` file (shape should match what the model expects). Example:
```python
import numpy as np
# x is a numpy array shaped (timesteps, features) or (1, timesteps, features)
np.save("example.npy", x)  # or np.save("example_single.npy", x[0])
```

3. Start the Flask app:
```
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000 and upload `example.npy` or a CSV containing a single example.

## If your model expects raw images
Tell me and I will modify the app to accept image uploads and the necessary preprocessing pipeline.

## Notes
- This scaffold intentionally avoids forcing heavy dependencies unless you place a Keras model in `models/model.h5`.
- The `glioma_model.py` returns a safe dummy prediction when no model is present so the web app remains error-free.
