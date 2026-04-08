from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import os, numpy as np
from glioma_model import predict_from_array, model_info

app = Flask(__name__)
app.secret_key = "change-me"

@app.route("/")
def index():
    return render_template("index.html", model_info=model_info())

@app.route("/predict", methods=["POST"])
def predict():
    # Supports uploading .npy (numpy array) or .csv files containing preprocessed features/sequences
    if "datafile" not in request.files:
        flash("No file part")
        return redirect(url_for("index"))
    f = request.files["datafile"]
    if f.filename == "":
        flash("No selected file")
        return redirect(url_for("index"))
    filename = f.filename.lower()
    try:
        if filename.endswith(".npy"):
            arr = np.load(f.stream)
        elif filename.endswith(".csv"):
            import io
            arr = np.loadtxt(io.TextIOWrapper(f.stream), delimiter=",")
        else:
            flash("Unsupported file type. Upload .npy or .csv")
            return redirect(url_for("index"))
    except Exception as e:
        flash(f"Could not read uploaded file: {e}")
        return redirect(url_for("index"))
    # Ensure shape (1, ...) for single example
    if arr.ndim == 1:
        arr = np.expand_dims(arr, axis=0)
    label, score = predict_from_array(arr)
    return render_template("result.html", label=label, score=f"{score:.4f}")

@app.route("/download/notebook")
def download_notebook():
    # provide link to the uploaded notebook path (on the same machine)
    # The path is provided in README; here we just show it as info.
    return redirect("file://" + os.path.abspath("/mnt/data/Glioma_Detction_With_LSTM_and_BILSTM.ipynb"))

if __name__ == "__main__":
    app.run(debug=True)
