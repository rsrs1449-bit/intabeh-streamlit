INTABEH Streamlit deployment repair

1. Use Streamlit Cloud Python 3.11 (Advanced settings).
2. Upload all extracted files preserving directories, including intabeh_runtime.py.
3. The frozen FINAL_v1.0 package is unchanged; all SHA-256 checks match FREEZE.json.
4. The runtime reads Keras 2.21 HDF5 weights directly to avoid normalization-layer load errors.
5. This package was statically and numerically checked but has NOT been executed with TensorFlow/Streamlit in this environment.
