# INTABEH Streamlit trial

This is a **separate deployment trial**. Core V1 files are byte-identical and checked against FREEZE.json. It excludes experimental CGM and Saudi siren services.

## Limitations
- Uses Streamlit `st.audio_input`: record then classify. **Not continuous listening**, not original UI.
- Does not establish iPhone background/locked-screen support.
- Free hosting memory limits may prevent TensorFlow+YAMNet from loading; test on deployed host.
- The YAMNet local TF-Hub cache is included; startup must not download models.
- Do not treat this as final graduation deployment until browser microphone and 9-class inference have been verified on real devices.

## Deploy
Create a GitHub repository, upload this folder's **contents** (not the ZIP), and set Streamlit entrypoint `streamlit_app.py`. If upload via browser fails due to GitHub file size or count, use Git on Windows. Do not upload private data or keys.
