# Waveguide Mode Visualizer

An English-language Streamlit web app for visualizing TE and TM field patterns in rectangular and circular waveguides.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py` and `requirements.txt`.
2. Visit https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app**, select your repository and branch, and set the main file path to `app.py`.
4. Deploy. Streamlit will provide a public URL that you can share.

The app accepts comma- or space-separated mode names, plots each valid mode, reports its cutoff wavenumber, and provides PNG downloads.
