#!/bin/bash
pip install --no-cache-dir -r requirements.txt
streamlit run APP.py --server.port=8000 --server.address=0.0.0.0
