#!/bin/bash
chmod +x startup.s
echo "Installing dependencies..."
pip install --no-cache-dir -r requirements.txt

echo "Starting Streamlit app..."
streamlit run app.py --server.port=8000 --server.address=0.0.0.0
