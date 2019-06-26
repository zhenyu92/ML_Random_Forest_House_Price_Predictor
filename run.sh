#!/bin/bash
pip3 install -r requirements.txt
curl -o real_estate.csv  https://aisgaiap.blob.core.windows.net/aiap4-assessment/real_estate.csv
python3 AIAP.py
