G3 - A3/
│
├── CODE/
│   ├── app.py                 # Streamlit Web UI
│   ├── backend.py             # NustCopilot Class (Logic)
│   ├── config.py              # Configuration & Seeds
│   ├── eval_final.py          # Evaluation Script
│   └── requirements.txt       # Dependencies list
│   └── data/
│        └── nust_corpus_preprocessed.json
│
├── DATA/
│   ├── nust_corpus_preprocessed.json  # Data Chunks
│   ├── gen_data.py                    # Question Generator Script
│   └── eval_set.json                  # 120 Questions
│
├── DOCS/
│   ├── Report.pdf             # Final Report
│   └── Provenance_Sheet.csv   # The Excel sheet from Assignment 1
│
└── README.txt                 # Folder Info


Run 'streamlit run CODE/app.py' to start
Run 'pip install -r CODE/requirements.txt' to download dependencies.