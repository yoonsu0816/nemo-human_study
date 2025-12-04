# Human Study: Evaluation of Model Error Explanations

A Streamlit-based web application for conducting human evaluation studies on model error explanations. Participants can rank different explanation methods and provide feedback on their quality.

## Features

- **Participant Registration**: Simple participant ID entry system
- **Randomized Explanation Display**: Explanations are shown in random order to prevent bias
- **Ranking System**: Participants rank explanations from best (1st) to worst (4th)
- **Feedback Collection**: Participants can provide detailed feedback for each explanation
- **Progress Tracking**: Visual progress indicator showing completion status
- **Data Storage**: Automatic data saving to MongoDB and local JSON backup

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yoonsu0816/nemo-human_study.git
cd nemo-human_study
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
streamlit run main.py
```

2. **Participant Flow**:
   - Enter your Participant ID
   - Click "Start Evaluation"
   - Review each sample's image and error information
   - Rank the 4 explanations from best (1st) to worst (4th)
   - Provide feedback for each explanation
   - Click "Next →" to proceed to the next sample
   - Use "← Previous" to go back if needed

## Project Structure

```
nemo-human_study/
├── main.py                 # Entry point (participant registration)
├── pages/
│   └── 1_human-study.py   # Main evaluation page
├── human-study-data/       # Data directory
│   ├── data/              # Image samples
│   ├── outputs/           # Explanation JSON files
│   └── results/           # Local JSON backups
└── README.md
```

## Data Structure

Place your data files in the following structure:
```
human-study-data/
├── data/
│   └── nemo/
│       └── imagenet-r/
│           └── samples/
│               └── *.jpg
└── outputs/
    └── imagenet-r_clip/
        ├── error_retrieval.json
        ├── error_retrieval_cei.json
        ├── pixel_attribution.json
        ├── pixel_attribution_cei.json
        ├── change_of_caption.json
        ├── change_of_caption_cei.json
        ├── scitx.json
        └── scitx_cei.json
```

Data is saved both to MongoDB and locally in `human-study-data/results/{participant_id}_ratings.json`.
