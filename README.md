# AIRAJ AI Service

FastAPI Python service for resume analysis, job matching, and ATS scoring.

## Quick Start

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Runs at: http://localhost:8000

## ML Models (Optional)

Place your trained models in the `models/` directory:
- `models/best_model.pkl` — XGBoost resume category classifier
- `models/label_encoder.pkl` — Label encoder for categories

If not present, the service uses rule-based category prediction automatically.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check — model status |
| `POST` | `/extract-skills` | Extract skills from text |
| `POST` | `/analyze` | Full resume analysis + optional job match |
| `POST` | `/bulk-analyze` | Match resume against multiple jobs |
| `POST` | `/ats-analyze` | ATS compatibility analysis |
| `GET` | `/skill-resources` | All 60 skill learning resources |

## Skills Tracked (60)

Python, JavaScript, TypeScript, React, Vue, Angular, Node.js, Express, Django, Flask, FastAPI, MongoDB, PostgreSQL, MySQL, Redis, Docker, Kubernetes, AWS, GCP, Azure, Git, CI/CD, Machine Learning, Deep Learning, TensorFlow, PyTorch, scikit-learn, Pandas, NumPy, SQL, HTML, CSS, Tailwind, Bootstrap, REST API, GraphQL, Microservices, Linux, Bash, Java, C++, C#, PHP, Ruby, Swift, Kotlin, Flutter, React Native, Tableau, Power BI, Excel, Spark, Hadoop, NLP, Computer Vision, Data Analysis, Data Science, XGBoost, OpenCV, Selenium, Jest, pytest
