# Health Assist AI 🏥

Health Assist AI is an AI-powered multilingual health awareness and disease prognosis system designed for the Indian public-health context.

The system combines machine learning, semantic natural-language processing, conversational AI, emergency symptom triage, and vaccination tracking into a unified web application.

> **Disclaimer:** Health Assist AI is intended for health-awareness and educational purposes only. It is not a replacement for professional medical diagnosis or treatment.

---

## Features

- AI-powered disease prediction from natural-language symptoms
- Semantic symptom understanding using Sentence Transformers
- XGBoost-based disease classification
- Support for approximately 70 disease classes
- Conversational symptom clarification
- Emergency symptom triage
- Confidence-based prediction routing
- Groq-powered fallback for low-confidence cases
- Indian vaccination schedule tracker
- Natural-language and multilingual interaction
- Progressive Web App (PWA) frontend
- FastAPI backend
- Interactive Swagger API documentation

---

## System Architecture

```text
                         Health Assist AI
                                │
                                ▼
                    ┌──────────────────────┐
                    │     PWA Frontend     │
                    │   HTML / CSS / JS    │
                    └──────────┬───────────┘
                               │
                            /chat
                               │
                               ▼
                    ┌──────────────────────┐
                    │       FastAPI        │
                    │      AI Gateway      │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼─────────────────┐
              │                │                 │
              ▼                ▼                 ▼
        Emergency          Vaccine          Chitchat /
         Triage            Tracker          Vagueness
              │                │
              └────────────────┼─────────────────┘
                               │
                               ▼
                         Preprocessing
                               │
                               ▼
                    Sentence Transformers
                    all-mpnet-base-v2
                       768-D Embedding
                               │
                               ▼
                       XGBoost Classifier
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
             High Confidence        Lower Confidence
                    │                     │
                    ▼                     ▼
             Top Predictions       Clarification Questions
                                          │
                                          ▼
                                    Groq LLM Fallback
                                          │
                    ┌─────────────────────┘
                    ▼
                JSON Response
                    │
                    ▼
                Frontend UI
```

---

## Machine Learning

Health Assist AI uses a semantic embedding and XGBoost classification pipeline for disease prediction.

### Model Pipeline

```text
Natural-Language Symptoms
          │
          ▼
     Preprocessing
          │
          ▼
Sentence Transformer
all-mpnet-base-v2
          │
          ▼
  768-Dimensional
     Embedding
          │
          ▼
   XGBoost Classifier
          │
          ▼
 Disease Prediction
```

### Model Components

| Component       | Technology                                |
| --------------- | ------------------------------------------ |
| Embedding Model | `sentence-transformers/all-mpnet-base-v2` |
| Embedding Size  | 768 dimensions                            |
| Classifier      | XGBoost                                   |
| Disease Classes | Approximately 70                          |
| Label Encoding  | scikit-learn                              |

The user's natural-language symptom description is converted into a semantic embedding using `all-mpnet-base-v2`. The resulting 768-dimensional representation is then passed to the trained XGBoost classifier.

### Model Files

The final trained model artifacts are stored in:

```text
models_saved/
└── universal_disease_model/
    ├── classifier.pkl
    └── label_encoder.pkl
```

The pretrained MPNet model is downloaded through the Sentence Transformers library when required and is not stored inside the repository.

---

## Confidence-Based Prediction

Health Assist AI uses confidence-aware routing instead of blindly returning a prediction for every input.

```text
                  XGBoost Prediction
                          │
                          ▼
                     Confidence
                          │
             ┌────────────┼────────────┐
             │            │            │
             ▼            ▼            ▼
           ≥ 90%       40% – <90%     < 40%
             │            │            │
             ▼            ▼            ▼
        Top disease    Clarification   Groq
        predictions     questions      fallback
```

This allows the system to request additional information when the classifier is uncertain and use an AI fallback when the confidence is particularly low.

---

## Conversational Health Assistance

The FastAPI backend supports conversational health interactions in addition to direct disease prediction.

The system can process:

* Symptom-based health queries
* Disease prediction requests
* Follow-up clarification
* General health conversations
* Vague or insufficient symptom descriptions
* Vaccine-related questions
* Potential emergency situations

The conversational layer helps transform a simple symptom description into a more structured interaction before producing a response.

---

## Emergency Triage

Health Assist AI includes an emergency-triage layer that checks symptom descriptions for potentially serious situations before normal prediction processing.

This provides an additional safety layer for potentially urgent cases.

> Health Assist AI is not an emergency medical service. Users experiencing a medical emergency should seek immediate professional medical assistance.

---

## Vaccination Tracker

Health Assist AI includes an Indian vaccination tracking feature.

The vaccination dataset is stored at:

```text
data/raw/Complete Vaccination Dataset.csv
```

The vaccination tracking implementation is located at:

```text
src/api/vaccine.py
```

The feature is integrated with the FastAPI backend and frontend to provide vaccination-related information and schedule tracking.

---

## Frontend

The frontend is implemented as a lightweight Progressive Web App using:

* HTML5
* CSS3
* Vanilla JavaScript
* Web App Manifest
* Service Worker

The frontend files are located in:

```text
front/
├── index.html
├── app.html
├── app.js
├── style.css
├── manifest.json
└── service-worker.js
```

No frontend framework or build system is required.

---

## Project Structure

```text
Health-Assist-AI/
│
├── front/
│   ├── index.html
│   ├── app.html
│   ├── app.js
│   ├── style.css
│   ├── manifest.json
│   └── service-worker.js
│
├── src/
│   ├── api/
│   │   ├── main.py
│   │   ├── chatbot.py
│   │   ├── predictor.py
│   │   ├── schemas.py
│   │   ├── utils.py
│   │   └── vaccine.py
│   │
│   ├── static/
│   │   └── script.js
│   │
│   └── templates/
│       └── index.html
│
├── data/
│   └── raw/
│       └── Complete Vaccination Dataset.csv
│
├── models_saved/
│   └── universal_disease_model/
│       ├── classifier.pkl
│       └── label_encoder.pkl
│
├── requirements.txt
├── Procfile
├── README.md
└── .gitignore
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/donatosam04/Disease-Chatbot-Backend.git
cd Disease-Chatbot-Backend
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

The `.env` file is excluded from Git.

**Never commit API keys, passwords, or other sensitive credentials to the repository.**

---

## Running the Backend

Start the FastAPI application using:

```bash
uvicorn src.api.main:app --reload
```

The backend will be available at:

```text
http://127.0.0.1:8000
```

### Health Check

```text
http://127.0.0.1:8000/health
```

Example response:

```json
{
  "status": "ok"
}
```

### Swagger Documentation

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### Chat Endpoint

```text
POST /chat
```

The `/chat` endpoint handles conversational input and routes requests through the appropriate Health Assist AI components.

---

## Running the Frontend

The frontend is located in the `front/` directory and does not require a build process.

For local testing, run:

```bash
python -m http.server 5500 --directory front
```

Then open:

```text
http://127.0.0.1:5500
```

When deploying the frontend separately, update the backend URL configuration in `front/app.js` if required.

---

## Deployment

The FastAPI backend can be deployed on a Python-compatible hosting platform.

Example production command:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
```

The `front/` directory can be deployed separately as a static website or Progressive Web App.

---

## Model Evaluation

During development, an earlier evaluation pipeline was found to contain duplicate-data leakage between training and evaluation data, resulting in inflated performance.

The disease-classification pipeline was subsequently rebuilt using a more realistic evaluation methodology.

The final approach is based on:

```text
Natural-Language Symptoms
          ↓
MPNet Semantic Embeddings
          ↓
XGBoost Classification
          ↓
Confidence-Based Routing
```

This project therefore avoids presenting inflated evaluation results caused by data leakage.

---

## Technology Stack

### Backend

* Python
* FastAPI
* Uvicorn
* XGBoost
* scikit-learn
* Sentence Transformers

### AI / NLP

* `sentence-transformers/all-mpnet-base-v2`
* Groq API

### Frontend

* HTML5
* CSS3
* Vanilla JavaScript
* Progressive Web App APIs

### Machine Learning

* Semantic text embeddings
* XGBoost classification
* Confidence-based prediction routing

---

## Project Objective

The objective of Health Assist AI is to demonstrate how modern Natural Language Processing, machine learning, conversational AI, and public-health data can be combined to create an accessible health-awareness platform.

The project focuses on:

* Natural-language health interaction
* Disease prediction
* Uncertainty-aware AI
* Emergency awareness
* Vaccination tracking
* Conversational assistance
* Accessible web-based healthcare technology

---

## Limitations

Health Assist AI is a research and educational project.

The system may produce incorrect predictions or responses because:

* Symptoms can overlap between multiple diseases.
* User descriptions may be incomplete or ambiguous.
* Machine-learning predictions are probabilistic.
* AI-generated responses may contain errors.
* Training data cannot represent every real-world medical situation.

The system should therefore not be used as a substitute for a qualified healthcare professional.

---

## Medical Disclaimer

**Health Assist AI does not provide medical diagnoses.**

All predictions and responses are intended for educational and health-awareness purposes only.

For severe, persistent, or emergency symptoms, seek assistance from a qualified healthcare professional or appropriate emergency medical service.
