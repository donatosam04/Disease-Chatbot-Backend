# Health Assist AI 🏥

**Health Assist AI** is an AI-powered multilingual health awareness and disease prognosis system designed for the Indian public-health context.

The project combines semantic symptom understanding, machine-learning-based disease prediction, conversational clarification, emergency symptom triage, AI-assisted fallback responses, and an Indian vaccination tracker into a unified web application.

> ⚠️ **Disclaimer:** Health Assist AI is intended for health awareness and educational purposes. It is not a replacement for professional medical diagnosis or treatment.

---

## ✨ Features

- 🧠 AI-powered disease prediction from natural-language symptoms
- 🔎 Semantic symptom understanding using Sentence Transformers
- 📊 XGBoost-based disease classification
- 🩺 Support for approximately 70 disease classes
- 💬 Conversational symptom clarification
- 🚨 Emergency symptom triage
- 🤖 Groq-powered fallback for low-confidence cases
- 💉 Indian vaccination schedule tracker
- 🌐 Natural-language / multilingual interaction
- 📱 Progressive Web App (PWA) frontend
- ⚡ FastAPI REST API
- 📖 Interactive Swagger API documentation

---

## 🏗️ System Architecture

```text
                    Health Assist AI
                           │
                           ▼
                ┌────────────────────┐
                │   PWA Frontend     │
                │ HTML / CSS / JS    │
                └─────────┬──────────┘
                          │
                     POST /chat
                          │
                          ▼
                ┌────────────────────┐
                │      FastAPI       │
                │    AI Gateway      │
                └─────────┬──────────┘
                          │
          ┌───────────────┼────────────────┐
          │               │                │
          ▼               ▼                ▼
      Emergency        Vaccine          Chitchat /
       Triage          Tracker          Vagueness
          │               │
          └───────────────┼────────────────┘
                          ▼
                    Preprocessing
                          │
                          ▼
             Sentence Transformers
             all-mpnet-base-v2
                  768-D embedding
                          │
                          ▼
                    XGBoost Model
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
       High-confidence          Lower-confidence
         prediction             input / uncertainty
              │                       │
              ▼                       ▼
        Top predictions        Clarification questions
                                      │
                                      ▼
                              Groq LLM fallback
                                      │
                    ┌─────────────────┘
                    ▼
               JSON Response
                    │
                    ▼
               Frontend UI