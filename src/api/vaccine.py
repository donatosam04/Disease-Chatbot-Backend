import os
import pandas as pd
from datetime import datetime
import re

# ---------------------------------------------------
# Load dataset once (safe loading with encoding fix)
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Complete Vaccination Dataset.csv")

try:
    vaccine_df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")
except Exception as e:
    print("Vaccine dataset loading error:", e)
    vaccine_df = None


# ---------------------------------------------------
# Extract birth date from text
# ---------------------------------------------------

def extract_birth_date(text: str):
    """
    Supports:
    - 17 February 2025
    - 17 february 2025
    - 17-02-2025
    """

    text = text.strip()

    # Format: DD Month YYYY
    match = re.search(r'(\d{1,2}\s+\w+\s+\d{4})', text)
    if match:
        try:
            return datetime.strptime(match.group(1), "%d %B %Y")
        except:
            pass

    # Format: DD-MM-YYYY
    match = re.search(r'(\d{1,2}-\d{1,2}-\d{4})', text)
    if match:
        try:
            return datetime.strptime(match.group(1), "%d-%m-%Y")
        except:
            pass

    return None


# ---------------------------------------------------
# Vaccine Response Logic
# ---------------------------------------------------

def vaccine_response(text: str):

    if vaccine_df is None:
        return None

    birth_date = extract_birth_date(text)
    if not birth_date:
        return None  # Not a vaccine-related message

    today = datetime.today()
    age_weeks = (today - birth_date).days // 7

    # Normalize column names
    df = vaccine_df.copy()
    df.columns = [col.strip().lower() for col in df.columns]

    if "weeks_due" not in df.columns or "vaccine_name" not in df.columns:
        return "⚠ Vaccine dataset format is incorrect."

    # Convert weeks_due safely to numeric
    df["weeks_due"] = pd.to_numeric(df["weeks_due"], errors="coerce")
    df = df.dropna(subset=["weeks_due"])

    # Get vaccines due up to current age
    due_vaccines = df[df["weeks_due"] <= age_weeks]

    if due_vaccines.empty:
        return "No vaccines are currently due."

    message = f"Your baby is currently {age_weeks} weeks old.\n\n"
    message += "Recommended vaccines:\n"

    for _, row in due_vaccines.iterrows():
        message += f"• {row['vaccine_name']} (Week {int(row['weeks_due'])})\n"

    message += "\n⚠ This schedule follows standard immunization guidelines."
    message += "\nPlease consult your pediatrician before vaccination."

    return message
