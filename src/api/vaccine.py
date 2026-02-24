from pathlib import Path
import pandas as pd
from datetime import datetime
import re

# --------------------------------------------------
# DATASET LOADING (Bulletproof Path Handling)
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FOLDER = BASE_DIR / "data" / "raw"

vaccine_df = None

try:
    for file_name in [
        "Complete Vaccination Dataset.csv",
        "Complete Vaccination Dataset.xlsx"
    ]:
        path = DATA_FOLDER / file_name

        if path.exists():
            print(f"✅ Loading vaccine dataset from: {path}")

            if file_name.endswith(".csv"):
                vaccine_df = pd.read_csv(path, encoding="latin1")
            else:
                vaccine_df = pd.read_excel(path)

            break

    if vaccine_df is None:
        print("⚠ Vaccine dataset NOT found in data/raw")

except Exception as e:
    print("❌ Vaccine dataset loading error:", e)
    vaccine_df = None

# --------------------------------------------------
# DATE EXTRACTION
# --------------------------------------------------

def extract_birth_date(text: str):
    patterns = [
        r'(\d{1,2}\s+\w+\s+\d{4})',
        r'(\d{1,2}-\d{1,2}-\d{4})',
        r'(\d{1,2}/\d{1,2}/\d{4})'
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            date_str = match.group(1)

            for fmt in ("%d %B %Y", "%d-%m-%Y", "%d/%m/%Y"):
                try:
                    birth_date = datetime.strptime(date_str, fmt)

                    # Prevent future dates
                    if birth_date > datetime.today():
                        return None

                    return birth_date
                except:
                    continue

    return None

# --------------------------------------------------
# AGE CALCULATION
# --------------------------------------------------

def calculate_age(birth_date):
    today = datetime.today()
    days = (today - birth_date).days

    return {
        "weeks": max(days // 7, 0),
        "months": max(days // 30, 0),
        "years": max(days // 365, 0)
    }

# --------------------------------------------------
# STATUS CLASSIFICATION
# --------------------------------------------------

def classify_status(current_age_weeks, scheduled_week):

    if current_age_weeks < scheduled_week:
        return "Upcoming"

    elif scheduled_week <= current_age_weeks <= scheduled_week + 2:
        return "Due"

    else:
        return "Overdue"

# --------------------------------------------------
# MAIN TRACKING FUNCTION
# --------------------------------------------------

def vaccine_tracker(text: str):

    if vaccine_df is None:
        return None

    birth_date = extract_birth_date(text)

    if not birth_date:
        return None

    age = calculate_age(birth_date)
    age_weeks = age["weeks"]

    vaccine_status_list = []

    for _, row in vaccine_df.iterrows():

        age_group = str(row.get("Age_Group", "")).lower()
        vaccine_name = str(row.get("Vaccine", "Unknown"))
        disease = str(row.get("Disease_Prevented", "Unknown"))

        week_match = re.search(r'(\d+)\s*week', age_group)

        if week_match:
            scheduled_week = int(week_match.group(1))
            status = classify_status(age_weeks, scheduled_week)

            vaccine_status_list.append({
                "vaccine": vaccine_name,
                "prevents": disease,
                "scheduled_week": scheduled_week,
                "status": status
            })

    if not vaccine_status_list:
        return None

    return {
        "child_age": age,
        "vaccines": vaccine_status_list
    }