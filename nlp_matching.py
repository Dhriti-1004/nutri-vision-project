import json
import re
import pandas as pd
from rapidfuzz import process, fuzz
import sqlite3

with open("modifiers.json", "r") as f:
    modifiers = json.load(f)

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_modifiers_from_text(user_input, threshold=90):
    normalized_input = normalize_text(user_input)
    words = normalized_input.split()
    matched_modifiers = []

    modifier_descriptions = [normalize_text(m["description"]) for m in modifiers]
    for i, mod in enumerate(modifiers):
        mod_desc = normalize_text(mod["description"])
        if mod_desc == "air fried":
            if "air fried" in normalized_input or "air-fried" in normalized_input:
                matched_modifiers.append((100, mod))
            continue

    for i, mod_desc in enumerate(modifier_descriptions):
        if mod_desc == "air fried":
            continue
        score = fuzz.partial_ratio(mod_desc, normalized_input)
        if score >= threshold:
            if modifiers[i] not in [m[1] for m in matched_modifiers]:
                matched_modifiers.append((score, modifiers[i]))

    for word in words:
        for i, mod_desc in enumerate(modifier_descriptions):
            if mod_desc == "air fried":
                continue
            score = fuzz.ratio(mod_desc, word)
            if score >= threshold:
                if modifiers[i] not in [m[1] for m in matched_modifiers]:
                    matched_modifiers.append((score, modifiers[i]))

    matched_modifiers = sorted(matched_modifiers, key=lambda x: x[0], reverse=True)
    return [m[1] for m in matched_modifiers]

def remove_modifier_phrases(text, matched_modifiers):
    text = " " + normalize_text(text) + " "
    for mod in matched_modifiers:
        mod_text = normalize_text(mod["description"])
        pattern = r'\b' + re.escape(mod_text) + r'\b'
        text = re.sub(pattern, ' ', text)
    return normalize_text(text)

def get_best_matching_food(user_input, df):
    normalized_input = normalize_text(user_input)
    input_tokens = set(normalized_input.split())
    choices = df['name'].tolist()

    def has_token_overlap(name):
        name_tokens = set(normalize_text(name).split())
        return len(name_tokens & input_tokens) > 0

    filtered_choices = [name for name in choices if has_token_overlap(name)]

    matches = process.extract(
        normalized_input,
        filtered_choices if filtered_choices else choices,
        scorer=fuzz.token_sort_ratio,
        limit=5
    )

    if matches:
        for match in matches:
            if normalize_text(match[0]) == normalized_input:
                return match[0]
        return matches[0][0]

    return None

def get_food_data(food_name, df):
    matches = df[df['name'].str.lower() == food_name.lower()]
    if not matches.empty:
        return matches.iloc[0]
    return df[df['name'] == food_name].iloc[0] if food_name in df['name'].values else None

def apply_modifiers(food_data, matched_modifiers):
    factors = {"calories": 1, "protein": 1, "carbs": 1, "fat": 1, "sugar": 1}
    for mod in matched_modifiers:
        for key in factors:
            factors[key] *= mod["multipliers"].get(key, 1)

    modified_data = food_data.copy()
    for key, factor in zip(["cals", "protein", "carbs", "fat", "sugar"], factors.values()):
        modified_data[key] *= factor
    return modified_data

def calculate_final_nutrition(user_input, db_path="foods.db"):
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM food_info", conn)
    except sqlite3.Error as e:
        return f"Database connection failed: {e}", []
    finally:
        if conn:
            conn.close()

    matched_modifiers = extract_modifiers_from_text(user_input)
    cleaned_input = remove_modifier_phrases(user_input, matched_modifiers)
    food_name = get_best_matching_food(cleaned_input, df)

    if food_name:
        food_data = get_food_data(food_name, df)
        if food_data is not None:
            return apply_modifiers(food_data, matched_modifiers), [mod["description"] for mod in matched_modifiers]
        return f"Food item '{food_name}' not found in the database.", []
    return f"Food item '{cleaned_input}' not found in the database.", []