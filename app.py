import pandas as pd
import sqlite3
import os

def build_db():
    excel_path = 'final_food_data.xlsx'
    db_path = 'foods.db'
    table_name = 'food_info'

    if not os.path.exists(excel_path):
        print(f"Error: Source file not found at '{excel_path}'")
        print("Please make sure the final dataset exists before running this script.")
        return

    try:
        print(f"Reading data from '{excel_path}'...")
        df = pd.read_excel(excel_path)

        print(f"Connecting to or creating database at '{db_path}'...")
        conn = sqlite3.connect(db_path)
        
        dtype_mapping = {
            'name': 'TEXT PRIMARY KEY',
            'cals': 'INTEGER',
            'carbs': 'INTEGER',
            'protein': 'INTEGER',
            'fat': 'INTEGER',
            'sugar': 'INTEGER'
        }

        print(f"Writing data to table '{table_name}'...")
        df.to_sql(table_name, conn, if_exists='replace', index=False, dtype=dtype_mapping)

        print("Database build successful.")
        
        print("\nVerifying database content...")
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"Table '{table_name}' contains {count} rows.")
        
        print("\nFirst 5 rows:")
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
        for row in cursor.fetchall():
            print(row)

    except Exception as e:
        print(f"An error occurred during database creation: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()
            print("\nDatabase connection closed.")



import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import io
from nlp_matching import calculate_final_nutrition

DB_FILE = "foods.db"
if not os.path.exists(DB_FILE):
    st.info("First time setup: Building the food database... Please wait.")
    build_db()
    st.success("Database is ready!")
    st.rerun()

st.set_page_config(
    page_title="Nutri-Vision AI",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
        /* Main background and text colors */
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        /* Reduce top margin */
        div.block-container {
            padding-top: 2rem;
        }
        /* Main title */
        h1 {
            color: #00C4B4; 
        }
        /* Section headers */
        h3 {
            color: #FAFAFA;
        }
        label[data-testid="stWidgetLabel"] {
            color: #FAFAFA !important;
        }
        /* Buttons */
        .stButton>button {
            border: 2px solid #00C4B4;
            background-color: transparent;
            color: #00C4B4;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            transition: all 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #00C4B4;
            color: #0E1117;
            border-color: #00C4B4;
        }
        /* Nutrient card styling */
        .nutrient-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
        .nutrient-card {
            background-color: #262730;
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
            border-left: 5px solid #00C4B4;
        }
        .nutrient-label {
            font-size: 0.9rem;
            color: #A0A0A5;
            margin-bottom: 0.5rem;
        }
        .nutrient-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #FAFAFA;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Loads the pre-trained Keras model."""
    try:
        model = tf.keras.models.load_model('final_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model 'final_model.h5': {e}")
        return None

CLASS_NAMES = [
    'bean nachos', 'biryani', 'burger', 'chaat', 'chhole bhature', 'chhole chawal', 'dhokla', 'donuts', 'dosa', 'french fries', 'fried chicken', 'fried rice', 'gulab jamun', 'halwa', 'idli', 'jalebi', 'kadhi chawal', 'kheer', 'khichdi', 'kulfi', 'litti chokha', 'momos', 'noodles', 'pakore', 'paneer tikka', 'paratha', 'pav bhaji', 'pizza', 'poha', 'rajma chawal', 'rasmalai', 'red sauce pasta', 'samosa', 'sandwich', 'shawarma', 'spring roll', 'tomato soup', 'upma', 'vada pav', 'white sauce pasta']
CONFIDENCE_THRESHOLD = 0.04
model = load_model()

def display_results(nutrition_result, applied_modifiers, food_name):
    """Renders the nutritional information in a modern card layout."""
    if isinstance(nutrition_result, pd.Series):
        st.subheader("Estimated Nutritional Information")
        
        if applied_modifiers:
            st.info(f"Applied modifiers: **{', '.join(applied_modifiers)}**")

        st.markdown(f"""
            <div class="nutrient-grid">
                <div class="nutrient-card">
                    <div class="nutrient-label">Calories</div>
                    <div class="nutrient-value">{nutrition_result['cals']:.0f}</div>
                </div>
                <div class="nutrient-card">
                    <div class="nutrient-label">Carbs (g)</div>
                    <div class="nutrient-value">{nutrition_result['carbs']:.1f}</div>
                </div>
                <div class="nutrient-card">
                    <div class="nutrient-label">Protein (g)</div>
                    <div class="nutrient-value">{nutrition_result['protein']:.1f}</div>
                </div>
                <div class="nutrient-card">
                    <div class="nutrient-label">Fat (g)</div>
                    <div class="nutrient-value">{nutrition_result['fat']:.1f}</div>
                </div>
                <div class="nutrient-card">
                    <div class="nutrient-label">Sugar (g)</div>
                    <div class="nutrient-value">{nutrition_result['sugar']:.1f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.error(f"Could not find nutritional information for '{food_name}'. Please try a different description.")


st.title("🥗 Nutri-Vision AI")
st.markdown("Upload an image or just type a description to get an estimate of its nutritional content.")

col1, col2 = st.columns([0.9, 1.1])

with col1:
    st.header("Your Meal")
    uploaded_file = st.file_uploader("Upload a food image (optional)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        image.thumbnail((350, 350)) 
        st.image(image, caption='Uploaded Image')

    text_input = st.text_input(
        "Add food name and details (e.g., 'large homemade pasta', 'fried samosa')",
        help="Provide details like portion size, cooking style, or extra ingredients."
    )
    analyze_button = st.button("Analyze My Meal", use_container_width=True)

with col2:
    st.markdown("<h2 style='text-align: center; color: #00F0D0;'>Nutritional Analysis</h2>", unsafe_allow_html=True)
    
    results_placeholder = st.container()

    if analyze_button:
        food_to_search = ""
        if uploaded_file is not None:
            with st.spinner('Analyzing your meal...'):
                original_image = Image.open(uploaded_file).convert('RGB')
                image_resized_for_model = original_image.resize((224, 224))
                image_array = tf.keras.preprocessing.image.img_to_array(image_resized_for_model)
                image_array = np.expand_dims(image_array, axis=0)

                if model:
                    predictions = model.predict(image_array)
                    score = tf.nn.softmax(predictions[0])
                    confidence = np.max(score)
                    
                    if confidence >= CONFIDENCE_THRESHOLD:
                        predicted_class = CLASS_NAMES[np.argmax(score)]
                        results_placeholder.subheader(f"Image recognized as: **{predicted_class.title()}**")
                        food_to_search = f"{text_input} {predicted_class}"
                    else:
                        results_placeholder.warning("Image confidence low. Relying on text description.")
                        if not text_input:
                            results_placeholder.error("Please provide a text description.")
                            st.stop()
                        food_to_search = text_input
                else:
                    results_placeholder.error("Model not loaded. Relying on text input only.")
                    food_to_search = text_input
        
        elif text_input:
            with st.spinner('Analyzing your meal...'):
                results_placeholder.subheader("Analysis based on text description")
                food_to_search = text_input

        else:
            results_placeholder.warning("Please upload an image or provide a text description.")
            st.stop()
        if food_to_search:
            nutrition_result, applied_modifiers = calculate_final_nutrition(food_to_search)
            results_placeholder.divider()
            with results_placeholder:
                display_results(nutrition_result, applied_modifiers, food_to_search)

    else:
        results_placeholder.info("Upload an image or enter a description and click 'Analyze'.")