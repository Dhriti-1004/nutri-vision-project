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

if __name__ == '__main__':
    build_db()