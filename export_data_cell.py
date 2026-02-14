# 📤 Data Export Script for FastAPI
# =====================================
# Add this cell to your notebook to export processed data

import os

# Create data directory
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Export processed DataFrame to CSV
output_path = os.path.join(data_dir, "processed_reviews.csv")
df_cleaned.to_csv(output_path, index=False)
print(f"✅ Exported {len(df_cleaned)} reviews to {output_path}")

# Export as pickle (faster loading)
pickle_path = os.path.join(data_dir, "processed_reviews.pkl")
df_cleaned.to_pickle(pickle_path)
print(f"✅ Exported pickle to {pickle_path}")

# Vector store is already persisted at /kaggle/working/restaurant_vector_db
# Copy it to the project directory if needed
import shutil

source_vector_db = "/kaggle/working/restaurant_vector_db"
target_vector_db = "restaurant_vector_db"

if os.path.exists(source_vector_db):
    if os.path.exists(target_vector_db):
        shutil.rmtree(target_vector_db)
    shutil.copytree(source_vector_db, target_vector_db)
    print(f"✅ Copied vector database to {target_vector_db}")
else:
    print("⚠️ Vector database not found at source location")

print("\n🎉 Data export complete! You can now use the FastAPI application.")
print("\nNext steps:")
print("1. Copy the 'data' and 'restaurant_vector_db' folders to your local project")
print("2. Install dependencies: pip install -r requirements.txt")
print("3. Run the server: python main.py")
