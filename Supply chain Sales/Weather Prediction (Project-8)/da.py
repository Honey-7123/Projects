import os

# Define the folders and files
structure = {
    "supply-chain-ai/backend": ["main.py", "models/prophet_model.py", "models/arima_model.py"],
    "supply-chain-ai/dashboard": ["dashboard.py"],
    "supply-chain-ai/data": ["favorita_sample.csv"],
    "supply-chain-ai/db": ["mongo_handler.py"],
    "supply-chain-ai": ["requirements.txt", "README.md"]
}

def create_structure():
    for folder, files in structure.items():
        os.makedirs(folder, exist_ok=True)
        for file in files:
            file_path = os.path.join(folder, file)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                pass  # create empty file

if __name__ == "__main__":
    create_structure()
    print("✅ Folder structure for supply-chain-ai created!")
