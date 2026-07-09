import os
from dotenv import load_dotenv

load_dotenv()


KAGGLE_API_KEY = os.getenv("KAGGLE_API_KEY")
VEHICLE_API_KEY = os.getenv("VEHICLE_API_KEY")
GEGEVENS_API_KEY = os.getenv("GEGEVENS_API_KEY")
