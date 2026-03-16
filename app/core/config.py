from dotenv import load_dotenv
import os

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DBT_PROJECT_PATH = os.getenv("DBT_PROJECT_PATH", "./dbt_demo")
MANIFEST_PATH = os.getenv("MANIFEST_PATH", "./dbt_demo/target/manifest.json")
