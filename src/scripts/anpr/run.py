from ...pipelines.bronze_pipeline import BronzePipeline
from ... import config as app_config

import json
import os

if __name__ == "__main__":
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    
    config = json.load(open(config_path))
    
    # Inject API keys from environment into config
    for dataset in config.get("datasets", []):
        api_key_id = dataset.get("api_key_id")
        if api_key_id:
            dataset["params"]["api_key"] = getattr(app_config, api_key_id, None)
    
    # Initialize the pipeline
    bronze = BronzePipeline()
    # silver = SilverPipeline()
    # gold = GoldPipeline()
    
    # Run the pipeline
    bronze.run(config)

