# webui/config_manager.py
import toml
from pathlib import Path
from typing import Dict, Any, List
from ascii_colors import ASCIIColors, LogLevel

CONFIG_FILE_PATH = Path("config.toml") # In the same directory as main.py or project root

# Define the root for all database storage
DATABASES_ROOT = Path("databases")

DEFAULT_CONFIG = {
    "lollms": {
        "binding_name": "ollama",
        "host_address": "http://localhost:11434",
        "model_name": "mistral-nemo:latest",
        "service_key": None, # Explicitly None if not set
    },
    "safestore": {
        # General settings, db_file and doc_dir are now per-database
        "default_vectorizer": "st:all-MiniLM-L6-v2",
        "chunk_size": 10000,
        "chunk_overlap": 100,
    },
    "graphstore": {
        "fusion": {
            "enabled": False # Default to disabled
        }
    },
    "webui": {
        "host": "0.0.0.0",
        "port": 8000,
        "temp_upload_dir": "temp_uploaded_files_webui",
        "log_level": "INFO",
        "active_database_name": "default",
    },
    # Databases are now stored in a standardized structure
    "databases": [
        {
            "name": "default",
            "db_file": str(DATABASES_ROOT / "default" / "default.db"),
            "doc_dir": str(DATABASES_ROOT / "default" / "docs"),
        }
    ]
}

config_data: Dict[str, Any] = {}

def load_config() -> Dict[str, Any]:
    global config_data
    if CONFIG_FILE_PATH.exists():
        ASCIIColors.info(f"Loading configuration from: {CONFIG_FILE_PATH.resolve()}")
        try:
            config_data = toml.load(CONFIG_FILE_PATH)
            # Ensure all sections and keys from default_config exist, fill if not
            for section, defaults in DEFAULT_CONFIG.items():
                if section not in config_data:
                    config_data[section] = defaults
                elif isinstance(defaults, dict):
                    for key, default_value in defaults.items():
                        if isinstance(default_value, dict):
                             if key not in config_data[section]:
                                config_data[section][key] = default_value
                             else:
                                for subkey, sub_default_value in default_value.items():
                                    config_data[section][key].setdefault(subkey, sub_default_value)
                        else:
                            config_data[section].setdefault(key, default_value)


            # Special handling for the list of databases
            if "databases" not in config_data or not config_data["databases"]:
                config_data["databases"] = DEFAULT_CONFIG["databases"]

            # Special handling for None values if toml library omits them
            if config_data.get("lollms") and "service_key" not in config_data["lollms"]:
                config_data["lollms"]["service_key"] = None

        except toml.TomlDecodeError as e:
            ASCIIColors.error(f"Error decoding {CONFIG_FILE_PATH}: {e}. Using default configuration.")
            config_data = DEFAULT_CONFIG.copy() # Use a copy
        except Exception as e:
            ASCIIColors.error(f"Unexpected error loading {CONFIG_FILE_PATH}: {e}. Using default configuration.")
            config_data = DEFAULT_CONFIG.copy()
    else:
        ASCIIColors.warning(f"{CONFIG_FILE_PATH} not found. Creating default configuration file.")
        config_data = DEFAULT_CONFIG.copy()
        save_config(config_data) # Save the default config
    return config_data

def save_config(data_to_save: Dict[str, Any]):
    try:
        with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
            toml.dump(data_to_save, f)
        ASCIIColors.info(f"Configuration saved to: {CONFIG_FILE_PATH.resolve()}")
    except Exception as e:
        ASCIIColors.error(f"Error saving configuration to {CONFIG_FILE_PATH}: {e}")

def get_config() -> Dict[str, Any]:
    global config_data
    if not config_data: # Ensure it's loaded if accessed directly
        load_config()
    return config_data

def get_log_level_from_str(level_str: str) -> LogLevel:
    level_map = {
        "DEBUG": LogLevel.DEBUG,
        "INFO": LogLevel.INFO,
        "WARNING": LogLevel.WARNING,
        "ERROR": LogLevel.ERROR,
        "CRITICAL": LogLevel.CRITICAL,
        "SUCCESS": LogLevel.INFO, # If you use SUCCESS level
    }
    return level_map.get(level_str.upper(), LogLevel.INFO)

# Load config when module is imported
load_config()