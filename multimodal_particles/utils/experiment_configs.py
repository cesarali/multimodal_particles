
import yaml
import random
from datetime import datetime

import yaml
from types import SimpleNamespace
from dataclasses import is_dataclass

def namespace_to_dict(obj):
    """
    Recursively converts a SimpleNamespace or dataclass object (or a list of these) back into a dictionary (or a list of dictionaries).
    """
    if is_dataclass(obj):
        return {k: namespace_to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [namespace_to_dict(item) for item in obj]
    else:
        return obj

def dict_to_yaml(dict_obj, output_path):
    """
    Converts a dictionary to a YAML string and optionally saves it to a file.
    
    Parameters:
        dict_obj (dict): The dictionary to convert to YAML.
        output_path (str): The path to the file where the YAML string should be saved.
        
    Returns:
        str: The YAML string representation of the dictionary.
    """
    yaml_str = yaml.dump(dict_obj, default_flow_style=False)
    with open(output_path, 'w') as f:
        f.write(yaml_str)
    return yaml_str

def yaml_to_namespace(data):
    """
    Recursively converts a dictionary into a nested SimpleNamespace object.
    """
    if isinstance(data, dict):
        return SimpleNamespace(**{k: yaml_to_namespace(v) for k, v in data.items()})
    elif isinstance(data, list):
        return [yaml_to_namespace(item) for item in data]
    else:
        return data

def load_config(config_source):
    """
    Loads the YAML configuration from a file or dictionary and converts it to a nested object.
    
    Parameters:
        config_source (str | dict): Path to the YAML file or a dictionary representing the configuration.
    
    Returns:
        SimpleNamespace: Nested namespace object representing the configuration.
    """
    if isinstance(config_source, str):
        # Read from file
        with open(config_source, "r") as f:
            config_dict = yaml.safe_load(f)
    elif isinstance(config_source, dict):
        # Directly use provided dictionary
        config_dict = config_source
    else:
        raise ValueError("config_source must be a file path (str) or a dictionary")

    return yaml_to_namespace(config_dict)

class Configs:
    def __init__(self, config_source):
        if isinstance(config_source, str):
            with open(config_source, "r") as f:
                config_dict = yaml.safe_load(f)
        elif isinstance(config_source, dict):
            config_dict = config_source
        else:
            raise ValueError("config_source must be a file path or a dictionary")
        
        self.dynamics = None
        self.experiment = None
        self.data = None
        self.pipeline = None

        self.config_dict = config_dict
        self._set_attributes(config_dict)  # set attributes recursively

        if hasattr(self, "experiment"):
            if not hasattr(self.experiment, "type"):
                if hasattr(self.dynamics, "discrete") and not hasattr(
                    self.dynamics, "continuous"
                ):
                    self.experiment.type = "discrete"
                    assert self.data.dim.features_discrete > 0
                    self.data.dim.features_continuous = 0

                elif hasattr(self.dynamics, "continuous") and not hasattr(
                    self.dynamics, "discrete"
                ):
                    assert self.data.dim.features_continuous > 0
                    self.experiment.type = "continuous"
                    self.data.dim.features_discrete = 0

                else:
                    self.experiment.type = "multimodal"
                    assert (
                        self.data.dim.features_continuous > 0
                        and self.data.dim.features_discrete > 0
                    )

            if not hasattr(self.experiment, "name"):
                self.experiment.name = (
                    f"{self.data.source.name}_to_{self.data.target.name}"
                )

                if hasattr(self.dynamics, "continuous"):
                    self.experiment.name += f"_{self.dynamics.continuous.bridge}"

                if hasattr(self.dynamics, "discrete"):
                    self.experiment.name += f"_{self.dynamics.discrete.bridge}"

                self.experiment.name += f"_{self.model.name}"

                time = datetime.now().strftime("%Y.%m.%d_%Hh%M")
                rnd = random.randint(0, 10000)
                self.experiment.name += f"_{time}_{rnd}"
                print(
                    "INFO: created experiment instance {}".format(self.experiment.name)
                )

            if self.experiment.type == "classifier":
                if len(self.data.train.path) > 1:
                    self.experiment.name = "multi_model"
                else:
                    self.experiment.name = "binary"

    def _set_attributes(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):  # create a sub-config object
                sub_config = Configs(value)
                setattr(self, key, sub_config)
            else:
                setattr(self, key, value)

    def to_dict(self):
        """
        Recursively converts the Configs object into a dictionary.
        """
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Configs):
                config_dict[key] = value.to_dict()
            else:
                config_dict[key] = value
        return config_dict

    def print(self):
        """
        Prints the configuration parameters in a structured format.
        """
        config_dict = self.to_dict()
        self._print_dict(config_dict)

    def _print_dict(self, config_dict, indent=0):
        """
        Helper method to recursively print the config dictionary.
        """
        for key, value in config_dict.items():
            prefix = " " * indent
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                self._print_dict(value, indent + 4)
            else:
                print(f"{prefix}{key}: {value}")

    def log_config(self, logger):
        """
        Logs the configuration parameters using the provided logger.
        """
        config_dict = self.to_dict()
        self._log_dict(config_dict, logger)

    def _log_dict(self, config_dict, logger, indent=0):
        """
        Helper method to recursively log the config dictionary.
        """
        for key, value in config_dict.items():
            prefix = " " * indent
            if isinstance(value, dict):
                logger.logfile.info(f"{prefix}{key}:")
                self._log_dict(value, logger, indent + 4)
            else:
                logger.logfile.info(f"{prefix}{key}: {value}")

    def save(self, path):
        """
        Saves the configuration parameters to a YAML file.
        """
        config_dict = self.to_dict()
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
