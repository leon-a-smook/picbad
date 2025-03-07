"""utils.py

Contains general utilities.
"""

def validate_distribution_dict(params):
    """Checks whether the provided distribution type is supported and all parameters are included."""
    distributions = {
        "gaussian": {"Mn": int, "sigma": float},
        "schulz-zimm": {"Mn": int, "D": float},
        "flory-schulz": {"p": float}
    }

    # Check if type is mentioned
    if "type" not in params:
        raise(ValueError("Missing key: 'type'."))
        return False

    # Check if type is supported
    distribution = distributions.get(params["type"])
    if distribution is None:
        raise(ValueError(f"Unsupported distribution: {params['type']}"))
        return False


    # Check if all required parameters are supplied
    for key, expected_type in distribution.items():
        if key not in params:
            raise(ValueError(f"Missing key: {key} for distribution {params['type']}"))
            return False

        if not isinstance(params[key], expected_type):
            raise(ValueError(f"Incorrect type for {key}. Expected {expected_type}, got {type(params[key])}"))
            return False

    return True

def validate_load_profile_dict(params):
    """Checks whether the provided distribution type is supported and all parameters are included."""
    
    # Check if file is provided
    if "filename" not in params:
        raise(ValueError("Missing key: 'filename'."))
        return False
    return True