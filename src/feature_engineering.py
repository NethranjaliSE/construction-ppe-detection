"""Label design: map the dataset's 25 classes to the paper's 10 classes."""

# Final classes, in this order = class IDs 0..9
TARGET_CLASSES = ["Hardhat", "Machinery", "NO-Hardhat", "Mask", "Safety Cone",
                  "NO-Mask", "Safety Vest", "Vehicle", "NO-Safety Vest", "Person"]

# dataset class name -> final class name (None = drop)
CLASS_MAP = {
    "Hardhat": "Hardhat", "NO-Hardhat": "NO-Hardhat",
    "Mask": "Mask", "NO-Mask": "NO-Mask",
    "Safety Vest": "Safety Vest", "NO-Safety Vest": "NO-Safety Vest",
    "Person": "Person", "Safety Cone": "Safety Cone",
    # Machinery: heavy site equipment
    "Excavator": "Machinery", "dumptruck": "Machinery",
    "wheel loader": "Machinery", "machinery": "Machinery",
    # Vehicle: road vehicles
    "vehicle": "Vehicle", "sedan": "Vehicle", "truck": "Vehicle", "van": "Vehicle",
    "SUV": "Vehicle", "trailer": "Vehicle", "mini-van": "Vehicle", "semi": "Vehicle",
    "truck and trailer": "Vehicle", "bus": "Vehicle",
    # Not in the paper
    "Gloves": None, "Ladder": None, "fire hydrant": None,
}


def _key(name):
    """Compare names ignoring case and spaces ('dump truck' == 'dumptruck')."""
    return name.lower().replace(" ", "")


def build_id_map(dataset_names):
    """old class id -> new class id (or None). Fails loudly if a name is missing."""
    lookup = {_key(k): v for k, v in CLASS_MAP.items()}
    id_map = {}
    for old_id, name in enumerate(dataset_names):
        if _key(name) not in lookup:
            raise KeyError(f"Class '{name}' is not in CLASS_MAP - add it.")
        target = lookup[_key(name)]
        id_map[old_id] = TARGET_CLASSES.index(target) if target else None
    return id_map