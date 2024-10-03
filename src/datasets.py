import json
import pandas as pd
from typing import Set, Dict, TextIO, Any
from pandas import DataFrame

class FileReader:
    @staticmethod
    def import_txt(file: TextIO) -> Set[str]:
        return set(file.read().split('\n'))
    
    @staticmethod
    def import_json(file: TextIO) -> Dict[str, Any]:
        return json.load(file)
    
    @staticmethod
    def import_csv(path: str) -> DataFrame:
        return pd.read_csv(path)
    

# PATHS
directory = "./data"

def get_file_path(name: str) -> str:
    return f"{directory}/{name}"

paths = {
    "foods": get_file_path("foods.txt"),
    "brands": get_file_path("brands.txt"),
    "numbers": get_file_path("numbers.json"),
    "generated": get_file_path("generated.json"),
    "conversions": get_file_path("conversions.json"),
    "contractions": get_file_path("contractions.json"),
    "nutrients":  get_file_path("nutrients.csv")
}


with open(paths["foods"]) as file_foods, \
     open(paths["brands"]) as file_brands, \
     open(paths["numbers"]) as file_numbers, \
     open(paths["generated"]) as file_generated, \
     open(paths["conversions"]) as file_conversions, \
     open(paths["contractions"]) as file_contractions:

    # Import text files
    FOODS: Set[str] = FileReader.import_txt(file_foods)
    BRANDS: Set[str] = FileReader.import_txt(file_brands)
    
    # Import JSON files
    NUMBERS: Dict[str, Any] = FileReader.import_json(file_numbers)
    GENERATED: Dict[str, Any] = FileReader.import_json(file_generated)
    CONVERSIONS: Dict[str, Any] = FileReader.import_json(file_conversions)
    CONTRACTIONS: Dict[str, Any] = FileReader.import_json(file_contractions)

# Import CSV file
NUTRIENTS: DataFrame = FileReader.import_csv(paths["nutrients"])