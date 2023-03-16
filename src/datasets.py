import json
import pandas as pd

# PATHS
path_foods = "./data/foods.txt"
path_brands = "./data/brands.txt"
path_conversions = "./data/conversions.json"
path_contractions = "./data/contractions.json"
path_generated = "./data/generated.json"
path_nutrients = "./data/nutrients.csv"

# FILES
file_foods = open(path_foods)
file_brands = open(path_brands)
file_conversions = open(path_conversions)
file_contractions = open(path_contractions)
file_generated = open(path_generated)

# .txt files
FOODS = set(file_foods.read().split('\n'))
BRANDS = set(file_brands.read().split('\n'))

# .json files
GENERATED = json.load(file_generated)
CONVERSIONS = json.load(file_conversions)
CONTRACTIONS = json.load(file_contractions)

# .csv files
NUTRIENTS = pd.read_csv(path_nutrients)
