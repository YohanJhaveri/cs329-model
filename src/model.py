# IMPORTS
import re
import spacy
from spacy.tokenizer import Tokenizer
from prettytable import PrettyTable

# CORE DATA
from datasets import FOODS, BRANDS, CONVERSIONS, NUTRIENTS
from nums import convert_word_to_number
from utils import convert_plural_to_singular, perform_spell_check, expand_contractions, remove_punctuation, remove_accents, handle_units

# DERIVED DATA
VOLUME = CONVERSIONS["volume_to_milliliter"]
MASS = CONVERSIONS["mass_to_gram"]
ALIAS = CONVERSIONS["alias"]
EXTRA = CONVERSIONS["extra"]
UNITS = list(VOLUME) + list(MASS) + list(ALIAS) + EXTRA
BRANDS_LOWER = [brand.lower() for brand in BRANDS]

# INITIALIZE
nlp = spacy.load("en_core_web_sm")
nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)
nlp2 = spacy.load("en_core_web_sm")


def clean(text):
  text = remove_accents(text)
  text = expand_contractions(text)
  text = handle_units(text)
  text = convert_word_to_number(text)
  text = remove_punctuation(text)
  doc = nlp(text)
  text = perform_spell_check(doc)
  doc = nlp(text)
  text = convert_plural_to_singular(doc)

  return text


# UTILS
def edit_distance(s, t):
  if s == "": return len(t)
  if t == "": return len(s)
  rs = s[:-1]
  rt = t[:-1]
  cost = s[-1] != t[-1]

  return min([
    edit_distance(rs, t) + 1,
    edit_distance(s, rt) + 1,
    edit_distance(rs, rt) + cost
  ])


def is_unit(token):
  token = token.lemma_.lower()
  if token in UNITS: return True
  elif token[-1] == "s" and token[:-1] in UNITS: return True
  elif token[-1] == "." and token[:-1] in UNITS: return True
  elif token[-2:] == "s." and token[:-2] in UNITS: return True

  return False


def is_brand(text):
  # brand = next((brand for brand in BRAND if edit_distance(brand, text) < (0.2 * len(text))), None)
  # return not not brand
  without_punct = " ".join(re.split(r"[^a-zA-Z0-9]", text))
  return (
    text in BRANDS
    or text.lower() in BRANDS_LOWER
    or without_punct in BRANDS
    or without_punct.lower() in BRANDS_LOWER
  )


def read_gazetteers():
  data = {}
  sets = {
    "UNIT": list(VOLUME) + list(MASS) + list(ALIAS) + EXTRA,
    "FOOD": FOODS,
    "BRAND": BRANDS
  }

  for s in sets:
    for x in sets[s]:
      labels = data.get(x, set())
      labels.add(s)
      data[x] = labels

  return data



def recognize_ngram(tokens, gazetteer):
  entities = []
  doc = nlp(" ".join(tokens))
  for i in range(len(tokens)):
    for j in range(i+1, len(tokens)+1):
      key = ' '.join(tokens[i:j])

      val = gazetteer.get(key.lower(), None)
      if val:
        entities.append((key, i, j, val))
        continue


      quantity = ((j - i) == 1) and re.match("^[0-9\.]+$", key)
      if quantity:
        entities.append((key, i, j, {"QUANTITY"}))
        continue


      # checks if unit
      unit = ((j - i) == 1) and is_unit(doc[i])
      if unit:
        entities.append((key, i, j, {"UNIT"}))
        continue


       # checks if brand
      brand = is_brand(key)
      if brand:
        entities.append((key, i, j, {"BRAND"}))
        continue


  return entities


def remove_overlaps(entities):
  entities.sort(key=lambda x: x[2])
  sublists = []

  def aux(end, sub):
    for entity in entities:
      s = entity[1]
      e = entity[2]

      if end <= s:
        aux(e, sub + [entity])
      else:
        sublists.append(sub)

  aux(-1, [])

  keys = {(sum([entity[2] - entity[1] for entity in sublist]), -len(sublist)): sublist for sublist in sublists}
  key = sorted(keys.keys())[-1]

  return keys[key]


def combine_food_entities(entities):
  combined = []
  index = 1
  length = len(entities)

  while index < length:
    prev = entities[index - 1]
    curr = entities[index]

    if "FOOD" in prev[3] and "FOOD" in curr[3] and prev[2] == curr[1]:
      combined.append((prev[0] + " " + curr[0], prev[1], curr[2], {"FOOD"}))
      index += 2
    else:
      combined.append(prev)
      index += 1

  if index == length:
    combined.append(entities[-1])

  return combined


def find_entities(text):
  text = clean(text)
  tokens = text.split()

  entities = recognize_ngram(tokens, read_gazetteers())
  if entities: entities = remove_overlaps(entities)
  if entities: entities = combine_food_entities(entities)
  return entities


def parse_entities(text):
  def get_unit(pre, nxt):
    if nxt and "UNIT" in nxt[3]: return nxt
    if pre and "UNIT" in pre[3]: return pre

  def get_quantity(pre, nxt):
    if nxt and "QUANTITY" in nxt[3]: return nxt
    if pre and "QUANTITY" in pre[3]: return pre

  def get_brand(pre, nxt):
    if nxt and "BRAND" in nxt[3]: return nxt
    if pre and "BRAND" in pre[3]: return pre

  entities = [entity for entity in find_entities(text) if "BRAND" not in entity[3]][::-1]
  food_entities = [entity for entity in entities if "FOOD" in entity[3]]
  tokens = text.split()
  items = []

  for cur in food_entities:
    i = entities.index(cur)
    n = len(entities)

    item = {}

    pre = i - 1 >= 0 and entities[i - 1]
    nxt = i + 1 < n and entities[i + 1]

    def dec_pre(pre):
      idx = entities.index(pre) - 1
      if idx >= 0: return entities[idx]

    def inc_nxt(nxt):
      idx = entities.index(nxt) + 1
      if idx < n: return entities[idx]


    # ==== UNITS ==== #
    unit = get_unit(pre, nxt)
    if unit:
      item["unit"] = unit[0]
      if pre == unit: pre = dec_pre(pre)
      if nxt == unit: nxt = inc_nxt(nxt)
    else:
      item["unit"] = "count"

    # ==== BRAND ==== #
    brand = get_brand(pre, nxt)
    if brand:
      if pre == brand: pre = dec_pre(pre)
      if nxt == brand: nxt = inc_nxt(nxt)

    # ==== FOOD ==== #
    food = cur

    # ==== QUANTITY ==== #
    quantity = get_quantity(pre, nxt)
    quantityFromList = not not quantity

    if quantity:
      item["quantity"] = float(quantity[0])

    else:
      prior = i - 1 >= 0 and entities[i - 1]
      start = prior[2] if prior else 0
      pattern = r"[\b\s]an?[\b\s]"

      if unit:
        end = unit[1]
        found = re.search(pattern, " ".join(tokens[start:end+1]).lower())
        if found: item["quantity"] = 1

      if food:
        end = food[1]
        found = re.search(pattern, " ".join(tokens[start:end+1]).lower())
        if found: item["quantity"] = 1

    if not item.get("quantity", None): item["quantity"] = 0

    item["food"] = ((brand[0] + " ") if brand else "") + food[0]

    if food:
      index = entities.index(food)
      del entities[index]

    if brand:
      index = entities.index(brand)
      del entities[index]

    if quantityFromList:
      index = entities.index(quantity)
      del entities[index]

    if unit:
      index = entities.index(unit)
      del entities[index]


    items.append(item)

  return items


def f1_score():
  correct = 0
  total = 0
  for i in data:
    text = i["input"]
    entities = i["entities"]

    true = set()
    pred = set()

    for entity in entities:
      true.add((entity["quantity"], entity["unit"], entity["food"]))

    for entity in parse_entities(text):
      pred.add((entity["quantity"], entity["unit"], entity["food"]))

    correct += len(true.intersection(pred))
    total += len(true)

  print()
  print("{} / {}".format(correct, total))
  print()



def match_item(food):
  names = NUTRIENTS["Name"]
  match_counts = list()

  food_tokens = food.lower().split()

  for name in names:
    name_tokens = name.lower().split()
    match_count = 0

    for i1, t1 in enumerate(food_tokens):
      index = next((i2 for i2, t2 in enumerate(name_tokens) if t1 == t2), None)

      if index != None:
        name_tokens.pop(index)
        match_count += 1

      match_counts.append((match_count, -len(name), name))

  match = max(match_counts)[2] if match_counts else None

  if match:
    return NUTRIENTS[NUTRIENTS["Name"] == match].iloc[0]
  else:
    return None



def clean_unit(unit):
  unit = unit.lower()
  if unit in ALIAS: return ALIAS[unit]
  elif unit[-1] == "s" and unit[:-1] in ALIAS: return ALIAS[unit[:-1]]
  elif unit[-1] == "." and unit[:-1] in ALIAS: return ALIAS[unit[:-1]]
  elif unit[-2:] == "s." and unit[:-2] in ALIAS: return ALIAS[unit[:-2]]
  return unit



def find_ratio(quantity_data, quantity_true, unit_data, unit_true, mass_weight):
  unit_data = clean_unit(unit_data)
  unit_true = clean_unit(unit_true)

  if quantity_true == 0:
    return 0

  if unit_true == unit_data:
      return quantity_true / quantity_data

  if unit_data in VOLUME:
    unit_data_ml = quantity_data * VOLUME[unit_data]
    density = unit_data_ml / mass_weight

    if unit_true in VOLUME:
      unit_true_ml = quantity_true * VOLUME[unit_true]
      mass_true = density * unit_true_ml

      return mass_true / 100

  if unit_true in MASS:
    unit_true_g = quantity_true * MASS[unit_true]
    return unit_true_g / 100

  return 1




def calculate_nutrients(foods):
  nutrients = {
    "fat": 0,
    "protein": 0,
    "carbohydrate": 0
  }

  for food in foods:
    details = match_item(food["food"])

    if details.any():
      split = str(details["Volume"]).split()

      quantity_data = float(split[0])
      quantity_true = float(food["quantity"])

      unit_data = " ".join(split[1:]).strip()
      unit_true = str(food["unit"])

      ratio = find_ratio(quantity_data, quantity_true, unit_data, unit_true, details["Mass(g)"])

      nutrients["fat"] += float(details["Fat"]) * ratio
      nutrients["protein"] += float(details["Protein"]) * ratio
      nutrients["carbohydrate"] += float(details["Carbohydrate"]) * ratio

  return nutrients



while True:
  text = input("\nPlease enter what food you had today or enter (q) to quit: \n")

  if text == 'q' or text == 'Q':
    break

  else:
    print()
    print()
    print("============ TABLE OF ENTITIES ============")
    entities = find_entities(text)
    entity_table = PrettyTable(['Span', 'Start', 'End', 'Entity'])
    for entity in entities: entity_table.add_row(entity)
    print(entity_table)


    print()
    print()
    print("============ TABLE OF FOODS ============")
    foods = parse_entities(text)
    food_table = PrettyTable(['Quantity', 'Unit', 'Food'])
    for food in foods: food_table.add_row((food.get("quantity", "?"), food.get("unit", "count"), food.get("food", None)))
    print(food_table)
    print()


    print()
    print()
    print("============ TABLE OF NUTRIENTS ============")
    nutrients = calculate_nutrients(foods)
    nutrient_table = PrettyTable(['Nutrient', 'Quantity (g)'])
    for nutrient in nutrients: nutrient_table.add_row((nutrient.upper(), round(nutrients[nutrient], 3)))
    print(nutrient_table)
    print()
