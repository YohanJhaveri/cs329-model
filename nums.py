import re

def find_ceil(n):
  for multiple in [100, 1000, 1000000, 1000000000, 1000000000000]:
    if n // multiple == 0:
      return multiple
  return 1

def evaluate(buf):
  print(buf)
  while len(buf) > 1:
    cur = float(buf[0])
    nxt = float(buf[1])

    if 0 <= cur <= 9 and nxt % 1 != 0:
      buf[0] = cur * nxt

    if nxt in {100, 1000, 1000000, 1000000000, 1000000000000}:
      if len(buf) > 3:
        buf[2] = evaluate(buf[2:])
        del buf[3:]


    if cur < nxt:
      if nxt not in {100, 1000, 1000000, 1000000000, 1000000000000}:
        buf[0] = cur * find_ceil(nxt)
        continue
      else:
        buf[0] = cur * nxt

    else:
      buf[0] = cur + nxt

    del buf[1]

  return int(buf[0])


def combine_numbers(tokens):
  buf = []
  combined = []

  for token in tokens:
    if token.isdecimal() and int(token) > 0:
      buf.append(token)
      continue

    elif buf:
      combined.append(str(evaluate(buf)))
      buf = []

    combined.append(token)

  if buf:
    combined.append(str(evaluate(buf)))

  return combined


def combine_symbols(tokens):
  combined = []

  for token in tokens:
    if m := re.match(r"([0-9]+)/([0-9]+)", token):
        p = int(m.group(0))
        q = int(m.group(1))
        combined.append(str(p / q))
        continue

    combined.append(token)

  return combined



def convert_word_to_number(text):
  NUMBERS = {
    "tenth": 1/10,
    "ninth": 1/9,
    "eigth": 1/8,
    "seventh": 1/7,
    "sixth": 1/6,
    "fifth": 1/5,
    "quarter": 1/4,
    "third": 1/3,
    "half": 1/2,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "fourty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
    "thousand": 1000,
    "million": 1000000,
    "billion": 1000000000,
    "trillion": 1000000000000
  }

  # convert symbols to words so they don't get removed when cleaning sentence puntuation
  text = re.sub(r"(\d+?),(\d+?)", r"\1 comma \2", text)
  text = re.sub(r"(\d+)\.(\d+)", r"\1 point \2", text)
  text = re.sub(r"\.(\d+)", r" point \1", text)
  text = re.sub(r"(\d+)\s+/\s+(\d+)", r"\1 out of \2", text)

  # split at hyphens
  tokens = []
  for token in text.split():
    if "-" in token:
      words = token.split("-")
      numbers = [NUMBERS.get(re.sub(r"[-./,]", r"", word.lower()), False) for word in words]
      if numbers.count(False) == 0:
        tokens.extend(words)
        continue

    tokens.append(token)

  text = " ".join(tokens)

  text = " ".join([str(NUMBERS.get(re.sub(r"[-./,]", r"", token.lower()), token)) for token in text.split()])
  text = re.sub(r"(\d+?)\s+comma\s+(\d+?)", r"\1\2", text)
  text = re.sub(r"(\d+)\s+and\s+(\d+)", r"\1 \2", text)
  text = re.sub(r"(\d+)\s+of\s+(\d+)", r"\1/\2", text)
  text = re.sub(r"(\d+)\s+out of\s+(\d+)", r"\1/\2", text)
  text = re.sub(r"(\d+)\s+point\s+(\d+)", r"\1.\2", text)
  text = re.sub(r"(?:\b|\s+)point\s+(\d+)", r"0.\1", text)

  tokens = text.split()
  tokens = combine_numbers(tokens)
  tokens = combine_symbols(tokens)
  return " ".join(tokens)

# ======== TESTING ======== #
word_to_num = convert_word_to_number

# Rules

# remove out of (\d+) for our input



# if "a" is before a multiple {100, 1000, 1000000, 1000000000, 1000000000000} or "couple", convert to 1
# if unit {1, 2, 3, 4, 5, 6, 7, 8, 9} is before a fraction {1/2, 1/3, 1/4...}, multiply both
# except for tens and ones, implicit

# four fifth => 0.8

# a hundred => 100
# one hundred => 100
# hundred => 100

# twenty three hundred => 2300

# (hyphen)
# sixty-five => 65

# couple => 2
# a couple => 2

# tenth of a hundred => 10
# 5,000,000 => 5000000
# three sixty five => 365

# DECIMALS
# <num> point <num>
# four point five => 4.5
assert word_to_num("four point five") == "4.5"
assert word_to_num("4 point five") == "4.5"
assert word_to_num("four point 5") == "4.5"
assert word_to_num("4 point 5") == "4.5"

# point <num>
# point three => 0.3
assert word_to_num("point three") == "0.3"
assert word_to_num("point 3") == "0.3"
assert word_to_num(". three") == ". 3"
assert word_to_num(". 3") == ". 3"


# HYPHEN
assert word_to_num("I, along with my friend, went to Chick-Fil-A at two twenty-three") == "I, along with my friend, went to Chick-Fil-A at 223"
assert word_to_num("two-hundred") == "200"
assert word_to_num("twenty-five-hundred") == "2500"
assert word_to_num("twenty-two-million, three-hundred-fifty-five") == "22000355"

# COMMAS
assert word_to_num("5,000,000") == "5000000"
assert word_to_num("500,000") == "500000"
assert word_to_num("500, 40") == "500, 40"
assert word_to_num("50, 100") == "50, 100"
assert word_to_num("3,423") == "3423"

# MISC
print(word_to_num("four fifth"))
assert word_to_num("five fifty five") == "555"
assert word_to_num("four fifth") == "0.8"