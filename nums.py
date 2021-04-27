import re

def evaluate(buf):
  while len(buf) > 1:
    cur = int(buf[0])
    nxt = int(buf[1])

    if cur < nxt: buf[0] = cur * nxt
    else:         buf[0] = cur + nxt

    del buf[1]

  return buf[0]


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
  text = re.sub(r"(\d+)\s*\.\s*(\d+)", r"\1 point \2", text)
  text = re.sub(r"(\d+)\s+/\s+(\d+)", r"\1 out of \2", text)

  # remove non-numerical related punctuation
  text = re.sub(r"[-./,]", r" ", text)

  text = " ".join([str(NUMBERS[token.lower()]) if token.lower() in NUMBERS else token for token in text.split()])
  text = re.sub(r"(\d+?)\s+comma\s+(\d+?)", r"\1\2", text)
  text = re.sub(r"(\d+)\s+and\s+(\d+)", r"\1 \2", text)
  text = re.sub(r"(\d+)\s+of\s+(\d+)", r"\1 / \2", text)
  text = re.sub(r"(\d+)\s+out of\s+(\d+)", r"\1 / \2", text)
  text = re.sub(r"(\d+)\s+point\s+(\d+)", r"\1 . \2", text)

  tokens = text.split()

  buf = []
  com = []

  for token in tokens:
    if token.isdecimal():
      buf.append(token)
      continue

    elif buf:
      combined = str(evaluate(buf))
      com.append(combined)
      buf = []

    com.append(token)

  if buf:
    combined = str(evaluate(buf))
    com.append(combined)


  # resolve decimal and fractions

  res = []
  i = 0

  n = len(com)
  while i < n:
    curr = com[i]
    nxt1 = i + 1 < n and com[i + 1]
    nxt2 = i + 2 < n and com[i + 2]


    if nxt2 and nxt1 == ".":
      if curr.isdecimal() and nxt2.isdecimal():
        res.append(curr + nxt1 + nxt2)
        i += 3
        continue

    if nxt2 and nxt1 == "/":
      if curr.isdecimal() and nxt2.isdecimal():
        res.append(str(int(curr)/int(nxt2)))
        i += 3
        continue

    res.append(curr)
    i += 1

  return " ".join(res)

print(convert_word_to_number("I have eaten 3 twenty three dumplings and 0.8 kiwis and 5.5 mangos and a 0.2 of an almond and 5,000,000 grapes"))