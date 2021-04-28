import re
import spacy
import unidecode

from spacy.tokenizer import Tokenizer
from spellchecker import SpellChecker
from datasets import CONTRACTIONS
from word2number import w2n
from num2words import num2words

spell = SpellChecker()
nlp_space = spacy.load("en_core_web_sm")
nlp_space.tokenizer = Tokenizer(nlp_space.vocab, token_match=re.compile(r'\S+').match)


# ================== #
# ==== Cleaning ==== #
# ================== #

def remove_accents(text):
  return unidecode.unidecode(text)


def remove_punctuation(text):
  # removes all non-alphanumeric characters except ( % | & | $ | . | / | , )
  tokens = re.split(r"[^a-zA-Z0-9'%&./,]", text)
  stripped = [token.strip() for token in tokens]
  filtered = [token for token in tokens if token]
  return " ".join(filtered)


def expand_contractions(text):
  tokens = text.split()

  for token in tokens:
    # if token is a contraction, replace with its expanded form
    if expanded := CONTRACTIONS.get(token.lower()):
      # if contraction is capitalized, change expansion capitalization to match input
      if token[0].isupper(): expanded = expanded[0].upper() + expanded[1:]
      text = text.replace(token, expanded)

  return text


def perform_spell_check(doc):
  checked = []

  for token in doc:
    updated = token.text

    # we don't want to spell-check proper noun and unit shortforms
    if token.pos_ != "PROPN" and len(token.text) > 4:
      if mispelled := spell.unknown(token.text):
        updated = spell.correction(token.text)

    checked.append(updated)

  return " ".join(checked)


def convert_plural_to_singular(doc):
  tokens = []

  for token in doc:
    updated = token.text

    # check if token is a plural noun before lemmatizing
    if token.tag_ == "NNS":
      updated = token.lemma_

    tokens.append(updated)

  return " ".join(tokens)


def handle_units(text):
  return re.sub(r"([\d.]+)([a-zA-Z]+)", r"\1 \2", text) # handles units


# I had 100 mL of milk
# i had a hundred ml of milk
# I had a couple cups of coffee
# I had a fifth of a watermelon