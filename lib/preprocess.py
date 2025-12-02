import re
import html
import string
import unicodedata
import os
from datasets import load_dataset, concatenate_datasets, load_from_disk, Dataset
# import unidecode

from tokenizers import Tokenizer, normalizers, decoders
from tokenizers.trainers import WordPieceTrainer
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFC, Lowercase
from tokenizers.pre_tokenizers import Whitespace
# from tokenizers.processors import TemplateProcessing

def load_datasets(path_datasets, save_folder, lang_src, lang_tgt,test_prob=0.1):
    raw_datasets = []
    for dataset_dir in os.listdir(path_datasets):
      path_to_dir = os.path.join(path_datasets, dataset_dir)
      if os.path.isdir(path_to_dir):
        # processs
        src_text = tgt_text = None
        for txt in os.listdir(path_to_dir):
          if txt.endswith(f".{lang_src}"):
            src_text = os.path.join(path_to_dir, txt)
          elif txt.endswith(f".{lang_tgt}"):
            tgt_text = os.path.join(path_to_dir, txt)
        if src_text != None and tgt_text != None:
          src_dataset = load_dataset("text", data_files=src_text)["train"]
          tgt_dataset = load_dataset("text", data_files=tgt_text)["train"]
          src_dataset = src_dataset.rename_column("text", lang_src) # type: ignore
          tgt_dataset = tgt_dataset.rename_column("text", lang_tgt) # type: ignore
          combine = concatenate_datasets([src_dataset, tgt_dataset], axis=1) # type: ignore
          raw_datasets.append(combine)
    dataset = concatenate_datasets(raw_datasets)
    if test_prob > 0:
      dataset = dataset.train_test_split(test_size=test_prob)
    # saving
    dataset.save_to_disk(save_folder)

class Preprocessing:
  def __init__(self, lang_src, lang_tgt):
    self.re_handler = re.compile(r'@\w+')
    self.re_html_tag = re.compile(r'<[^>]+>')
    self.re_html_entity = re.compile(r'&[a-zA-Z]+;')
    self.re_email = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w{2,}\b', flags=re.UNICODE)
    self.re_url = re.compile(
    r'((http|https):\/\/[^\s]+)|(www\.[^\s]+)|(\b[\w-]+\.(com|net|org|edu|id|gov|io|co|biz|info|me)(\/[^\s]*)?)',
      flags=re.IGNORECASE
    )
    self.lang_src = lang_src
    self.lang_tgt = lang_tgt
    # pass
  def save_preprocess(self, save_folder="preprocess_dataset"):
    self.dataset.save_to_disk(save_folder)
  def preprocessing(self, dataset, num_proc=0):
    def batch_examples(examples):
      src_texts = [self.clean_text(e) for e in examples[self.lang_src]]
      tgt_texts = [self.clean_text(e) for e in examples[self.lang_tgt]]
      return {
        self.lang_src:src_texts,
        self.lang_tgt:tgt_texts
      }
    self.dataset = dataset.map(batch_examples, batched=True, num_proc=num_proc)
    return self.dataset
  def clean_text(self, text):
    if not text: return ""
    text = html.unescape(text)
    normalized = unicodedata.normalize('NFKD', text)
    text = ''.join(ch for ch in normalized if not unicodedata.combining(ch))
    text = self.re_html_tag.sub(' ', text)
    text = self.re_html_entity.sub(' ', text)
    text = self.re_email.sub(' ', text)
    text = self.re_url.sub(' ', text)
    text = self.re_handler.sub(' ', text)
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))) # Hati-hati
    text = text.strip().lower()
    return ' '.join(text.split())