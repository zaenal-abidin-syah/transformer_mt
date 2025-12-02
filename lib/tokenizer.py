import os
import glob
import argparse
from tokenizers import Tokenizer
from tokenizers.trainers import WordPieceTrainer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import NFC, Lowercase
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import decoders
from tokenizers.processors import TemplateProcessing

from datasets import load_from_disk

special_token_dicts = {
  "unknown_token": "[UNK]",
  "padding_token": "[PAD]",
  "start_token": "[SOS]",
  "end_token": "[EOS]"
}

def train_tokenizer(path_dataset, format_lang, path_to_save, vocab_size):
  tokenizer = Tokenizer(WordPiece(unk_token="[UNK]")) # type: ignore
  tokenizer.normalizer = normalizers.Sequence([NFC(), Lowercase()]) # type: ignore
  tokenizer.pre_tokenizer = Whitespace() # type: ignore

  dataset = load_from_disk(path_dataset)
  dataset = dataset[format_lang]
  def _get_training_corpus():
    for i in range(0, len(dataset), 1000): # Process in batches of 1000
      yield dataset[i : i + 1000]
  # src_files = glob.glob(os.path.join(path, f"**/*.{format_lang}"))
  # print(src_files)
  trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=list(special_token_dicts.values()))
  # trainer = WordPieceTrainer(vocab_size=50, special_tokens=list(special_token_dicts.values()))
  # tokenizer.train(src_files, trainer=trainer)
  tokenizer.train_from_iterator(_get_training_corpus(), trainer=trainer)
  tokenizer.save(f"{path_to_save}/tokenizer_{format_lang}.json")


class LangTokenizer:
  def __init__(self, path_to_vocab, truncate=True, max_length=64):
    self.path_to_vocab = path_to_vocab
    self.tokenizer = self.prepare_tokenizer()
    self.vocab_size = self.tokenizer.get_vocab_size()
    self.pad_token = self.tokenizer.token_to_id("[PAD]")
    self.unk_token = self.tokenizer.token_to_id("[UNK]")
    self.sos_token = self.tokenizer.token_to_id("[SOS]")
    self.eos_token = self.tokenizer.token_to_id("[EOS]")
    self.post_processor = TemplateProcessing(
      single="[SOS] $A [EOS]",
      special_tokens=[
        ("[SOS]", self.sos_token),
        ("[EOS]", self.eos_token)
      ]
    ) # type: ignore
    self.truncate = truncate
    if self.truncate:
      self.max_length = max_length - self.post_processor.num_special_tokens_to_add(is_pair=False)
  def prepare_tokenizer(self):
    tokenizer = Tokenizer.from_file(self.path_to_vocab)
    tokenizer.decoder = decoders.WordPiece()
    return tokenizer
  def encode(self, input):
    def _process_tokenized(tokenized):
      if self.truncate:
        tokenized.truncate(self.max_length, direction="right")
      tokenized = self.post_processor.process(tokenized)
      return tokenized.ids

    if isinstance(input, str):
      tokenized = self.tokenizer.encode(input)
      tokenized = _process_tokenized(tokenized)
    elif isinstance(input, list):
      tokenized = self.tokenizer.encode_batch(input)
      tokenized = [_process_tokenized(t) for t in tokenized]
    else:
        raise TypeError("Input must be a string or a list of strings.")
    return tokenized
  def decode(self, input, skip_special_tokens=True):
    if isinstance(input, list):
      if all(isinstance(i, list) for i in input):
        # Handle list of lists (multiple sequences of tokens
        decoded = self.tokenizer.decode_batch(input, skip_special_tokens=skip_special_tokens)
      elif all(isinstance(i, int) for i in input):
        # Handle list of integers (single sequence of tokens)
        decoded = self.tokenizer.decode(input, skip_special_tokens=skip_special_tokens)
    # print(input)
    # print("list :", self.tokenizer.decode_batch(input, skip_special_tokens=skip_special_tokens))
    # print("int :", self.tokenizer.decode(input, skip_special_tokens=skip_special_tokens))
    return decoded
  
# trained from dataset
# dataset = load_from_disk("path")

# def get_training_corpus():
#   for i in range(0, len(dataset), 1000): # Process in batches of 1000
#     yield dataset[i : i + 1000]["text"]
