import torch
@torch.no_grad()
def greedy_decode_with_encode(model, src, src_pad_mask, max_len, sos_idx, eos_idx, tgt_is_causal=False, device='cpu'):
  model.eval()
  memory = model.encode(src, src_key_padding_mask=src_pad_mask)
  generated = [sos_idx]
  for step in range(max_len):
    tgt_input = torch.tensor([generated], device=device)  # (1, cur_len)
    # create causal mask for current tgt length
    # tgt_mask = generate_square_subsequent_mask(tgt_input.size(1), device=device)
    logits = model.decode(tgt_input, memory, tgt_mask=None, tgt_key_padding_mask=None, tgt_is_causal=tgt_is_causal, memory_key_padding_mask=src_pad_mask)
    next_token = torch.argmax(logits[0, -1, :]).item()
    generated.append(next_token)
    if next_token == eos_idx:
      break
  return generated[1:] 


def beam_search():
  pass