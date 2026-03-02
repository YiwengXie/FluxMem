import torch
import torch.nn.functional as F
from typing import List, Tuple


def scan_visual_indices(
    input_ids_row: torch.Tensor,
    vision_start_token_id: int,
    vision_end_token_id: int,
) -> torch.Tensor:
    """Return 1D LongTensor of visual token indices for a single sequence.

    Visual tokens are those strictly between matching pairs of
    `vision_start_token_id` and `vision_end_token_id`.
    """
    device = input_ids_row.device
    starts = (input_ids_row == vision_start_token_id).nonzero(as_tuple=True)[0]
    ends = (input_ids_row == vision_end_token_id).nonzero(as_tuple=True)[0]
    visual: list[int] = []
    for s, e in zip(starts.tolist(), ends.tolist()):
        if e - s > 1:
            visual.extend(range(s + 1, e))
    if len(visual) == 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    return torch.tensor(visual, dtype=torch.long, device=device)


def right_pad_and_stack(
    processed_list: List[dict],
    pos_key1: str = "pos_emb1",
    pos_key2: str = "pos_emb2",
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Right-pad variable-length samples and stack into batch tensors.

    Each item in `processed_list` must contain:
      - 'hidden': [L, D]
      - 'pos_ids': [3, L]
      - pos_key1: [3, L, E]
      - pos_key2: [3, L, E]
    Returns: hidden_states [B, Lmax, D], (pos_e1, pos_e2) each [3, B, Lmax, E],
             position_ids [3, B, Lmax], attention_mask [B, Lmax]
    """
    if not processed_list:
        return None, None, None, None

    max_len = max(p['hidden'].shape[0] for p in processed_list)
    batch_hidden, batch_pos_ids, batch_pos_e1, batch_pos_e2, batch_attn = [], [], [], [], []

    for p in processed_list:
        seq_len = int(p['hidden'].shape[0])
        if seq_len < max_len:
            pad_len = max_len - seq_len
            hidden = F.pad(p['hidden'], (0, 0, 0, pad_len))
            pos_ids = F.pad(p['pos_ids'], (0, pad_len))
            pos_e1 = F.pad(p[pos_key1], (0, 0, 0, pad_len))
            pos_e2 = F.pad(p[pos_key2], (0, 0, 0, pad_len))
            attn = torch.cat([
                torch.ones(seq_len, dtype=torch.long, device=p['hidden'].device),
                torch.zeros(pad_len, dtype=torch.long, device=p['hidden'].device)
            ], dim=0)
        else:
            hidden = p['hidden']
            pos_ids = p['pos_ids']
            pos_e1 = p[pos_key1]
            pos_e2 = p[pos_key2]
            attn = torch.ones(seq_len, dtype=torch.long, device=p['hidden'].device)

        batch_hidden.append(hidden)
        batch_pos_ids.append(pos_ids)
        batch_pos_e1.append(pos_e1)
        batch_pos_e2.append(pos_e2)
        batch_attn.append(attn)

    hidden_states = torch.stack(batch_hidden, dim=0)
    position_ids = torch.stack(batch_pos_ids, dim=0).transpose(0, 1)
    pos_e1_b = torch.stack(batch_pos_e1, dim=0).transpose(0, 1)
    pos_e2_b = torch.stack(batch_pos_e2, dim=0).transpose(0, 1)
    attention_mask = torch.stack(batch_attn, dim=0)

    return hidden_states, (pos_e1_b, pos_e2_b), position_ids, attention_mask
