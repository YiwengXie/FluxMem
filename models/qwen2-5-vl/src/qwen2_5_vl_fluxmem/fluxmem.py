"""
FluxMem streaming memory for Qwen2.5-VL:
- short-term buffer keeps the latest frames;
- mid-term temporal overlap with Otsu + similarity drop to prune near-duplicate tokens;
- long-term per-frame clustering to keep anchors only.
"""

from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F

from .utils import scan_visual_indices, right_pad_and_stack


class FluxMem:
    _OFFSETS_3X3 = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 0), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )
    _OFFSETS_NEIGHBOR = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    def __init__(
        self,
        vision_start_token_id: int,
        vision_end_token_id: int,
        short_term_frames: int = 8,
        mid_term_frames: int = 256,
        direct_drop_sim_threshold: float = 0.8,
    ):
        self.short_term_frames = short_term_frames
        self.mid_term_frames = mid_term_frames
        self.temporal_patch_size = 2
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.direct_drop_sim_threshold = float(direct_drop_sim_threshold)

    @staticmethod
    def _neighbor_offsets(device: torch.device, include_center: bool) -> torch.Tensor:
        offsets = FluxMem._OFFSETS_3X3 if include_center else FluxMem._OFFSETS_NEIGHBOR
        return torch.tensor(offsets, dtype=torch.long, device=device)

    @staticmethod
    def _compute_grid_hw(
        visual_indices: torch.Tensor,
        height_ids: torch.Tensor,
        width_ids: torch.Tensor,
    ) -> tuple[int, int] | None:
        if visual_indices.numel() == 0:
            return None
        max_h = int(height_ids[visual_indices].max().item())
        max_w = int(width_ids[visual_indices].max().item())
        return max_h + 1, max_w + 1

    @staticmethod
    def _pack_sample(
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        batch_index: int,
        kept_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "hidden": hidden_states[batch_index, kept_indices],
            "pos_ids": position_ids[:, batch_index, kept_indices],
            "pos_emb1": position_embeddings[0][:, batch_index, kept_indices],
            "pos_emb2": position_embeddings[1][:, batch_index, kept_indices],
        }

    @staticmethod
    def _otsu_threshold(
        values: torch.Tensor,
        nbins: int = 128,
        fallback_to_median: bool = False,
    ) -> float | None:
        if values.numel() == 0:
            return None

        v = values.detach().float().clamp(0.0, 2.0)
        if fallback_to_median and float(v.var().item()) <= 1e-12:
            return float(torch.median(v).item())

        hist = torch.histc(v, bins=nbins, min=0.0, max=2.0)
        total = float(hist.sum().item())
        if total <= 0:
            return float(torch.median(v).item()) if fallback_to_median else None

        p = hist / total
        step = 2.0 / nbins
        centers = (torch.arange(nbins, dtype=torch.float32, device=v.device) + 0.5) * step
        omega = torch.cumsum(p, 0)
        mu_k = torch.cumsum(p * centers, 0)
        mu_total = mu_k[-1]
        denom = (omega * (1.0 - omega)).clamp_min(1e-12)
        sigma_b2 = (mu_total * omega - mu_k) ** 2 / denom
        sigma_b2[omega < 1e-6] = -1
        sigma_b2[(1.0 - omega) < 1e-6] = -1
        threshold_index = int(torch.argmax(sigma_b2).item())
        if float(sigma_b2[threshold_index].item()) <= 0.0:
            return float(torch.median(v).item()) if fallback_to_median else None
        return float((threshold_index + 0.5) * step)

    @staticmethod
    def _frame_grids(
        visual_indices: torch.Tensor,
        time_ids: torch.Tensor,
        height_ids: torch.Tensor,
        width_ids: torch.Tensor,
        grid_hw: tuple[int, int] | None,
    ) -> tuple[List[int], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        frame_ids = torch.unique(time_ids[visual_indices], sorted=True)
        frames = [int(frame_id) for frame_id in frame_ids.tolist()]
        frame_to_indices: dict[int, torch.Tensor] = {}
        frame_to_grid: dict[int, torch.Tensor] = {}
        device = visual_indices.device

        for frame_id in frames:
            frame_indices = visual_indices[time_ids[visual_indices] == frame_id]
            frame_to_indices[frame_id] = frame_indices
            if grid_hw is None:
                continue

            frame_heights = height_ids[frame_indices].to(torch.long)
            frame_widths = width_ids[frame_indices].to(torch.long)
            grid = torch.full(grid_hw, -1, dtype=torch.long, device=device)
            if frame_indices.numel() > 0:
                grid[frame_heights, frame_widths] = frame_indices

            padded_grid = torch.full((grid_hw[0] + 2, grid_hw[1] + 2), -1, dtype=torch.long, device=device)
            padded_grid[1:-1, 1:-1] = grid
            frame_to_grid[frame_id] = padded_grid

        return frames, frame_to_indices, frame_to_grid

    def _max_sim_against_frames(
        self,
        query_indices: torch.Tensor,
        neighbor_frames: List[int],
        height_ids: torch.Tensor,
        width_ids: torch.Tensor,
        frame_grids: dict[int, torch.Tensor],
        hidden_norm: torch.Tensor,
        offsets_3x3: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query_heights = height_ids[query_indices].to(torch.long)
        query_widths = width_ids[query_indices].to(torch.long)
        if query_heights.numel() == 0:
            raise RuntimeError("No neighbors found for window/pair matching")

        grid_heights = query_heights + 1
        grid_widths = query_widths + 1
        neighbor_heights = grid_heights.view(-1, 1) + offsets_3x3[:, 0].view(1, -1)
        neighbor_widths = grid_widths.view(-1, 1) + offsets_3x3[:, 1].view(1, -1)

        neighbor_mats: List[torch.Tensor] = []
        for neighbor_frame in neighbor_frames:
            neighbor_mats.append(frame_grids[neighbor_frame][neighbor_heights, neighbor_widths])
        neighbor_mat = neighbor_mats[0] if len(neighbor_mats) == 1 else torch.cat(neighbor_mats, dim=1)

        valid_neighbors = neighbor_mat >= 0
        has_neighbors = valid_neighbors.any(dim=1)
        if not bool(has_neighbors.any().item()):
            raise RuntimeError("No neighbors found for window/pair matching")

        query_with_neighbors = query_indices[has_neighbors]
        neighbor_mat = neighbor_mat[has_neighbors]
        valid_neighbors = valid_neighbors[has_neighbors]

        safe_indices = neighbor_mat.clamp_min(0)
        neighbor_features = hidden_norm[safe_indices]
        query_features = hidden_norm[query_with_neighbors].unsqueeze(1)
        similarities = (neighbor_features * query_features).sum(dim=2)
        similarities = similarities.masked_fill(~valid_neighbors, float("-inf"))
        max_similarities = similarities.max(dim=1).values
        distances = 1.0 - max_similarities
        return query_with_neighbors, max_similarities, distances, has_neighbors

    def _apply_adjacent_pruning(
        self,
        frame_id: int,
        buffer_frames: List[int],
        frame_to_indices: dict[int, torch.Tensor],
        frame_grids: dict[int, torch.Tensor],
        height_ids: torch.Tensor,
        width_ids: torch.Tensor,
        hidden_norm: torch.Tensor,
        pair_distance_cache: torch.Tensor,
        keep_mask: torch.Tensor,
        pair_distance_threshold: Optional[float],
        offsets_3x3: torch.Tensor,
    ) -> None:
        query_indices = frame_to_indices[frame_id]
        adjacent_frames = [buffer_frames[0]] if len(buffer_frames) >= 1 else []
        if len(adjacent_frames) == 0:
            raise RuntimeError("Empty neighbor frame for adjacent-threshold")

        query_with_next, max_sim_next, next_distances, has_next_neighbors = self._max_sim_against_frames(
            query_indices=query_indices,
            neighbor_frames=adjacent_frames,
            height_ids=height_ids,
            width_ids=width_ids,
            frame_grids=frame_grids,
            hidden_norm=hidden_norm,
            offsets_3x3=offsets_3x3,
        )

        if has_next_neighbors.numel() != query_indices.numel():
            raise RuntimeError("Mismatch between query tokens and neighbor mask")
        if not bool(has_next_neighbors.all().item()):
            missing_neighbors = query_indices[~has_next_neighbors]
            if missing_neighbors.numel() > 0:
                keep_mask[missing_neighbors] = True

        if pair_distance_threshold is not None:
            next_threshold = float(max(0.0, min(2.0, float(pair_distance_threshold))))
        else:
            next_threshold = self._otsu_threshold(next_distances, fallback_to_median=True)
            if next_threshold is None:
                raise RuntimeError("Empty set for Otsu thresholding")
        keep_by_next = next_distances >= next_threshold

        prev_distances = pair_distance_cache[query_indices]
        valid_prev = prev_distances >= 0
        prev_threshold = None
        if bool(valid_prev.any().item()):
            if pair_distance_threshold is not None:
                prev_threshold = float(max(0.0, min(2.0, float(pair_distance_threshold))))
            else:
                prev_threshold = self._otsu_threshold(prev_distances[valid_prev], fallback_to_median=True)
                if prev_threshold is None:
                    raise RuntimeError("Empty set for Otsu thresholding")

        if prev_threshold is not None:
            prev_distances_on_next = pair_distance_cache[query_with_next]
            keep_local = keep_by_next | (prev_distances_on_next >= prev_threshold)
        else:
            keep_local = keep_by_next

        if query_with_next.numel() > 0:
            drop_high_similarity = max_sim_next > self.direct_drop_sim_threshold
            if bool(drop_high_similarity.any().item()):
                keep_local = keep_local & (~drop_high_similarity)

        kept_indices = query_with_next[keep_local]
        if kept_indices.numel() > 0:
            keep_mask[kept_indices] = True

    def process_memory_streaming(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        input_ids: torch.Tensor,
        video_grid_thw: Optional[torch.Tensor] = None,
        pair_distance_threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Streaming memory (prefill only):
        - short-term: keep all tokens from the latest S frames;
        - mid-term: when the buffer overflows, prune the oldest frame via prev/next frame Otsu thresholds on 3x3 neighbors,
          with an extra direct-drop when similarity is very high;
        - long-term: when mid-term exceeds L, merge each old frame by thresholded local clustering and anchor selection.
        """
        if video_grid_thw is None:
            raise ValueError("video_grid_thw is None for streaming memory")

        batch_size, sequence_length, _ = hidden_states.shape
        device = hidden_states.device
        temporal_patch_size = int(getattr(self, "temporal_patch_size", 2))
        short_term_grids = int(self.short_term_frames / temporal_patch_size)
        mid_term_limit = int(self.mid_term_frames / temporal_patch_size)
        if short_term_grids < 1:
            raise ValueError(
                f"short_term_frames ({self.short_term_frames}) too small for temporal_patch_size "
                f"({temporal_patch_size}); got short_grids={short_term_grids}."
            )

        offsets_3x3 = self._neighbor_offsets(device, include_center=True)

        processed_samples = []
        kept_indices_list: List[torch.Tensor] = []

        for batch_index in range(batch_size):
            sample_input_ids = input_ids[batch_index]
            visual_indices = scan_visual_indices(
                sample_input_ids,
                self.vision_start_token_id,
                self.vision_end_token_id,
            )
            if visual_indices.numel() == 0:
                kept_indices = torch.arange(sequence_length, device=device)
                processed_samples.append(
                    self._pack_sample(hidden_states, position_ids, position_embeddings, batch_index, kept_indices)
                )
                kept_indices_list.append(kept_indices)
                continue

            time_ids = position_ids[0, batch_index]
            height_ids = position_ids[1, batch_index].to(torch.long)
            width_ids = position_ids[2, batch_index].to(torch.long)
            grid_hw = self._compute_grid_hw(visual_indices, height_ids, width_ids)
            frames, frame_to_indices, frame_grids = self._frame_grids(
                visual_indices=visual_indices,
                time_ids=time_ids,
                height_ids=height_ids,
                width_ids=width_ids,
                grid_hw=grid_hw,
            )
            if len(frames) == 0:
                raise RuntimeError("No visual frames extracted")

            keep_mask = torch.zeros(sequence_length, dtype=torch.bool, device=device)
            non_visual_mask = torch.ones(sequence_length, dtype=torch.bool, device=device)
            non_visual_mask[visual_indices] = False
            keep_mask[non_visual_mask] = True

            hidden_norm = F.normalize(hidden_states[batch_index], p=2, dim=1, eps=1e-8)
            pair_distance_cache = torch.full((sequence_length,), -1.0, dtype=hidden_states.dtype, device=device)
            short_term_buffer: List[int] = []
            mid_term_frames: List[int] = []

            for frame_offset, frame_id in enumerate(frames):
                short_term_buffer.append(frame_id)

                if frame_offset > 0:
                    prev_frame_id = frames[frame_offset - 1]
                    current_indices = frame_to_indices[frame_id]
                    paired_indices, _, paired_distances, _ = self._max_sim_against_frames(
                        query_indices=current_indices,
                        neighbor_frames=[prev_frame_id],
                        height_ids=height_ids,
                        width_ids=width_ids,
                        frame_grids=frame_grids,
                        hidden_norm=hidden_norm,
                        offsets_3x3=offsets_3x3,
                    )
                    if paired_indices.numel() > 0:
                        pair_distance_cache[paired_indices] = paired_distances

                if len(short_term_buffer) > max(1, short_term_grids):
                    evicted_frame = short_term_buffer.pop(0)
                    self._apply_adjacent_pruning(
                        frame_id=evicted_frame,
                        buffer_frames=short_term_buffer,
                        frame_to_indices=frame_to_indices,
                        frame_grids=frame_grids,
                        height_ids=height_ids,
                        width_ids=width_ids,
                        hidden_norm=hidden_norm,
                        pair_distance_cache=pair_distance_cache,
                        keep_mask=keep_mask,
                        pair_distance_threshold=pair_distance_threshold,
                        offsets_3x3=offsets_3x3,
                    )

                    mid_term_frames.append(evicted_frame)
                    if len(mid_term_frames) > max(0, mid_term_limit):
                        oldest_mid_frame = mid_term_frames.pop(0)
                        self._long_term_memory_merge_per_frames(
                            batch_index=batch_index,
                            long_frames={oldest_mid_frame},
                            visual_indices=visual_indices,
                            time_ids=time_ids,
                            height_ids=height_ids,
                            width_ids=width_ids,
                            hidden_states=hidden_states,
                            hidden_norm=hidden_norm,
                            keep_mask=keep_mask,
                            grid_hw=grid_hw,
                        )

            for frame_id in short_term_buffer:
                keep_mask[frame_to_indices[frame_id]] = True

            kept_indices = keep_mask.nonzero(as_tuple=True)[0]
            processed_samples.append(
                self._pack_sample(hidden_states, position_ids, position_embeddings, batch_index, kept_indices)
            )
            kept_indices_list.append(kept_indices)

        hidden_states_out, position_embeddings_out, position_ids_out, attention_mask_out = right_pad_and_stack(
            processed_samples,
            pos_key1="pos_emb1",
            pos_key2="pos_emb2",
        )
        return hidden_states_out, position_embeddings_out, position_ids_out, attention_mask_out, kept_indices_list

    def _long_term_memory_merge_per_frames(
        self,
        batch_index: int,
        long_frames: set[int],
        visual_indices: torch.Tensor,
        time_ids: torch.Tensor,
        height_ids: torch.Tensor,
        width_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        keep_mask: torch.Tensor,
        hidden_norm: Optional[torch.Tensor] = None,
        grid_hw: tuple[int, int] | None = None,
    ) -> None:
        """
        In-place merge for long-term frames:
        - build intra-frame 3x3 neighbor graph with cosine distances;
        - Otsu threshold on d=1-sim, connect edges with d <= tau;
        - pick one anchor per cluster (closest to mean), replace it by the mean, drop others.
        """
        device = hidden_states.device
        neighbor_offsets = self._neighbor_offsets(device, include_center=False)
        time_ids_visual = time_ids[visual_indices]

        for frame_id in sorted(long_frames):
            frame_visual_indices = visual_indices[time_ids_visual == frame_id]
            if frame_visual_indices.numel() == 0:
                continue

            kept_visual_mask = keep_mask[frame_visual_indices]
            if kept_visual_mask.numel() == 0:
                continue

            kept_frame_indices = frame_visual_indices[kept_visual_mask]
            if kept_frame_indices.numel() == 0:
                continue

            frame_heights = height_ids[kept_frame_indices].to(torch.long)
            frame_widths = width_ids[kept_frame_indices].to(torch.long)
            num_nodes = int(kept_frame_indices.numel())
            if num_nodes <= 1:
                continue

            if grid_hw is None:
                local_grid_hw = self._compute_grid_hw(kept_frame_indices, frame_heights, frame_widths)
                if local_grid_hw is None:
                    continue
            else:
                local_grid_hw = grid_hw

            padded_grid = torch.full(
                (local_grid_hw[0] + 2, local_grid_hw[1] + 2),
                -1,
                dtype=torch.long,
                device=device,
            )
            node_indices = torch.arange(num_nodes, dtype=torch.long, device=device)
            grid_heights = frame_heights + 1
            grid_widths = frame_widths + 1
            padded_grid[grid_heights, grid_widths] = node_indices

            neighbor_heights = grid_heights.view(-1, 1) + neighbor_offsets[:, 0].view(1, -1)
            neighbor_widths = grid_widths.view(-1, 1) + neighbor_offsets[:, 1].view(1, -1)
            neighbor_mat = padded_grid[neighbor_heights, neighbor_widths]
            source_mat = node_indices.unsqueeze(1).expand_as(neighbor_mat)
            upper_triangle = (neighbor_mat >= 0) & (neighbor_mat > source_mat)
            if not bool(upper_triangle.any().item()):
                continue

            edge_i = source_mat[upper_triangle]
            edge_j = neighbor_mat[upper_triangle]

            if hidden_norm is not None and hidden_norm.dtype == torch.float32:
                features_norm = hidden_norm[kept_frame_indices]
            else:
                features_norm = F.normalize(hidden_states[batch_index, kept_frame_indices].float(), p=2, dim=1, eps=1e-8)

            edge_distances = 1.0 - (features_norm[edge_i] * features_norm[edge_j]).sum(dim=1)
            edge_distances = edge_distances.clamp(0.0, 2.0)
            threshold = self._otsu_threshold(edge_distances)
            if threshold is None:
                continue

            keep_edges = edge_distances <= threshold
            if not bool(keep_edges.any().item()):
                continue

            edge_i = edge_i[keep_edges]
            edge_j = edge_j[keep_edges]

            labels = torch.arange(num_nodes, device=device, dtype=torch.long)
            for _ in range(max(1, num_nodes)):
                min_labels = torch.minimum(labels[edge_i], labels[edge_j])
                new_labels = labels.clone()
                new_labels.scatter_reduce_(0, edge_i, min_labels, reduce="amin", include_self=True)
                new_labels.scatter_reduce_(0, edge_j, min_labels, reduce="amin", include_self=True)
                new_labels = new_labels[new_labels]
                if torch.equal(new_labels, labels):
                    break
                labels = new_labels

            _, inverse = torch.unique(labels, return_inverse=True)
            num_clusters = int(inverse.max().item()) + 1 if inverse.numel() > 0 else 0
            if num_clusters <= 0:
                continue

            features = hidden_states[batch_index, kept_frame_indices].float()
            cluster_sum = torch.zeros((num_clusters, features.size(1)), dtype=features.dtype, device=device)
            cluster_sum.index_add_(0, inverse, features)
            counts = torch.zeros(num_clusters, dtype=features.dtype, device=device)
            counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=features.dtype))
            counts = counts.clamp_min(1.0)
            cluster_mean = cluster_sum / counts.unsqueeze(1)

            squared_distances = (features - cluster_mean[inverse]).pow(2).sum(dim=1)
            min_squared_distances = torch.full((num_clusters,), float("inf"), dtype=squared_distances.dtype, device=device)
            min_squared_distances.scatter_reduce_(0, inverse, squared_distances, reduce="amin", include_self=True)
            is_cluster_min = squared_distances == min_squared_distances[inverse]
            candidate_indices = torch.nonzero(is_cluster_min, as_tuple=True)[0]
            candidate_clusters = inverse[candidate_indices]
            anchor_local = torch.full((num_clusters,), num_nodes, dtype=torch.long, device=device)
            anchor_local.scatter_reduce_(0, candidate_clusters, candidate_indices, reduce="amin", include_self=True)

            merge_clusters = counts > 1.0
            if not bool(merge_clusters.any().item()):
                continue

            anchor_local = anchor_local[merge_clusters]
            anchor_global = kept_frame_indices[anchor_local]
            hidden_states[batch_index, anchor_global] = cluster_mean[merge_clusters].to(hidden_states.dtype)

            local_keep = torch.ones(num_nodes, dtype=torch.bool, device=device)
            local_keep[merge_clusters[inverse]] = False
            local_keep[anchor_local] = True
            keep_mask[kept_frame_indices[~local_keep]] = False
