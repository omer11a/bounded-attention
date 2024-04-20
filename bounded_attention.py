import nltk
import einops
import torch
import torch.nn.functional as F
import torchvision.utils
from torch_kmeans import KMeans

import os

import injection_utils
import utils


class BoundedAttention(injection_utils.AttentionBase):
    EPSILON = 1e-5
    FILTER_TAGS = {
        'CC', 'CD', 'DT', 'EX', 'IN', 'LS', 'MD', 'PDT', 'POS', 'PRP', 'PRP$', 'RP', 'TO', 'UH', 'WDT', 'WP', 'WRB'}
    TAG_RULES = {'left': 'IN', 'right': 'IN', 'top': 'IN', 'bottom': 'IN'}

    def __init__(
        self,
        boxes,
        prompts,
        subject_token_indices,
        cross_loss_layers,
        self_loss_layers,
        cross_mask_layers=None,
        self_mask_layers=None,
        eos_token_index=None,
        filter_token_indices=None,
        leading_token_indices=None,
        mask_cross_during_guidance=True,
        mask_eos=True,
        cross_loss_coef=1,
        self_loss_coef=1,
        max_guidance_iter=15,
        max_guidance_iter_per_step=5,
        start_step_size=30,
        end_step_size=10,
        loss_stopping_value=0.2,
        min_clustering_step=15,
        cross_mask_threshold=0.2,
        self_mask_threshold=0.2,
        delta_refine_mask_steps=5,
        pca_rank=None,
        num_clusters=None,
        num_clusters_per_box=3,
        max_resolution=None,
        map_dir=None,
        debug=False,
        delta_debug_attention_steps=20,
        delta_debug_mask_steps=5,
        debug_layers=None,
        saved_resolution=64,
    ):
        super().__init__()
        self.boxes = boxes
        self.prompts = prompts
        self.subject_token_indices = subject_token_indices
        self.cross_loss_layers = set(cross_loss_layers)
        self.self_loss_layers = set(self_loss_layers)
        self.cross_mask_layers = self.cross_loss_layers if cross_mask_layers is None else set(cross_mask_layers)
        self.self_mask_layers = self.self_loss_layers if self_mask_layers is None else set(self_mask_layers)

        self.eos_token_index = eos_token_index
        self.filter_token_indices = filter_token_indices
        self.leading_token_indices = leading_token_indices
        self.mask_cross_during_guidance = mask_cross_during_guidance
        self.mask_eos = mask_eos
        self.cross_loss_coef = cross_loss_coef
        self.self_loss_coef = self_loss_coef
        self.max_guidance_iter = max_guidance_iter
        self.max_guidance_iter_per_step = max_guidance_iter_per_step
        self.start_step_size = start_step_size
        self.step_size_coef = (end_step_size - start_step_size) / max_guidance_iter
        self.loss_stopping_value = loss_stopping_value
        self.min_clustering_step = min_clustering_step
        self.cross_mask_threshold = cross_mask_threshold
        self.self_mask_threshold = self_mask_threshold

        self.delta_refine_mask_steps = delta_refine_mask_steps
        self.pca_rank = pca_rank
        num_clusters = len(boxes) * num_clusters_per_box if num_clusters is None else num_clusters
        self.clustering = KMeans(n_clusters=num_clusters, num_init=100)
        self.centers = None

        self.max_resolution = max_resolution
        self.map_dir = map_dir
        self.debug = debug
        self.delta_debug_attention_steps = delta_debug_attention_steps
        self.delta_debug_mask_steps = delta_debug_mask_steps
        self.debug_layers = self.cross_loss_layers | self.self_loss_layers if debug_layers is None else debug_layers
        self.saved_resolution = saved_resolution

        self.optimized = False
        self.cross_foreground_values = []
        self.self_foreground_values = []
        self.cross_background_values = []
        self.self_background_values = []
        self.mean_cross_map = 0
        self.num_cross_maps = 0
        self.mean_self_map = 0
        self.num_self_maps = 0
        self.self_masks = None

    def clear_values(self, include_maps=False):
        lists = (
            self.cross_foreground_values,
            self.self_foreground_values,
            self.cross_background_values,
            self.self_background_values,
        )

        for values in lists:
            values.clear()

        if include_maps:
            self.mean_cross_map = 0
            self.num_cross_maps = 0
            self.mean_self_map = 0
            self.num_self_maps = 0

    def before_step(self):
        self.clear_values()
        if self.cur_step == 0:
            self._determine_tokens()

    def reset(self):
        self.clear_values(include_maps=True)
        super().reset()

    def forward(self, q, k, v, is_cross, place_in_unet, num_heads, **kwargs):
        batch_size = q.size(0) // num_heads
        n = q.size(1)
        d = k.size(1)
        dtype = q.dtype
        device = q.device
        if is_cross:
            masks = self._hide_other_subjects_from_tokens(batch_size // 2, n, d, dtype, device)
        else:
            masks = self._hide_other_subjects_from_subjects(batch_size // 2, n, dtype, device)

        resolution = int(n ** 0.5)
        if (self.max_resolution is not None) and (resolution > self.max_resolution):
            return super().forward(q, k, v, is_cross, place_in_unet, num_heads, mask=masks)

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * kwargs['scale']
        attn = sim.softmax(-1)
        self._display_attention_maps(attn, is_cross, num_heads)
        sim = sim.reshape(batch_size, num_heads, n, d) + masks
        attn = sim.reshape(-1, n, d).softmax(-1)
        self._save(attn, is_cross, num_heads)
        self._display_attention_maps(attn, is_cross, num_heads, prefix='masked')
        self._debug_hook(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        out = torch.bmm(attn, v)
        return einops.rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)

    def update_loss(self, forward_pass, latents, i):
        if i >= self.max_guidance_iter:
            return latents

        step_size = self.start_step_size + self.step_size_coef * i

        self.optimized = True
        normalized_loss = torch.tensor(10000)
        with torch.enable_grad():
            latents = latents.clone().detach().requires_grad_(True)
            for guidance_iter in range(self.max_guidance_iter_per_step):
                if normalized_loss < self.loss_stopping_value:
                    break

                latent_model_input = torch.cat([latents] * 2)
                cur_step = self.cur_step
                forward_pass(latent_model_input)
                self.cur_step = cur_step

                loss, normalized_loss = self._compute_loss()
                grad_cond = torch.autograd.grad(loss, [latents])[0]
                latents = latents - step_size * grad_cond
                if self.debug:
                    print(f'Loss at step={i}, iter={guidance_iter}: {normalized_loss}')
                    grad_norms = grad_cond.flatten(start_dim=2).norm(dim=1)
                    grad_norms = grad_norms / grad_norms.max(dim=1, keepdim=True)[0]
                    self._save_maps(grad_norms, 'grad_norms')

        self.optimized = False
        return latents

    def _tokenize(self):
        ids = self.model.tokenizer.encode(self.prompts[0])
        tokens = self.model.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
        return [token[:-4] for token in tokens]  # remove ending </w>

    def _tag_tokens(self):
        tagged_tokens = nltk.pos_tag(self._tokenize())
        return [type(self).TAG_RULES.get(token, tag) for token, tag in tagged_tokens]

    def _determine_eos_token(self):
        tokens = self._tokenize()
        eos_token_index = len(tokens) + 1
        if self.eos_token_index is None:
            self.eos_token_index = eos_token_index
        elif eos_token_index != self.eos_token_index:
            raise ValueError(f'Wrong EOS token index. Tokens are: {tokens}.')

    def _determine_filter_tokens(self):
        if self.filter_token_indices is not None:
            return

        tags = self._tag_tokens()
        self.filter_token_indices = [i + 1 for i, tag in enumerate(tags) if tag in type(self).FILTER_TAGS]

    def _determine_leading_tokens(self):
        if self.leading_token_indices is not None:
            return

        tags = self._tag_tokens()
        leading_token_indices = []
        for indices in self.subject_token_indices:
            subject_noun_indices = [i for i in indices if tags[i - 1].startswith('NN')]
            leading_token_candidates = subject_noun_indices if len(subject_noun_indices) > 0 else indices
            leading_token_indices.append(leading_token_candidates[-1])

        self.leading_token_indices = leading_token_indices

    def _determine_tokens(self):
        self._determine_eos_token()
        self._determine_filter_tokens()
        self._determine_leading_tokens()

    def _split_references(self, tensor, num_heads):
        tensor = tensor.reshape(-1, num_heads, *tensor.shape[1:])
        unconditional, conditional = tensor.chunk(2)

        num_subjects = len(self.boxes)
        batch_unconditional = unconditional[:-num_subjects]
        references_unconditional = unconditional[-num_subjects:]
        batch_conditional = conditional[:-num_subjects]
        references_conditional = conditional[-num_subjects:]

        batch = torch.cat((batch_unconditional, batch_conditional))
        references = torch.cat((references_unconditional, references_conditional))
        batch = batch.reshape(-1, *batch_unconditional.shape[2:])
        references = references.reshape(-1, *references_unconditional.shape[2:])
        return batch, references

    def _hide_other_subjects_from_tokens(self, batch_size, n, d, dtype, device):  # b h i j
        resolution = int(n ** 0.5)
        subject_masks, background_masks = self._obtain_masks(resolution, batch_size=batch_size, device=device)  # b s n
        include_background = self.optimized or (not self.mask_cross_during_guidance and self.cur_step < self.max_guidance_iter_per_step)
        subject_masks = torch.logical_or(subject_masks, background_masks.unsqueeze(1)) if include_background else subject_masks
        min_value = torch.finfo(dtype).min
        sim_masks = torch.zeros((batch_size, n, d), dtype=dtype, device=device)  # b i j
        for token_indices in (*self.subject_token_indices, self.filter_token_indices):
            sim_masks[:, :, token_indices] = min_value

        for batch_index in range(batch_size):
            for subject_mask, token_indices in zip(subject_masks[batch_index], self.subject_token_indices):
                for token_index in token_indices:
                    sim_masks[batch_index, subject_mask, token_index] = 0

        if self.mask_eos and not include_background:
            for batch_index, background_mask in zip(range(batch_size), background_masks):
                sim_masks[batch_index, background_mask, self.eos_token_index] = min_value

        return torch.cat((torch.zeros_like(sim_masks), sim_masks)).unsqueeze(1)

    def _hide_other_subjects_from_subjects(self, batch_size, n, dtype, device):  # b h i j
        resolution = int(n ** 0.5)
        subject_masks, background_masks = self._obtain_masks(resolution, batch_size=batch_size, device=device)  # b s n
        min_value = torch.finfo(dtype).min
        sim_masks = torch.zeros((batch_size, n, n), dtype=dtype, device=device)  # b i j
        for batch_index, background_mask in zip(range(batch_size), background_masks):
            sim_masks[batch_index, ~background_mask, ~background_mask] = min_value

        for batch_index in range(batch_size):
            for subject_mask in subject_masks[batch_index]:
                subject_sim_mask = sim_masks[batch_index, subject_mask]
                condition = torch.logical_or(subject_sim_mask == 0, subject_mask.unsqueeze(0))
                sim_masks[batch_index, subject_mask] = torch.where(condition, 0, min_value).to(dtype=dtype)

        return torch.cat((sim_masks, sim_masks)).unsqueeze(1)

    def _save(self, attn, is_cross, num_heads):
        _, attn = attn.chunk(2)
        attn = attn.reshape(-1, num_heads, *attn.shape[-2:])  # b h n k

        self._save_mask_maps(attn, is_cross)
        self._save_loss_values(attn, is_cross)

    def _save_mask_maps(self, attn, is_cross):
        if (
            (self.optimized) or
            (is_cross and self.cur_att_layer not in self.cross_mask_layers) or
            ((not is_cross) and (self.cur_att_layer not in self.self_mask_layers))
        ):
            return

        if is_cross:
            attn = attn[..., self.leading_token_indices]
            mean_map = self.mean_cross_map
            num_maps = self.num_cross_maps
        else:
            mean_map = self.mean_self_map
            num_maps = self.num_self_maps

        num_maps += 1
        attn = attn.mean(dim=1)  # mean over heads
        mean_map = ((num_maps - 1) / num_maps) * mean_map +  (1 / num_maps) * attn
        if is_cross:
            self.mean_cross_map = mean_map
            self.num_cross_maps = num_maps
        else:
            self.mean_self_map = mean_map
            self.num_self_maps = num_maps

    def _save_loss_values(self, attn, is_cross):
        if (
            (not self.optimized) or
            (is_cross and (self.cur_att_layer not in self.cross_loss_layers)) or
            ((not is_cross) and (self.cur_att_layer not in self.self_loss_layers))
        ):
            return

        resolution = int(attn.size(2) ** 0.5)
        boxes = self._convert_boxes_to_masks(resolution, device=attn.device)  # s n
        background_mask = boxes.sum(dim=0) == 0

        if is_cross:
            saved_foreground_values = self.cross_foreground_values
            saved_background_values = self.cross_background_values
            contexts = [indices + [self.eos_token_index] for indices in self.subject_token_indices]  # TODO: fix EOS loss term
        else:
            saved_foreground_values = self.self_foreground_values
            saved_background_values = self.self_background_values
            contexts = boxes

        foreground_values = []
        background_values = []
        for i, (box, context) in enumerate(zip(boxes, contexts)):
            context_attn = attn[:, :, :, context]
            
            # sum over heads, pixels and contexts
            foreground_values.append(context_attn[:, :, box].sum(dim=(1, 2, 3)))
            background_values.append(context_attn[:, :, background_mask].sum(dim=(1, 2, 3)))

        saved_foreground_values.append(torch.stack(foreground_values, dim=1))
        saved_background_values.append(torch.stack(background_values, dim=1))

    def _compute_loss(self):
        cross_losses = self._compute_loss_term(self.cross_foreground_values, self.cross_background_values)
        self_losses = self._compute_loss_term(self.self_foreground_values, self.self_background_values)
        b, s = cross_losses.shape

        # sum over samples and subjects
        total_cross_loss = cross_losses.sum()
        total_self_loss = self_losses.sum()

        loss = self.cross_loss_coef * total_cross_loss + self.self_loss_coef * total_self_loss
        normalized_loss = loss / b / s
        return loss, normalized_loss

    def _compute_loss_term(self, foreground_values, background_values):
        # mean over layers
        mean_foreground = torch.stack(foreground_values).mean(dim=0)
        mean_background = torch.stack(background_values).mean(dim=0)
        iou = mean_foreground / (mean_foreground + len(self.boxes) * mean_background)
        return (1 - iou) ** 2

    def _obtain_masks(self, resolution, return_boxes=False, return_existing=False, batch_size=None, device=None):
        return_boxes = return_boxes or (return_existing and self.self_masks is None)
        if return_boxes or self.cur_step < self.min_clustering_step:
            masks = self._convert_boxes_to_masks(resolution, device=device).unsqueeze(0)
            if batch_size is not None:
                masks = masks.expand(batch_size, *masks.shape[1:])
        else:
            masks = self._obtain_self_masks(resolution, return_existing=return_existing)
            if device is not None:
                masks = masks.to(device=device)

        background_mask = masks.sum(dim=1) == 0
        return masks, background_mask

    def _convert_boxes_to_masks(self, resolution, device=None):  # s n
        boxes = torch.zeros(len(self.boxes), resolution, resolution, dtype=bool, device=device)
        for i, box in enumerate(self.boxes):
            x0, x1 = box[0] * resolution, box[2] * resolution
            y0, y1 = box[1] * resolution, box[3] * resolution

            boxes[i, round(y0) : round(y1), round(x0) : round(x1)] = True

        return boxes.flatten(start_dim=1)

    def _obtain_self_masks(self, resolution, return_existing=False):
        if (
            (self.self_masks is None) or
            (
                (self.cur_step % self.delta_refine_mask_steps == 0) and
                (self.cur_att_layer == 0) and
                (not return_existing)
            )
        ):
            self.self_masks = self._fix_zero_masks(self._build_self_masks())

        b, s, n = self.self_masks.shape
        mask_resolution = int(n ** 0.5)
        self_masks = self.self_masks.reshape(b, s, mask_resolution, mask_resolution).float()
        self_masks = F.interpolate(self_masks, resolution, mode='nearest-exact')
        return self_masks.flatten(start_dim=2).bool()

    def _build_self_masks(self):
        c, clusters = self._cluster_self_maps()  # b n
        cluster_masks = torch.stack([(clusters == cluster_index) for cluster_index in range(c)], dim=2)  # b n c
        cluster_area = cluster_masks.sum(dim=1, keepdim=True)  # b 1 c

        n = clusters.size(1)
        resolution = int(n ** 0.5)
        cross_masks = self._obtain_cross_masks(resolution)  # b s n
        cross_mask_area = cross_masks.sum(dim=2, keepdim=True)  # b s 1

        intersection = torch.bmm(cross_masks.float(), cluster_masks.float())  # b s c
        min_area = torch.minimum(cross_mask_area, cluster_area)  # b s c
        score_per_cluster, subject_per_cluster = torch.max(intersection / min_area, dim=1)  # b c
        subjects = torch.gather(subject_per_cluster, 1, clusters)  # b n
        scores = torch.gather(score_per_cluster, 1, clusters)  # b n

        s = cross_masks.size(1)
        self_masks = torch.stack([(subjects == subject_index) for subject_index in range(s)], dim=1)  # b s n
        scores = scores.unsqueeze(1).expand(-1 ,s, n)  # b s n
        self_masks[scores < self.self_mask_threshold] = False
        self._save_maps(self_masks, 'self_masks')
        return self_masks

    def _cluster_self_maps(self):  # b s n
        self_maps = self._compute_maps(self.mean_self_map)  # b n m
        if self.pca_rank is not None:
            dtype = self_maps.dtype
            _, _, eigen_vectors = torch.pca_lowrank(self_maps.float(), self.pca_rank)
            self_maps = torch.matmul(self_maps, eigen_vectors.to(dtype=dtype))

        clustering_results = self.clustering(self_maps, centers=self.centers)
        self.clustering.num_init = 1  # clustering is deterministic after the first time
        self.centers = clustering_results.centers
        clusters = clustering_results.labels
        num_clusters = self.clustering.n_clusters
        self._save_maps(clusters / num_clusters, f'clusters')
        return num_clusters, clusters

    def _obtain_cross_masks(self, resolution, scale=10):
        maps = self._compute_maps(self.mean_cross_map, resolution=resolution)  # b n k
        maps = F.sigmoid(scale * (maps - self.cross_mask_threshold))
        maps = self._normalize_maps(maps, reduce_min=True)
        maps = maps.transpose(1, 2)  # b k n
        existing_masks, _ = self._obtain_masks(
            resolution, return_existing=True, batch_size=maps.size(0), device=maps.device)
        maps = maps * existing_masks.to(dtype=maps.dtype)
        self._save_maps(maps, 'cross_masks')
        return maps

    def _fix_zero_masks(self, masks):
        b, s, n = masks.shape
        resolution = int(n ** 0.5)
        boxes = self._convert_boxes_to_masks(resolution, device=masks.device)  # s n

        for i in range(b):
            for j in range(s):
                if masks[i, j].sum() == 0:
                    print('******Found a zero mask!******')
                    for k in range(s):
                        masks[i, k] = boxes[j] if (k == j) else masks[i, k].logical_and(~boxes[j])

        return masks

    def _compute_maps(self, maps, resolution=None):  # b n k
        if resolution is not None:
            b, n, k = maps.shape
            original_resolution = int(n ** 0.5)
            maps = maps.transpose(1, 2).reshape(b, k, original_resolution, original_resolution)
            maps = F.interpolate(maps, resolution, mode='bilinear', antialias=True)
            maps = maps.reshape(b, k, -1).transpose(1, 2)

        maps = self._normalize_maps(maps)
        return maps

    @classmethod
    def _normalize_maps(cls, maps, reduce_min=False):  # b n k
        max_values = maps.max(dim=1, keepdim=True)[0]
        min_values = maps.min(dim=1, keepdim=True)[0] if reduce_min else 0
        numerator = maps - min_values
        denominator = max_values - min_values + cls.EPSILON
        return numerator / denominator

    def _save_maps(self, maps, prefix):
        if self.map_dir is None or self.cur_step % self.delta_debug_mask_steps != 0:
            return

        resolution = int(maps.size(-1) ** 0.5)
        maps = maps.reshape(-1, 1, resolution, resolution).float()
        maps = F.interpolate(maps, self.saved_resolution, mode='bilinear', antialias=True)
        path = os.path.join(self.map_dir, f'map_{prefix}_{self.cur_step}_{self.cur_att_layer}.png')
        torchvision.utils.save_image(maps, path)

    def _display_attention_maps(self, attention_maps, is_cross, num_heads, prefix=None):
        if (not self.debug) or (self.cur_step == 0) or (self.cur_step % self.delta_debug_attention_steps > 0) or (self.cur_att_layer not in self.debug_layers):
            return

        dir_name = self.map_dir
        if prefix is not None:
            splits = list(os.path.split(dir_name))
            splits[-1] = '_'.join((prefix, splits[-1]))
            dir_name = os.path.join(*splits)

        resolution = int(attention_maps.size(-2) ** 0.5)
        if is_cross:
            attention_maps = einops.rearrange(attention_maps, 'b (r1 r2) k -> b k r1 r2', r1=resolution)
            attention_maps = F.interpolate(attention_maps, self.saved_resolution, mode='bilinear', antialias=True)
            attention_maps = einops.rearrange(attention_maps, 'b k r1 r2 -> b (r1 r2) k')

        utils.display_attention_maps(
            attention_maps,
            is_cross,
            num_heads,
            self.model.tokenizer,
            self.prompts,
            dir_name,
            self.cur_step,
            self.cur_att_layer,
            resolution,
        )

    def _debug_hook(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        pass
