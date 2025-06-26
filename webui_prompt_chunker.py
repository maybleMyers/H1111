import math
import re
from collections import namedtuple
import torch
import numpy as np

# This script is a consolidation of several modules from the AUTOMATIC1111/stable-diffusion-webui
# repository, refactored to work as a standalone library for prompt parsing and chunking.
# It includes the necessary components from:
# - sd_hijack_clip.py
# - prompt_parser.py
# - sd_emphasis.py
# - sd_hijack.py
# - textual_inversion/textual_inversion.py
#
# All internal dependencies have been resolved within this single file.

# --- From prompt_parser.py ---

re_attention = re.compile(r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:\s*([+-]?[.\d]+)\s*\)|
\)|
]|
[^\\()\[\]:]+|
:
""", re.X)
re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)

def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    """
    res = []
    round_brackets = []
    square_brackets = []
    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text_match = m.group(0)
        weight = m.group(1)

        if text_match.startswith('\\'):
            res.append([text_match[1:], 1.0])
        elif text_match == '(':
            round_brackets.append(len(res))
        elif text_match == '[':
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))
        elif text_match == ')' and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text_match == ']' and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text_match)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                if part:
                    res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)
    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if not res:
        res = [["", 1.0]]

    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1] and res[i][0] != "BREAK" and res[i + 1][0] != "BREAK":
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1
    return res


# --- From sd_emphasis.py ---

class Emphasis:
    name: str = "Base"
    tokens: list[list[int]]
    multipliers: torch.Tensor
    z: torch.Tensor
    def after_transformers(self): pass

class EmphasisOriginal(Emphasis):
    name = "Original"
    def after_transformers(self):
        original_mean = self.z.mean()
        self.z = self.z * self.multipliers.reshape(self.multipliers.shape + (1,)).expand(self.z.shape)
        new_mean = self.z.mean()
        self.z = self.z * (original_mean / new_mean)

class EmphasisNone(Emphasis):
    name = "None"

def get_emphasis_implementation(name):
    if name == "Original":
        return EmphasisOriginal
    return EmphasisNone

# --- From textual_inversion/textual_inversion.py (Simplified) ---

class Embedding:
    def __init__(self, vec, name, shorthash=None):
        self.vec = vec
        self.name = name
        self.shorthash = shorthash
        self.vectors = vec.shape[0]

class EmbeddingDatabase:
    """A simplified functional dummy for textual inversion."""
    def __init__(self):
        self.word_embeddings = {}
        # You could add logic here to load TIs from a directory if needed
        # For now, it's a no-op database.

    def find_embedding_at_position(self, tokens, offset):
        # This implementation does not support textual inversions.
        # It always returns None, so the chunker proceeds with standard tokenization.
        return None, 1


# --- From sd_hijack.py (Simplified) ---

class EmbeddingsWithFixes(torch.nn.Module):
    def __init__(self, wrapped, hijack_instance):
        super().__init__()
        self.wrapped = wrapped
        self.hijack = hijack_instance

    def forward(self, input_ids):
        batch_fixes = self.hijack.fixes
        self.hijack.fixes = None

        inputs_embeds = self.wrapped(input_ids)

        if batch_fixes is None or len(batch_fixes) == 0 or max([len(x) for x in batch_fixes]) == 0:
            return inputs_embeds

        vecs = []
        for fixes, tensor in zip(batch_fixes, inputs_embeds):
            for offset, embedding in fixes:
                emb = embedding.vec.to(tensor.device, dtype=tensor.dtype)
                emb_len = min(tensor.shape[0] - offset - 1, emb.shape[0])
                tensor = torch.cat([tensor[0:offset + 1], emb[0:emb_len], tensor[offset + 1 + emb_len:]])
            vecs.append(tensor)

        return torch.stack(vecs)

class StableDiffusionModelHijack:
    """A simplified functional dummy for hijacking."""
    def __init__(self):
        self.fixes = None
        self.embedding_db = EmbeddingDatabase()
        self.extra_generation_params = {}


# --- From sd_hijack_clip.py (Core Logic) ---

PromptChunk = namedtuple('PromptChunk', ["tokens", "multipliers", "fixes"])
PromptChunkFix = namedtuple('PromptChunkFix', ['offset', 'embedding'])

class TextConditionalModel(torch.nn.Module):
    def __init__(self, config, hijack_instance):
        super().__init__()
        self.config = config
        self.hijack = hijack_instance
        self.chunk_length = 75
        self.return_pooled = True # We need the pooled output for HunyuanDiT
        self.comma_token = None
        self.id_start = None
        self.id_end = None
        self.id_pad = None

    def empty_chunk(self):
        tokens = [self.id_start] + [self.id_end] * (self.chunk_length + 1)
        multipliers = [1.0] * (self.chunk_length + 2)
        return PromptChunk(tokens, multipliers, [])

    def tokenize_line(self, line):
        if self.config.emphasis != "None":
            parsed = parse_prompt_attention(line)
        else:
            parsed = [[line, 1.0]]

        tokenized = self.tokenize([text for text, _ in parsed])

        chunks = []
        chunk = PromptChunk([], [], [])
        token_count = 0
        last_comma = -1

        def next_chunk(is_last=False):
            nonlocal token_count, last_comma, chunk
            if is_last:
                token_count += len(chunk.tokens)
            else:
                token_count += self.chunk_length

            to_add = self.chunk_length - len(chunk.tokens)
            if to_add > 0:
                chunk.tokens.extend([self.id_end] * to_add)
                chunk.multipliers.extend([1.0] * to_add)
            
            final_tokens = [self.id_start] + chunk.tokens + [self.id_end]
            final_multipliers = [1.0] + chunk.multipliers + [1.0]

            chunks.append(PromptChunk(final_tokens, final_multipliers, chunk.fixes))
            chunk = PromptChunk([], [], [])
            last_comma = -1
        
        for tokens, (text, weight) in zip(tokenized, parsed):
            if text == 'BREAK' and weight == -1:
                if chunk.tokens or chunk.fixes:
                    next_chunk()
                continue

            position = 0
            while position < len(tokens):
                token = tokens[position]
                if token == self.comma_token:
                    last_comma = len(chunk.tokens)
                
                if self.config.comma_padding_backtrack > 0 and len(chunk.tokens) == self.chunk_length and last_comma != -1 and len(chunk.tokens) - last_comma <= self.config.comma_padding_backtrack:
                    break_location = last_comma + 1
                    reloc_tokens = chunk.tokens[break_location:]
                    reloc_mults = chunk.multipliers[break_location:]
                    chunk.tokens[:] = chunk.tokens[:break_location]
                    chunk.multipliers[:] = chunk.multipliers[:break_location]
                    next_chunk()
                    chunk.tokens.extend(reloc_tokens)
                    chunk.multipliers.extend(reloc_mults)

                if len(chunk.tokens) == self.chunk_length:
                    next_chunk()

                embedding, embedding_length_in_tokens = self.hijack.embedding_db.find_embedding_at_position(tokens, position)
                if embedding is None:
                    chunk.tokens.append(token)
                    chunk.multipliers.append(weight)
                    position += 1
                    continue
                
                emb_len = int(embedding.vectors)
                if len(chunk.tokens) + emb_len > self.chunk_length:
                    next_chunk()
                
                chunk.fixes.append(PromptChunkFix(len(chunk.tokens), embedding))
                chunk.tokens.extend([0] * emb_len)
                chunk.multipliers.extend([weight] * emb_len)
                position += embedding_length_in_tokens

        if chunk.tokens or not chunks:
            next_chunk(is_last=True)
            
        return chunks, token_count

    def process_texts(self, texts):
        token_count = 0
        cache = {}
        batch_chunks = []
        for line in texts:
            if line in cache:
                chunks = cache[line]
            else:
                chunks, current_token_count = self.tokenize_line(line)
                token_count = max(current_token_count, token_count)
                cache[line] = chunks
            batch_chunks.append(chunks)
        return batch_chunks, token_count

    def forward(self, texts):
        batch_chunks, _ = self.process_texts(texts)
        chunk_count = max(len(x) for x in batch_chunks) if batch_chunks else 0
        
        if chunk_count == 0: # Handle empty prompt
            batch_chunks = [[self.empty_chunk()]]
            chunk_count = 1

        zs = []
        for i in range(chunk_count):
            batch_chunk = [chunks[i] if i < len(chunks) else self.empty_chunk() for chunks in batch_chunks]
            tokens = [x.tokens for x in batch_chunk]
            multipliers = [x.multipliers for x in batch_chunk]
            self.hijack.fixes = [x.fixes for x in batch_chunk]
            
            z = self.process_tokens(tokens, multipliers)
            zs.append(z)

        # For HunyuanDiT, we need the embeddings and the pooled output.
        # The pooled output is typically taken from the first chunk.
        # The final embeddings are a concatenation of all chunks' embeddings.
        if self.return_pooled:
            # Note: For HunyuanDiT, which doesn't use concatenated embeddings from multiple chunks for the text prompt,
            # we will return the embeddings from the *first* chunk only, along with its pooled output.
            # If a model needed all chunks, `torch.cat(all_z_embeddings, dim=1)` would be used.
            # Here, we only need the pooled output from the first chunk.
            # We return all embeddings stacked in case a model needs them, but also the first pooled.
            first_chunk_z = zs[0]
            pooled_output = getattr(first_chunk_z, 'pooled', None)
            
            # Reconstruct concatenated embeddings
            all_z_embeddings = [z for z in zs]
            full_embeddings = torch.cat(all_z_embeddings, dim=1)
            
            return full_embeddings, pooled_output
        else:
            return torch.cat(zs, dim=1)

    def process_tokens(self, remade_batch_tokens, batch_multipliers):
        tokens = torch.as_tensor(remade_batch_tokens).to(self.device)
        
        if self.id_end != self.id_pad:
            for batch_pos in range(len(remade_batch_tokens)):
                try:
                    index = remade_batch_tokens[batch_pos].index(self.id_end)
                    tokens[batch_pos, index+1:] = self.id_pad
                except ValueError:
                    pass

        z = self.encode_with_transformers(tokens)
        
        pooled = getattr(z, 'pooled', None)
        
        emphasis_executor = get_emphasis_implementation(self.config.emphasis)()
        emphasis_executor.tokens = remade_batch_tokens
        emphasis_executor.multipliers = torch.as_tensor(batch_multipliers).to(self.device)
        emphasis_executor.z = z
        emphasis_executor.after_transformers()
        
        z = emphasis_executor.z
        
        if pooled is not None:
            z.pooled = pooled
            
        return z

class FrozenCLIPEmbedderWithCustomWords(TextConditionalModel):
    def __init__(self, wrapped_hf_model, config, hijack_instance):
        super().__init__(config, hijack_instance)
        self.wrapped = wrapped_hf_model # This is the HF CLIPTextModelWithProjection
        self.tokenizer = self.wrapped.tokenizer
        self.device = self.wrapped.transformer.device

        vocab = self.tokenizer.get_vocab()
        self.comma_token = vocab.get(',</w>', None)

        self.id_start = self.tokenizer.bos_token_id
        self.id_end = self.tokenizer.eos_token_id
        self.id_pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.id_end

    def tokenize(self, texts):
        return self.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]

    def encode_with_transformers(self, tokens):
        # self.wrapped.transformer is the actual HF CLIPTextModelWithProjection
        outputs = self.wrapped.transformer(
            input_ids=tokens.to(self.wrapped.transformer.device), 
            output_hidden_states=True # We need hidden states for stop_at_last_layers
        )
        
        stop_at = self.config.CLIP_stop_at_last_layers
        
        if stop_at > 1:
            # We want the output of a layer *before* the final one.
            # outputs.hidden_states is a tuple of (embedding_output, layer_1_output, layer_2_output, ...)
            # The last element is the last layer's output. -stop_at will count from the end.
            z = outputs.hidden_states[-stop_at]
            # In HF Transformers, the layer norm is applied before the output, so this is fine.
        else:
            z = outputs.last_hidden_state

        # Attach the pooled output to the tensor. The TextConditionalModel will handle it.
        z.pooled = outputs.pooler_output
        return z

# --- Main Public Interface Class ---

class ChunkerConfig:
    """A simple configuration class to replace webui's shared.opts"""
    def __init__(self):
        self.emphasis = "Original" # "Original" or "None"
        self.comma_padding_backtrack = 20
        self.CLIP_stop_at_last_layers = 1 # For Hunyuan's CLIP-L, 1 is correct (use final layer)

class WebuiPromptChunker:
    """
    A self-contained class to provide WebUI-style prompt chunking and emphasis parsing.
    """
    def __init__(self, text_encoder, tokenizer):
        """
        Initializes the chunker.
        
        Args:
            text_encoder: A HuggingFace CLIPTextModel or CLIPTextModelWithProjection.
            tokenizer: The corresponding HuggingFace CLIPTokenizer.
        """
        if not hasattr(text_encoder, 'device'):
            raise ValueError("The provided text_encoder must have a '.device' attribute.")
        
        # We create a wrapper to make the HF model look like WebUI's internal structure
        class LDMStyleCLIPWrapper:
            def __init__(self, hf_clip_model, hf_tokenizer):
                self.transformer = hf_clip_model
                self.tokenizer = hf_tokenizer

        self.config = ChunkerConfig()
        self.hijack_instance = StableDiffusionModelHijack()
        
        wrapped_model = LDMStyleCLIPWrapper(text_encoder, tokenizer)
        
        self.internal_encoder = FrozenCLIPEmbedderWithCustomWords(
            wrapped_hf_model=wrapped_model,
            config=self.config,
            hijack_instance=self.hijack_instance
        )

    def encode(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes a single prompt string, handling chunking and emphasis.
        
        For HunyuanDiT's specific needs, this will return the pooled output from the
        first chunk and the UN-CONCATENATED hidden states of the first chunk.
        The Hunyuan model does not use concatenated embeddings for its CLIP-L input.
        
        Args:
            prompt (str): The prompt string to encode.
            
        Returns:
            A tuple containing:
            - prompt_embeds (torch.Tensor): The hidden-state embeddings of the first chunk. Shape: (1, N, 1024)
            - pooler_output (torch.Tensor): The pooled output of the first chunk. Shape: (1, 1024)
        """
        if not isinstance(prompt, str):
            raise TypeError("This encoder is designed to process a single string prompt at a time.")
            
        # The forward method of TextConditionalModel is complex and designed for batching.
        # We can call its components directly for a simpler, single-prompt path.
        chunks, _ = self.internal_encoder.tokenize_line(prompt)
        
        if not chunks:
            first_chunk = self.internal_encoder.empty_chunk()
        else:
            first_chunk = chunks[0]
            
        # Process only the first chunk to get its embeddings and pooled output
        self.hijack_instance.fixes = [first_chunk.fixes]
        z = self.internal_encoder.process_tokens([first_chunk.tokens], [first_chunk.multipliers])
        
        prompt_embeds = z # This is the last_hidden_state of the first chunk
        pooler_output = getattr(z, 'pooled', None)
        
        if pooler_output is None:
            raise RuntimeError("Could not retrieve pooled_output from the CLIP encoder.")
            
        return prompt_embeds, pooler_output