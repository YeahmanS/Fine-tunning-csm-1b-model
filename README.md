# Fine-tunning-csm-1b-model

This readme include errors i faced , issues in these tts models and suggestions



# **Error Debugging Report**

### **Project Context**

During fine-tuning of a multimodal (text + audio) transformer model using Hugging Face `Trainer` and `unsloth` processor, multiple preprocessing and training errors were encountered. These errors were due to sequence length mismatches, tokenizer padding/truncation conflicts, and dataset inconsistencies. Below is a systematic record of the issues and resolutions.

---

## **1. ValueError: expected sequence of length 416 at dim 1 (got 453)**

* **Cause**: Some preprocessed sequences (`input_ids` or `input_values`) exceeded the defined maximum length (416). Trainer requires all tensors in a batch to have the same length.
* **Fix**:

  * Added `truncation=True` in both `text_kwargs` and `audio_kwargs` of the processor.
  * Enforced consistent `max_length` for text and audio.
  * Verified with sanity checks (`len(ex['input_ids'])`).
  * **Final resolution**: deliberately fixed `max_length` to the **highest multiple of 8 within model limits (456)** so all samples fit without filtering.

---

## **2. Truncation and padding conflict warning**

```
Truncation and padding are both activated but truncation length (414) is not a multiple of pad_to_multiple_of (8)
```

* **Cause**: `max_length` was set to 414, but `pad_to_multiple_of=8`. 414 is not divisible by 8.
* **Fix**: Adjusted `max_length` to a multiple of 8 (456).

---

## **3. Error: Both padding and truncation were set**

```
Both padding and truncation were set. Make sure you only set one.
```

* **Cause**: The tokenizer was given **conflicting instructions**:

  * `padding="max_length"` + `pad_to_multiple_of=8` + `truncation=True`.
* **Fix**: Removed `pad_to_multiple_of` and kept:

  ```python
  "padding": "max_length",
  "truncation": True,
  "max_length": 456
  ```

---

## **4. 'list' object has no attribute 'shape'**

* **Cause**: Preprocessed dataset values (`input_ids`) were stored as **lists**, not tensors. Lists use `len()`, not `.shape`.
* **Fix**: Replaced `.shape[0]` with `len(example['input_ids'])` when validating dataset lengths.

---

## **5. Dataset inconsistency checks**

* **Observation**: Several examples exceeded the target max length (416), e.g.:

  ```
  Train example 7 is too long: 453
  Train example 26 is too long: 454
  Train example 44 is too long: 456
  ```
* **Fix**: Instead of filtering these samples, the preprocessing pipeline was updated to **pad/truncate all sequences to 456 tokens**, ensuring no data loss and consistent batching.

---

## **6. Clarification: Is only 416 allowed?**

* **Finding**: No, any value can be used for `max_text_length` as long as:

  * It is ≤ `model.config.max_position_embeddings` (e.g., 512).
  * Prefer multiples of 8 for efficiency.
* **Resolution**: Chose **456** (highest multiple of 8 covering the dataset) as the final fixed length.

---

# **Final Resolution**

* Preprocessing updated to enforce **consistent truncation + padding**:

  * `max_text_length = 456` (highest multiple of 8 within model limits).
  * `padding="max_length", truncation=True`.
  * Audio similarly padded/truncated.
* Dataset validated to ensure **all sequences have identical lengths**, preventing Trainer shape mismatch errors.
* This approach preserved **all training samples** without discarding long examples.

