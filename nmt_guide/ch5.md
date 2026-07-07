# 5. DATA PREPARATION

A translation model is only as good as the data it is trained on. In Machine Translation, the fundamental unit of data is the **Parallel Corpus**.

## What is a parallel corpus

A parallel corpus is a massive dataset of sentences in the source language perfectly aligned with their translations in the target language.

Example from an Amharic-English parallel corpus:
```json
{
  "am": "ውሻው ሰውን ነከሰ።",
  "en": "The dog bit the man."
}
```

To train a robust NMT model from scratch, you generally need millions of these sentence pairs.

## Data collection methods

Gathering clean, aligned parallel text is the hardest part of NMT for low-resource languages. Where do we find this data?

* **Religious texts (Bible, Quran):** Often the most widely translated texts in existence. They provide a massive, highly accurate parallel corpus, though the language can be archaic.
* **Legal documents:** The UN, EU, and African Union often mandate translations of all official documents.
* **Colloquial/conversational data:** Usually crowdsourced or collected from subtitle databases (like OpenSubtitles). Essential for casual translation.
* **Web scraped data (Common Crawl, OPUS):** Platforms like the OPUS repository aggregate translated web text. This data is ample but notoriously noisy.
* **News articles:** News agencies that publish in multiple languages (like BBC Amharic vs BBC English) are gold mines for high-quality, modern vocabulary.
* **Government documents & Social media:** Great for specialized terminology and modern slang.

## Data cleaning and normalization

Once you have your text, it will be messy. Amharic, for instance, has unique punctuation (like the Ethiopic full stop `።`), inconsistent comma usage, and varying spelling conventions for the same word.
A standard cleaning pipeline looks like:
1. **Remove HTML tags and URLs.**
2. **Normalize Punctuation:** Standardize quotation marks and hyphens.
3. **Filter Length:** Remove sentences longer than 150 words (they break the transformer memory limit).
4. **Filter Length Ratio:** If the Amharic sentence is 5 words and the English translation is 50 words, it's likely a bad alignment. Discard it.
5. **Deduplication:** Remove exact match duplicates to prevent the model from memorizing them.

## Tokenization strategies

We cannot feed raw text strings into the model. We must split them into tokens.
* **Word-level tokenization:** Splitting by spaces (e.g., `["The", "dog", "barked"]`). 
  * *Problem:* Fails on languages without spaces (Chinese) or highly morphological languages (Amharic) which can have millions of unique word combinations. The vocabulary becomes too huge.
* **Character-level tokenization:** Splitting into letters (`["T","h","e"...]`).
  * *Problem:* The sequence becomes too long, breaking the transformer memory, and it forces the model to learn spelling before it learns translation.
* **Subword tokenization (BPE, SentencePiece, WordPiece):** The industry standard. Very common words are kept as whole words, while rare words are broken into syllables or subwords.
  * For example, the very rare word "unhappiness" might become `["un", "##happi", "##ness"]`.
  * *SentencePiece* builds subwords directly from unformatted text, making it language-agnostic and perfect for languages like Amharic.

## Building a custom vocabulary

When training from scratch, you must train a tokenizer on your dataset to create the vocabulary mapping (e.g., Token 4992 = "house"). 
Usually, we restrict the vocabulary size to between 30,000 and 50,000 tokens per language to keep the Embedding layer memory requirements manageable.

## Fair vocabulary distribution for multilingual models

If you are training a single model to translate English into Amharic, Swahili, and Yoruba simultaneously, you must be careful. If your dataset is 90% English-Swahili and 10% English-Amharic, the WordPiece algorithm will dedicate 90% of the subword slots to Swahili, starving Amharic of good token representations.
To fix this, you must artificially balance the text corpus (e.g., using temperature upsampling) before training the tokenizer.

## Quantile based stratification for dataset splitting

When splitting data into Train/Validation/Test, standard random shuffles are dangerous for language. Why? Because sentence lengths vary wildly.
If your validation set happens to randomly get only short, 3-word sentences, you'll think your model is amazing, but it will fail in production on long paragraphs.
**Quantile Stratification** groups sentences by length (e.g., 1-10 words, 11-20 words, etc.), and randomly pulls exactly 10% from *each group* for validation. This guarantees your validation sets perfectly mirror the length distribution of your real-world data.

## Train/validation/test split strategy
* **Training Set:** (98%) Used to calculate gradients and adjust weights.
* **Validation Set:** (1%) Used to monitor training and verify the model isn't just memorizing the training data. Evaluated after every Epoch.
* **Test Set:** (1%) Kept completely hidden until the very end, used to report final BLEU scores.

<div class="page-break"></div>
