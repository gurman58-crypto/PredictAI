# Preprocessing

One module, `ml_models/preprocessing/text_processing.py`, is used by dataset preparation, all training scripts, the evaluation script and the API. That is how training and inference are guaranteed to see identical input. `vocab.json` records the preprocessing version and the tokenizer pattern; `Vocabulary.load` refuses to load a vocabulary produced by a different tokenizer.

## Pipeline

1. **Normalise:** Unicode NFKC, typographic quotes/dashes → ASCII, lower-case, collapse whitespace.
2. **Tokenise** (NLTK `RegexpTokenizer`): numbers (`1,000`, `3.5`), words with internal hyphens (`role-playing`), the clitics `'s 're 've 'll 'd 'm 't` as their own tokens, and the punctuation `. , ; : ! ? ( ) "`. Anything else is dropped.
3. **Paragraph end** becomes `<eos>`. Headings (`= Title =`) are skipped.
4. **Vocabulary:** the 10 000 most frequent training tokens (min. frequency 2) plus `<pad>`, `<unk>`, `<eos>`. Fitted on the training split only. Everything else → `<unk>`.
5. **Source-specific clean-up (training data only):** WikiText pre-tokenises text (`1 @,@ 000`, `did n't`); `prepare_dataset.py` undoes this so the corpus looks like text a user types.

## At inference

`TextPreprocessor.parse(text_before_cursor)`:
- keeps only the last 2000 characters and the last 30 tokens;
- if the text ends **inside a word** (no trailing space) that word is a *prefix*: the context excludes it and suggestions are filtered to words starting with it (mode `completion`);
- a trailing word that is already a complete vocabulary word with no longer extension (e.g. `sat`) is treated as finished;
- empty or all-unknown input uses the sentence-start context `<eos>`;
- `<unk>`, `<pad>`, `<eos>` and punctuation are never suggested.

Suggestions are lower-case; the editor re-applies capitalisation only to the text it inserts (sentence starts, or to match what the user typed), never to existing text.
