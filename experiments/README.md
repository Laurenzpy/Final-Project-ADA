# Experiments (Legacy & Exploration)

This folder contains experimental, deprecated, or exploratory code created
during the development of the project. These scripts are **not part of the
final reproducible pipeline**, but are kept for transparency and to document
our iterative learning process.

## Final pipeline (relevant for reproduction)
The final and clean pipeline lives in the project root and consists of:
- `prepare_data.py` – prepares the final emoji-to-text datasets and splits
- `train_t5_generator.py` – fine-tunes a T5 sequence-to-sequence model
- `eval_final_generator.py` – evaluates the trained model on Stage 5 (IID) and Stage 6 (OOD)

## What you find here
Examples of content in this folder:
- Early seq2seq / RNN-based attempts
- BART-based emoji-to-text models
- A RAG-style ranker experiment
- Debugging and sanity-check scripts
- Legacy model checkpoints and vocabularies
- Dataset analysis and plotting utilities

## Important note
Files in this folder are **not guaranteed to be runnable or up to date**.
They are included to show the exploration process and methodological breadth
of the project.
