# Sexism Detection in Low-Resource Languages
This repository contains the code for our master thesis as NTNU, SPRING 2026, where we investigated strategies for automatic sexism detection in Norwegian, a low-resource language. The project systematically compares cross-lingual transfer learning and machine translation across three multilingual transformer models (XLM-R Base, XLM-R Large, and mBERT) on both binary and multiclass sexism classification tasks.
The experiments are trained on a combined multilingual dataset built from the EXIST 2023 and EDOS datasets, and evaluated on our own annotated Norwegian test set of 1 084 comments collected from the VG comment section.
## Structure

detection/ — model training and evaluation across 8 task setups (binary and multiclass)

requirements.txt — project dependencies

## Data
The data folder is not included in this repository due to privacy restrictions and dataset licensing. To reproduce the experiments, the EXIST 2023, EDOS and test datasets must be obtained directly from their respective sources.
