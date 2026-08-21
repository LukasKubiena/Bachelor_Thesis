# Topic or Proficiency? Topic Associations in CEFR-Labelled Corpora

Bachelor thesis, Cognitive Science, Aarhus University.
Lukas Maria Kubiena, supervised by Ross Deans Kristensen-McLachlan.


## Repository

```
in/     input data
src/    all the analysis code
out/    everything the scripts produces
tests/  unit tests for the statistics functions
```

## Data

All data comes from [UniversalCEFR](https://huggingface.co/UniversalCEFR)
(Imperial et al., 2025). None of it is committed, because two of the corpora
are licensed for academic use only and can't be redistributed. They need to be accessed with an academic license.

**Open corpora** downloads automatically 

**Gated corpora**

DEplain-APA: <https://zenodo.org/records/7674560>
APA-LHA: <https://zenodo.org/records/5148163>

Unzip both into `in/raw/`
run `python src/00_convert_raw.py` (converts them into the UniversalCEFR schema)


## Reproduction

```bash
git clone <repo-url>
cd thesis
bash setup.sh          # venv + requirements
source env/bin/activate
bash run_all.sh de     # german pipeline
bash run_all.sh en     # english replication
```

The pinned environment requires Python 3.10 or newer.


### The scripts

| script | what |
|---|---|
| `00_convert_raw.py` | puts the two datasets into the same format |
| `01_load_data.py` | downloads and cleans the datasets |
| `01b_descriptives.py` | shows basic information about the datasets |
| `01c_near_duplicates.py` | checks for texts that are very similar |
| `02_topic_model.py` | finds the main topics in the texts |
| `03_confound_analysis.py` | checks if topics are related to language level or dataset |
| `03b_association_extended.py` | runs some extra statistical tests |
| `03c_length_stratified.py` | checks the results for different text lengths |
| `04_topic_only_baseline.py` | tests if language level can be predicted from topic alone |
| `05_robustness.py` | checks if the results stay similar with different numbers of topics |
| `06_length_benchmark.py` | compares text length, topic, and both together |
| `07_cross_corpus.py` | checks if the results also work across different datasets |
| `07b_topic_stratified.py` | does an extra check of the topic results |
| `07c_topic_overlap_within_corpus.py` | checks topic effects within each dataset |
| `08_topic_model_quality.py` | checks how good and stable the topic model is |
| `08b_encoder_sensitivity.py` | checks the results with two other text models |
| `08c_model_sensitivity.py` | checks the results with other topic models |
| `09_build_manifest.py` | saves all main results in one file |
| `10_figures.py` | creates all the figures |

