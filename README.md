<i>This benchmark is still in progress:
- We keep running experiments on different datasets
- We still prepare RecTools models for release in the framework (documentaion, tests and tutorials are not yet finished)</i>

# [RecTools](https://github.com/MobileTeleSystems/RecTools) transformers benchmark results

RecTools models (SASRec and BERT4Rec) results were computed using this fork os the [original repository](https://github.com/asash/bert4rec_repro). Results for other implementations were taken from the original paper [A Systematic Review and Replicability Study of BERT4Rec for Sequential Recommendation](https://arxiv.org/abs/2207.07483)



**RecTools implementations achieve highest metrics on both datasets out of all available implementations from the original paper.**

### ML-20M Dataset results
|Model       |Pop-sampled Recall@10|Pop-sampled NDCG@10| Recall@10| NDCG@10| Training time  |
|--------------------------|--------------------------------|---------------------------------|-----------|---------|----------------|
|MF-BPR          |0.6126|  0.3424  | 0.0807    |  0.0407 | 197    |
|SASRec original |0.6582|  0.4002 | 0.1439    |  0.0724 | 3635    |
|BERT4Rec original |0.4027|  0.2193  | 0.0939    |  0.0474 | 6,029    |
|BERT4Rec RecBole |0.4611|  0.2589  | 0.0906    |  0.0753 | 519,666    |
|BERT4Rec BERT4Rec-VAE |0.7409|  0.5259  | 0.2886    |  0.1732 | 23,030    |
|BERT4Rec ber4rec_repro |0.7127|  0.4805  | 0.2393    |  0.1310 | 44,610    |
|BERT4Rec ber4rec_repro (longer seq) |0.7268|  0.4980  | 0.2514    |  0.1456 | 39,632    |
|**SASRec RecTools** | <u>0.7562</u> |  <u>0.5422</u>   | <u>0.2994</u>   |  <u>0.1834</u> | *    |
|**BERT4Rec RecTools** |-|  -  | -    |  - | *    |
Reported BERT4Rec|0.7473|  0.5340  | N/A    |  N/A | N/A    |

### ML-1M Dataset results
|Model       |Pop-sampled Recall@10|Pop-sampled NDCG@10| Recall@10| NDCG@10| Training time  |
|--------------------------|--------------------------------|---------------------------------|-----------|---------|----------------|
|MF-BPR          |0.5134|  0.2736  | 0.0740    |  0.0377 | 58    |
|SASRec original |0.6370|  0.4033 | 0.1993    |  0.1078 | 316    |
|BERT4Rec original |0.5215|  0.3042  | 0.1518    |  0.0806 | 2,665    |
|BERT4Rec RecBole |0.4562|  0.2589  | 0.1061    |  0.0546 | 20,499    |
|BERT4Rec BERT4Rec-VAE |0.6698|  0.4533  | 0.2394    |  0.1314 | 1,085    |
|BERT4Rec ber4rec_repro |0.6865|  0.4602  | 0.2584    |  0.1392 | 3,679    |
|BERT4Rec ber4rec_repro (longer seq) |0.6975|  0.4751  | 0.2821    |  0.1516 | 2,889    |
|DeBERTa4Rec ber4rec_repro | - |  - | 0.290    |  0.159 | -    |
|ALBERT4Rec ber4rec_repro | - |  - | 0.300    |  0.165 | -    |
|**SASRec RecTools** |-|  -  | -    |  <u>0.1778</u> | 535*    |
|**BERT4Rec RecTools** |-|  -  | -    |  0.1558 | 369*    |
Reported BERT4Rec|0.6970|  0.4818  | N/A    |  N/A | N/A    |


### Notes
- To assure same settings with the paper experiments with RecTools models were run together with BERT4Rec-VAE model from published paper. We achieved the same metric results for this model as were reported in the original paper.
- RecTools models training time was computed relative to BERT4Rec-VAE training time during simultaneous experiments on our hardware. To make that our model training time is comparable to those reported in the paper, we compute it as a product of reported BERT4Rec-VAE trainig time and our model relative difference which was obtained during actual experiments.
- RecTools model params were set equal to BERT4Rec-VAE model params reported in the paper. 
- RecTools models were trained for 200 epochs on each dataset.
- SASRec model from Rectools was trained on softmax loss.

# Reproduce our results:

## Installation 

Create working directory:

```
mkdir aprec_repro
cd aprec_repro
```

Clone repositories:
```
git clone https://github.com/asash/b4rvae.git b4rvae
```

```
git clone https://github.com/blondered/bert4rec_repro.git aprec
```

Optionally clone RecTools repository if you are not using a released version:
```
git clone https://github.com/MobileTeleSystems/RecTools.git
```

Install required packages: 
If you didn't clone RecTools repository, make sure `rectools` package is uncommented in requirements.txt. Please check that rectools version is the one that has transformer models released.
```
cd aprec
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt 
```


If you did clone RecTools repository, install it in virtual environment:
```
cd ../RecTools
git checkout feature/checkpoints  # or whatever branch you need
pip install -e RecTools
cd ../aprec
```

## Reproducing benchmark results

To reproduce RecTools and BERT4Rec-VAE results from the tables above do the following

Open aprec/evaluation directory:
```
cd evaluation
```
Run experiments on different datasets:

```
sh run_n_experiments.sh configs/rectools/ml_1m.py
```
```
sh run_n_experiments.sh configs/rectools/ml_20m.py
```

Each experiemnt result you can find in the directory: `aprec/evaluation/results/<experiment_id>/experiment_.json`



## Running and analyzing experiments guide


For experiment reproducibility purposes run_n_experiments.sh requires that all code in the repository is commited before running the experiment. The framework records commit id in the experiment results, so that it is always possible to return to exatly the same state of the repository and rerun the experiment. If you want to override this behaviour, set environment variable CHECK_COMMIT_STATUS=false. For example:

```
CHECK_COMMIT_STATUS=false sh run_n_experiments.sh configs/ML1M-bpr-example.py
```

You can tail experiment stdout:
```
tail -f run_n_experiments.sh ./results/latest_experiment/stdout
```

You may also check results of the models that already have been evaluated using ```analyze_experiment_in_progress.py``` script: 

```
python3 analyze_experiment_in_progress.py ./results/latest_experiment/stdout
```

# Original README is below:

### This is a joint code repository for two papers published at 16th ACM Conference on Recommender Systems 
(Seattle, WA, USA, 18th-23rd September 2022)

**1. Effective and Efficient Training for Sequential Recommendation using Recency Sampling**
[Aleksandr Petrov](https://scholar.google.com/citations?user=Cw7DY8IAAAAJ&hl=en) (University of Glasgow, United Kingdom)
and [Craig Macdonald](https://scholar.google.com/citations?user=IBjMKHQAAAAJ&hl=en) (University of Glasgow, United Kingdom)
([paper link](https://arxiv.org/abs/2207.02643)). 

**2. A Systematic Review and Replicability Study of BERT4Rec for Sequential Recommendation**
Aleksandr Petrov (University of Glasgow, United Kingdom) and Craig Macdonald (University of Glasgow, United Kingdom)
([paper link](https://arxiv.org/abs/2207.07483)). 


The code includes benchmark of three available BERT4Rec implementations, as well as our own implementation of BERT4Rec based on Hugging Face Transformers library. 


if you use any part of this code, please cite one or both papers  of the papers using following BibTex: 

```
@inproceedings{petrov2022recencysampling,
  title={Effective and Efficient Training for Sequential Recommendation using Recency Sampling},
  author={Petrov, Aleksandr and Macdonald, Craig},
  booktitle={Sixteen ACM Conference on Recommender Systems},
  year={2022}
}
```


```
@inproceedings{petrov2022replicability,
  title={A Systematic Review and Replicability Study of BERT4Rec for Sequential Recommendation},
  author={Petrov, Aleksandr and Macdonald, Craig},
  booktitle={Sixteen ACM Conference on Recommender Systems},
  year={2022}
}
```


## Expected BERT4Rec results on Standard datasets
We hope our work becomes a resource for verifying expected BERT4Rec results. When you use BERT4Rec as a baseline, the numbers you should expect are as follows: 

### Recall@10 (also known as HIT@10)
|Dataset       |Uniformly Sampled, 100 negatives|Popularity sampled, 100 negatives| Unsampled |
|--------------------------|--------------------------------|---------------------------------|-----------|
|Movielens-1M<sup>1</sup>  |0.8039                          |  0.6975                         | 0.2821    |
|Movielens-20M<sup>2</sup> |0.9453                          |  0.7409                         | 0.2886    |



### NDCG@10 
|Dataset       |Uniformly Sampled, 100 negatives|Popularity sampled, 100 negatives| Unsampled |
|----------------------------------|--------------------------------|---------------------------------|-----------|
|Movielens-1M<sup>1</sup>          |0.6008                          |  0.4751                         | 0.1516    |
|Movielens-20M<sup>2</sup>         |0.7827                          |  0.5259                         | 0.1732    |

If your results are lower by a large margin you are likely using underfit version of BERT4Rec. 

<sup>1</sup> Result achieved using our model from this repository, based on Hugging Face transformers

<sup>2</sup> Result achieved using <a href="https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch">BERT4Rec-VAE</a> implementation


## Installation 
The instruction has been tested on an Ubuntu 22.04 LTS machine with an NVIDIA RTX 3090 GPU. 

Please follow this step-by-step instruction to reproduce our results


### 1. Install Anaconda environment manager: 

If you don't have anaconda installed in your system, please follow the instruction https://docs.anaconda.com/anaconda/install/linux/

### 2. Create the project working directory
```
mkdir aprec_repro
cd aprec_repro
```


### 3. Create an anaconda environment with necessary package versions:
```
conda create -y -c pytorch -c conda-forge --name aprec_repro python=3.9.12 cudnn=8.2.1.32 cudatoolkit=11.6.0
conda install pytorch-gpu=1.10.0
conda install tensorflow-gpu=2.6.2
conda install gh=2.1.0 
conda install expect=5.45.4
```

### 4. Add working working directory to the PYTHONPATH of the anaconda environment: 
```
conda env config vars set -n aprec_repro PYTHONPATH=`pwd`
```

### 5. Activate the environment
```
conda activate aprec_repro
```

### 6. Install python packages in the environment: 
```
pip3  install "jupyter>=1.0.0" "tqdm>=4.62.3" "requests>=2.26.0" "pandas>=1.3.3" "lightfm>=1.16" "scipy>=1.6.0" "tornado>=6.1" "numpy>=1.19.5" "scikit-learn>=1.0" "lightgbm>=3.3.0" "mmh3>=3.0.0" "matplotlib>=3.4.3" "seaborn>=0.11.2" "jupyterlab>=3.2.2" "telegram_send>=0.25" "transformers>=4.16.1" "recbole>=1.0.1" "wget>=3.2" "pytest>=7.1.2" "pytest-forked>=1.4.0" "setuptools==59.5.0"
```

### 7. Clone necessary github repos into workdir: 
#### 7.1 our patched version of BERT4Rec-VAE-Pytorch version. 
The patch allows the code to be used as a library: We have to keep this repo separately due to licenses incompatibility


```
git clone git@github.com:asash/b4rvae.git b4rvae
```

#### 7.2 This github repo: 

```
git clone git@github.com:asash/bert4rec_repro.git aprec
```

### 8. Download Yelp Dataset
8.1 Create folder for the dataset: 

```
  mkdir -p aprec/data/yelp
```

8.2 Go to https://www.yelp.com/dataset

8.3 Click "Download" button

8.4 Fill the form 

8.5 Donwload JSON version of the dataset. 

8.6 Put the `yelp_dataset.tar` file to the freshly created dataset folder `aprec/data/yelp`




### 9. Test the code
Your environment is now ready to run the experiments. To ensure that everything works correctly, run the tests:

```
cd aprec/tests
pytest --verbose --forked . 
```

# Runnig experiments

### 1.  Go to aprec evaluation folder: 
```
cd <your working directory>
cd aprec/evaluation
```

### 2. Run example experiment: 
you need to run `run_n_experiments.sh` with the experiment configuration file. Here is how to do it with an example configuration: 


```
sh run_n_experiments.sh configs/ML1M-bpr-example.py
```

**Note**

For experiment reproducibility purposes run_n_experiments.sh requires that all code in the repository is commited before running the experiment. The framework records commit id in the experiment results, so that it is always possible to return to exatly the same state of the repository and rerun the experiment. 
If you want to override this behaviour, set environment variable `CHECK_COMMIT_STATUS=false`. For example: 

```
CHECK_COMMIT_STATUS=false sh run_n_experiments.sh configs/mf_config.py
```

### 3. Analyze experiment results
Experiment may be running for some time. 
To watch what's going on with the experiment in realtime, just tail the the experiment stdout (the link to stdout is given as an output of `run_n_experiments.sh`)

```
tail -f run_n_experiments.sh ./results/<experiment_id>/stdout
````

You may also check results of the models that already have been evaluated using ```analyze_experiment_in_progress.py``` script: 

```
python3 analyze_experiment_in_progress.py ./results/<experiment_id>/stdout
```
This will give pretty formatted representation of the results: 

![screenshot](images/example_analysis.png)


# Hugging Face Transformers based implementation of BERT4Rec. 

Many models in this framework are based on `DNNSequentialRecommender` - basic class for many deep neural networks based models ( [code](https://github.com/asash/bert4rec_repro/blob/main/recommenders/dnn_sequential_recommender/dnn_sequential_recommender.py)).

To use this code, you need to configure it with specific model architecture, data splitting strategy, loss function and so on. Here is an example, how to configure it for BERT4Rec with Hugging Face based model:

```python
         model = BERT4Rec(embedding_size=32) # model architeture
        recommender = DNNSequentialRecommender(model, train_epochs=10000, early_stop_epochs=50000,
                                               batch_size=5,
                                               training_time_limit=10, 
                                               loss = MeanPredLoss(), # Loss function. Hugging Face model computes loss inside the model, so we just use its output. 
                                               debug=True, sequence_splitter=lambda: ItemsMasking(), #Items Masking - strategy to separate labels from sequence used by BERT4Rec 
                                               targets_builder=lambda: ItemsMaskingTargetsBuilder(), #Also items masking - this class is used to convert splitted data to model targets. 
                                               val_sequence_splitter=lambda: ItemsMasking(force_last=True), #How we split data for validation: only mask element in the sequence. 
                                               metric=MeanPredLoss(), 
                                               pred_history_vectorizer=AddMaskHistoryVectorizer(), #How we convert sequences for inference. 
                                               )
```

More examples how to configure models can be found in [tests](https://github.com/asash/bert4rec_repro/blob/main/tests/test_own_bert.py) and [configuration files](https://github.com/asash/bert4rec_repro/blob/main/evaluation/configs/bert4rec_repro_paper/common_benchmark_config.py). 

The model itself is quite simple and can be found in this [file](https://github.com/asash/bert4rec_repro/blob/main/recommenders/dnn_sequential_recommender/models/bert4rec/bert4rec.py). We also provide implementations of ALBERT4Rec ([code](https://github.com/asash/bert4rec_repro/blob/main/recommenders/dnn_sequential_recommender/models/albert4rec/albert4rec.py), [example config](https://github.com/asash/bert4rec_repro/blob/main/evaluation/configs/bert4rec_repro_paper/ml_1m_albert.py)) and DeBERTa4Rec ([code](https://github.com/asash/bert4rec_repro/blob/main/recommenders/dnn_sequential_recommender/models/deberta4rec/deberta4rec.py), [test](https://github.com/asash/bert4rec_repro/blob/main/tests/test_deberta4rec.py), [example config](https://github.com/asash/bert4rec_repro/blob/main/evaluation/configs/bert4rec_repro_paper/ml_1m_deberta.py))

# Experiment configrations for Recency Sampling paper:  

| Experiment                                       | Dataset | Training time limit (hours) | Models in the experiment                                                                                                            |
|----------------------------------|-------|--------|-------------------------------------------------------------------------------------------------------------------------------------|
| configs/configs/booking_benchmark.py   | Booking.com  | 1  | Baselines (Top popular; MF-BPR, SASRec); BERT4Rec-1h; GRU4rec, Caser, Sasrec with lambdarank/bce loss and Continuation/RSS objective |
| configs/configs/booking_benchmark_bert4rec16h.py   | Booking.com | 16  | BERT4Rec-16h |
| configs/configs/yelp_benchmark.py   | yelp  | 1 |  Baselines (Top popular; MF-BPR, SASRec); BERT4Rec-1h; GRU4rec, Caser, Sasrec with lambdarank/bce loss and Continuation/RSS objective |
| configs/configs/yelp_benchmark_bert4rec_16h.py  |  yelp | 16 | BERT4Rec-16h |
| configs/configs/ml_benchmark20m.py   | MovieLens-20M | 1 | Baselines (Top popular; MF-BPR, SASRec); BERT4Rec-1h; GRU4rec, Caser, Sasrec with lambdarank/bce loss and Continuation/RSS objective |
| configs/configs/ml_benchmark20m_bert4rec16h.py   |  MovieLens-20M | 16 | BERT4Rec-16h |
| configs/configs/gowalla_benchmark.py   | Gowalla  | 1 | Baselines (Top popular; MF-BPR, SASRec); BERT4Rec-1h; GRU4rec, Caser, Sasrec with lambdarank/bce loss and Continuation/RSS objective |



# Experiment configurations for reproducing the Replicability paper results: 
## RQ1. Default BERT4Rec configurations


| Experiment                                       | Dataset | Models in the experiment                                                                                                            |
|--------------------------------------------------|---------|-------------------------------------------------------------------------------------------------------------------------------------|
| configs/bert4rec_repro_paper/ml_benchmark1m.py   | MovieLens-1M   | Baselines (MF-BPR, SASRec), 5 BERT4Rec versions with default configuration (Original, BERT4Rec-Vae, Recbole, Ours, Ours-longer-seq) |
| configs/bert4rec_repro_paper/beauty_benchmark.py | Beauty  | Baselines (MF-BPR, SASRec), 4 BERT4Rec versions with default configuration (Original, BERT4Rec-Vae, Recbole, Ours)                  |
| configs/bert4rec_repro_paper/steam_benchmark.py  | Steam   | Baselines (MF-BPR, SASRec), 4 BERT4Rec versions with default configuration (Original, BERT4Rec-Vae, Recbole, Ours)                  |
| configs/bert4rec_repro_paper/ml_20m_benchmark.py | MovieLens-20M  | Baselines (MF-BPR, SASRec), 5 BERT4Rec versions with default configuration (Original, BERT4Rec-Vae, Recbole, Ours, Ours-longer-seq) |


## RQ2. Original BERT4Rec training time. 

| Experiment                                       | Dataset | Models in the experiment                                                                                                            |
|--------------------------------------------------|---------|-------------------------------------------------------------------------------------------------------------------------------------|
| ml_1m_original_200000steps.py                    | ML-1M   | Original BERT4Rec (200000 training steps)                                                                                           |
| ml_1m_original_400000steps.py                    | ML-1M   | Original BERT4Rec (400000 training steps)                                                                                           |
| ml_1m_original_800000steps.py                    | ML-1M   | Original BERT4Rec (800000 training steps)                                                                                           |
| ml_1m_original_1600000steps.py                   | ML-1M   | Original BERT4Rec (1600000 training steps)                                                                                          |
| ml_1m_original_3200000steps.py                   | ML-1M   | Original BERT4Rec (3200000 training steps)                                                                                          |
| ml_1m_original_6400000steps.py                   | ML-1M   | Original BERT4Rec (6400000 training steps)                                                                                          |
| ml_1m_original_12800000steps.py                  | ML-1M   | Original BERT4Rec (12800000 training steps)                                                                                         |


## Systematic review of BERT4Rec and SASRec
Spreadsheet with systematic comparison of BERT4Rec and SASRec can be found by the link: 
https://github.com/asash/bert4rec_repro/blob/main/Systematic%20Review.xlsx?raw=true


## RQ3. Other Transformers. 
| Experiment                                       | Dataset | Models in the experiment                                                                                                            |
|--------------------------------------------------|---------|-------------------------------------------------------------------------------------------------------------------------------------|
| configs/bert4rec_repro_paper/ml_1m_deberta.py    | ML-1M   | DeBERTa4Rec                                                                                                                         |
| configs/bert4rec_repro_paper/ml_1m_albert.py     | ML-1M   | ALBERT4Rec                                                                                                                          |

# Benchmark results (on MovieLens-1M dataset):
![Benchmark](images/models_benchmark.svg)
