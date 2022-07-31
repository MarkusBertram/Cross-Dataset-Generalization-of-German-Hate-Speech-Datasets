# Cross-Dataset-Generalization-of-German-Hate-Speech-Datasets

By Markus Bertram.

# Instructions

Install packages using:


```
pip install -r requirements.txt
```


Add Hate-Speech Datasets to /data/.. folders.


## Bias and Comparison Framework

### LSI-based intra-dataset class similarity

```
python lsi-based-similarity.py
```

### Word embedding based inter- and intra-dataset class similarity


```
python class-word-embedding-similarity.py
```


```
python dataset-word-embedding-similarity.py
```

### pmi-based word ranking for classes

```
python pmi-based-word-rankings.py
```

### Overarching topic modeling


```
git clone https://github.com/facebookresearch/fastText.git
cd fastText
sudo pip install .
```

Download cc.de.300.bin at:

https://fasttext.cc/docs/en/crawl-vectors.html

and move it to /embeddings folder.


```
python topic-model-hate-only.py
```


```
python topic-model-all-classes.py
```


## Comparison of Generalization Strategies

Change shared experiment settings at "basic_settings" key in  /settings/experiment_settings.json

Add experiments by adding dictionaries to the "exp_settings" list and change model specific parameters. Each experiment must have "exp_type", "exp_name", "feature_extractor", "task_classifier" and/or "domain_classifier".

Add or change models in src/model/.

Change experiment procedures in src/experiments/

Run experiments using:


```
python main.py
```


This will run all experiments in the experiments list in /settings/experiment_settings.json


Results are saving in /runs in Tensorboard SummaryWriter format. 

Use

```
tensorboard --logdir runs
```

to access experiment results.

# References

WICH, Maximilian, et al. Bias and comparison framework for abusive language datasets.  *AI and Ethics* , 2021, S. 1-23.

MOZAFARI, Marzieh; FARAHBAKHSH, Reza; CRESPI, Noel. A BERT-based transfer learning approach for hate speech detection in online social media. In:  *International Conference on Complex Networks and Their Applications* . Springer, Cham, 2019. S. 928-940.

GANIN, Yaroslav, et al. Domain-adversarial training of neural networks.  *The journal of machine learning research* , 2016, 17. Jg., Nr. 1, S. 2096-2030.

SHU, Rui, et al. A dirt-t approach to unsupervised domain adaptation.  *arXiv preprint arXiv:1802.08735* , 2018.

SAITO, Kuniaki, et al. Semi-supervised domain adaptation via minimax entropy. In:  *Proceedings of the IEEE/CVF International Conference on Computer Vision* . 2019. S. 8050-8058.

LI, Bo, et al. Learning invariant representations and risks for semi-supervised domain adaptation. In:  *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* . 2021. S. 1104-1113.

ZHAO, Han, et al. Adversarial multiple source domain adaptation.  *Advances in neural information processing systems* , 2018, 31. Jg.

PENG, Xingchao, et al. Moment matching for multi-source domain adaptation. In:  *Proceedings of the IEEE/CVF international conference on computer vision* . 2019. S. 1406-1415.
