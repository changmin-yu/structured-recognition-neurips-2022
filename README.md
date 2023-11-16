# structured-recognition-neurips2022

Python codes for the Structured Recognition non-linear Gaussian Process Factor Analysis model in our NeurIPS 2022 paper: [Structured Recognition for Generative Models with Explaining Away](https://openreview.net/forum?id=ySB7IbdseGC). 

For any question regarding the paper or the code, please contact changmin.yu98[at]gmail.com and maneesh[at]gatsby.ucl.ac.uk

For executing the codes, try running:

- SR-nlGPFA
```
python experiments/place_cell.py --model sr-nlgpfa
```

- treeSRVAE
```
python treeSRVAE/treeSRVAE.py --dataset bar_test --tree-structured-gen True
```

- gmmSRVAE
```
python gmmSRVAE/models/SRVAE.py --dataset pinwheel --full-dependency True --seed 0
```

If you find the paper or the code helpful for your research, please consider citing us with the following format:

```
@inproceedings{yustructured,
    title={Structured Recognition for Generative Models with Explaining Away},
    author={Yu, Changmin and Soulat, Hugo and Burgess, Neil and Sahani, Maneesh},
    booktitle={Advances in Neural Information Processing Systems}
}
```
