# UnSuperPoint
## Conducted as a part of URP @ KAIST
## How to train
```
python main.py train <CONFIGURATION> <EXPORTNAME>
```

## How to test
```
python main.py export <CONFIGURATION> <EXPORTNAME> <MODELNAME>
python evaluation.py <EXPORTPATH>
```

## Example command
```
python main.py train config/train.yaml unsuperpoint
python main.py export config/test.yaml unsuperpoint UnsuperPoint_single_8000.pkl
python evaluation.py logs/unsuperpoint/predictions --repeatibility --outputImg --homography --plotMatching
```

## Tensorboard
```
tensorboard --logdir=./runs/ [--host | static_ip_address] [--port | 6008]
```

## TODO
 - [x] Tensorboard writer should be used for easy debug
 - [x] Batch training should be more efficiently mangaed
 - [ ] Evaluation code should be implemented
 - [x] Test code should be implemented
 - [x] Test code should use ground truth homography
 - [X] Optimizer should be changed to Adam
 - [x] Unixy loss was debugged
 
## Acknowledgments
 - Based on a paper which is not yet published, UnSuperPoint: <https://arxiv.org/abs/1907.04011v1>
 - Based on code: <https://github.com/lydproject/UnSuperPoint_Project>
