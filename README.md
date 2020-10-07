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
 - [ ] Tensorboard writer should be used for easy debug
 - [ ] All code does not consider batch_size larger than 1
 - [x] Evaluation/Test code should be implemented
 - [ ] Evaluation code should use ground truth homography
 - [ ] Optimizer should be changed to Adam
 
## Acknowledgments：
 - Based on a paper which is not yet published, UnSuperPoint: <https://arxiv.org/abs/1907.04011v1>
 - Based on code: <https://github.com/lydproject/UnSuperPoint_Project>
