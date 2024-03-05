# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test

Train ActFormer with 8 GPUs

```
./tools/dist_train.sh ./projects/configs/Actformer/Actformer_base.py 8
```

Eval ActFormer with 8 GPUs

```
./tools/dist_test.sh ./projects/configs/Actformer/Actformer_base.py ./path/to/ckpts.pth 8
```

# Visualization

see [visual.py](../tools/analysis_tools/visual.py)
