# README

Ensure that the following packages are pre-installed in your Python environment:
```
pip install ray
pip install aim==3.16.2
pip install hyperopt
```

Before running the hyperparameter tuning, you need to start a Ray cluster. In your head node, execute the following command:
```
ray start --head --port=6379
```

Note that the head node is automatically a worker node. If you want to start a distributed cluster with more worker nodes for hyperparameter tuning, execute the following command in other nodes:
```
ray start --address=HEAD_NODE_IP:6379
```

You need to specify the hyperparameters and their ranges in `/research/RevMLPMixer/tune_config/tune_test.yaml`. Currently, two types of range specification are supported: `choice` and `uniform`. Here is an example:
```
batch_size:
  type: choice
  values: [100, 500, 1000]

lr:
  type: uniform
  min: 0.2
  max: 2.0
```

Also, Ray requires to specify a full path for distributed tuning. You need to specify your absolute data path in `research/RevMLPMixer/config/train_mnist.yaml`:
```
Setup:
  Data:
    batch_size: 1000
    batch_size_test: 100
    name: mnist
    path: **ABSOLUTE_PATH**
    subset: false
    num_classes: 10
    add_random: false
    random_proj: false
```

Once the above setup is ready, you can run hyperparameter tuning by executing the following command:
```
python3 tune.py --config config/train_mnist.yaml
```

You can start the Aim dashboard to monitor the training process:
```
aim up --repo=/proj/HOME/ray_results/ray_tune
```
