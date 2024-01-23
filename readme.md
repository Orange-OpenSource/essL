## Towards efficient self-supervised representation learning in speech processing

Efficient self-supervised learning (ESSL) model to learn speech representations. The focus is primarily on computational costs, limiting the resources available for pretraining, and evaluating it with ASR as the downstream task.

### Installation

Create a virtual environment and activate it
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev python3-tk
sudo pip3 install -U virtualenv
virtualenv --system-site-packages -p python3 ~/torch21
source ~/torch21/bin/activate
```

Now, let's install Python packages
```bash
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

### Experiments

For replicating results, settings are stored in _config/efficientssl.py_. Experiments use an NVIDIA GeForce RTX 3090 GPU, with 24 GB of RAM. Make sure you have at least 24 GB available in your GPU to avoid out-of-memory exceptions. Otherwise, decrease _self.batch_length_ in _config/efficientssl.py_ and change training steps accordingly.

Download datasets:
```bash
python3 main.py --run download
```
Pretrain and finetune the model for ASR:
```bash
python3 main.py --run train
```
Results are stored in the log file _data/marcel_exp0_resutls.pt_log_. Checkpoints are stored under _data/_. 

### Reference
```
@inproceedings{lugo2024towards,
  title={Towards efficient self-supervised representation learning in speech processing},
  author={Lugo, Luis and Vielzeuf, Valentin},
  booktitle={Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics},
  year={2024},
}
```

### License
This project is licensed under the terms of the [MIT](https://opensource.org/licenses/MIT) license. See the LICENSE file for more information.