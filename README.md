### Download Kaggle competition files

- create kaggle API Token on Kaggle.com; save under: ~/.kaggle/kaggle.json

```bash
chmod 600 ~/.kaggle/kaggle.json
```

- kaggle competitions download sartorius-cell-instance-segmentation
- unzip sartorius-cell-instance-segmentation data/input/

### Kaggle CLI support for commands

kaggle competitions {list, files, download, submit, submissions, leaderboard}
kaggle datasets {list, files, download, create, version, init}
kaggle kernels {list, init, push, pull, output, status}
kaggle config {view, set, unset}

### Training with a GPU

You may want to consider to reduce the total power consumption, and thereby reduce the vRAM may temp. To find the ideal configuration, observe your vRAM under heavy GPU load. Tooling on Linux is not good for doing so. I suggest you use windows HWinfo64

```bash
sudo nvidia-smi -i 0 -pl 230
watch -n 1 nvidia-smi
```

### Use docker kaggle container

In VS Code: Ctrl + Shift + P: Remote-Containers: Rebuild and Reopen in Container
Spin up jupyter notebook: from within the container ```bash jupyter notebook --allow-root```
Spin up tensorboard: from within the container ```bash tensorboard --logdir data/working/ --bind_all```

### Create & activate conda env if you are running without docker

```bash
conda create --name sartorius
conda activate sartorius
conda install --file requirements.txt
```

### Use docker Tensorflow-GPU container

```bash
docker build -t sartorius-tf .
```

```bash
docker run -it --rm -v $(realpath ~/repos/sartorius-cell-segmentation):/tf/notebooks --runtime=nvidia -p 8888:8888 -p 6006:6006 sartorius-tf
```

To use Tensorboard from the container above:

1. run the container, expose port 6006 (-p 6006:6006)
2. attach shell to the container, in the dir run  ```bash tensorboard --logdir=outputs --bind_all```
3. browse to localhost:6006

### Training with hydra configuration

inspect current config without running training:

```bash
python training.py --cfg job
```

or

```bash
python training.py --info config
```

inspect new overwritten config without running training:

```bash
python training.py training.model.EPOCHS=100 training.model.BATCH_SIZE=10 --cfg job
```

run with verbose logging:

```bash
python training.py hydra.verbose=[__main__]
```

Multi-run:

```bash
python training.py -m training=exp_0, exp_1
```
