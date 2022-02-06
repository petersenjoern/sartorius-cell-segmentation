### Download Kaggle competition files
- create kaggle API Token on Kaggle.com; save under: ~/.kaggle/kaggle.json

```bash
chmod 600 ~/.kaggle/kaggle.json
```

- kaggle competitions download sartorius-cell-instance-segmentation
- unzip data/sartorius-cell-instance-segmentation data/

### Kaggle CLI support for commands:
kaggle competitions {list, files, download, submit, submissions, leaderboard}
kaggle datasets {list, files, download, create, version, init}
kaggle kernels {list, init, push, pull, output, status}
kaggle config {view, set, unset}

### Create & activate conda env if you are running without docker
```bash
conda create --name sartorius
conda activate sartorius
conda install --file requirements.txt
```

### Training with a GPU with docker:

You may want to consider to reduce the total power consumption, and thereby reduce the vRAM may temp. To find the ideal configuration, observe your vRAM under heavy GPU load. Tooling on Linux is not good for doing so. I suggest you use windows HWinfo64
```bash
sudo nvidia-smi -i 0 -pl 230
watch -n 1 nvidia-smi
```

```bash
docker build -t sartorius-tf .
```

```bash
docker run -it --rm -v $(realpath ~/repos/sartorius-cell-segmentation):/tf/notebooks --runtime=nvidia -p 8888:8888 -p 6006:6006 sartorius-tf
```

To use Tensorboard from the container above:
- run the container, expose port 6006 (-p 6006:6006)
- attach shell to the container, in the dir run  ```bash tensorboard --logdir=outputs --bind_all```
- browse to localhost:6006