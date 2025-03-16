# NeuroMerge
## OS
```
ubuntu 20.04
```
### Recreate the Conda env
```
conda env create --file neuse_env.yml
```
### Activate the env
```
conda activate neuse
```
### Install the server (from the NeuSE dir)
```
/data/{YOUR_USER_NAME}/miniconda3/envs/neuse/bin/pip3 install serve/
```
### Set necessary environment variable
Set it to bypass some permission issue (don't know why yet).
```
export TEMP="/data/{YOUR_USER_NAME}/temp"

```
### Start the TorchServer server (port is 50051)
```
torchserve --start --model-store test_model --models mergegraph0429.mar --ts-config serve/ts.config --foreground
```
### Test using an example JSON
```
curl http://localhost:50051/predictions/mergegraph0409 -T test_merge_graph.json
```