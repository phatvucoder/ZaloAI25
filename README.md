# ZaloAI25



## How to


1. Build docker 

```
sudo docker build -t zac2025:v1 .
```


2. Run docker

```
sudo docker run --gpus '"device=0"' \
    -v /path/to/testdata:/data \
    -v /home/user/output:/result \
    zac2025:v1

```