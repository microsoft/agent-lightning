conda create -n retriever python=3.10
conda activate retriever

# we recommend installing torch with conda for faiss-gpu
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini

## install the gpu version faiss to guarantee efficient RL rollout
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

## API function
pip install uvicorn fastapi

git clone https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus

mkdir data
cd data
wget https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus/resolve/main/wiki-18.jsonl.gz
wget https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index/resolve/main/part_aa
wget https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index/resolve/main/part_ab
wget https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train/resolve/main/train.parquet
wget https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train/resolve/main/test.parquet

cat part_* > e5_Flat.index
gzip -d wiki-18.jsonl.gz
cd ..