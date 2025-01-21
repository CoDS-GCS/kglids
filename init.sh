#!/bin/bash

# create kglids conda env
echo "********** [1/5] Creating kglids conda environment **********"
conda create -n kglids python=3.8 -y
conda activate kglids

# install requirements
echo "********** [2/5] Installing pip requirements **********"
pip install -r requirements.txt

# install required embeddings
echo "********** [3/5] Downloading word embedding models **********"
mkdir -p storage/embeddings
# download fasttext embeddings
wget -nv -N -P storage/embeddings/ https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
gunzip storage/embeddings/cc.en.300.bin.gz
python -c "import fasttext; import fasttext.util; ft = fasttext.load_model('storage/embeddings/cc.en.300.bin'); fasttext.util.reduce_model(ft, 50); ft.save_model('storage/embeddings/cc.en.50.bin')"
# download glove embeddings
wget -nv -N -P storage/embeddings/ http://nlp.uoregon.edu/download/embeddings/glove.6B.100d.txt


# install graph db and configure default user and .graphdb-import
echo "********** [4/5] Downloading GraphDB graph store **********"
sudo docker pull ontotext/graphdb:10.7.6
sudo docker run -d --restart always -p 7200:7200 --name graphdb -v ~/graphdb-import/:/root/graphdb-import ontotext/graphdb:10.7.6

# install postgres and pgvector and configure
echo "********** [5/5] Installing Postgresql and pgvector (default password is postgres) **********"
sudo apt-get install -y postgresql postgresql-contrib
sudo -u postgres psql << EOF
ALTER USER postgres WITH PASSWORD 'postgres';

EOF
sudo systemctl restart postgresql
sudo apt install -y postgresql-16-pgvector
sudo -u postgres psql << EOF
CREATE EXTENSION vector;
EOF

