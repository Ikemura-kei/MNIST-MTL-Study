mkdir -p raw
mkdir -p processed

cd raw

wget https://web.archive.org/web/20220331130319/https://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget https://web.archive.org/web/20220331130319/https://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget https://web.archive.org/web/20220331130319/https://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget https://web.archive.org/web/20220331130319/https://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz