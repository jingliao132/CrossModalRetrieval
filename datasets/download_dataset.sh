FILE=$1

if [[ $FILE != "CUB_200_2011" &&  $FILE != "flowers" ]]; then
  echo "Available datasets are CUB_200_2011, flowers"
  exit 1
fi

echo "Specified [$FILE]"

URL=http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/$FILE.tgz

TAR_FILE=./datasets/$FILE.tgz
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $TAR_FILE
mkdir -p $TARGET_DIR
tar -zxvf $TAR_FILE -C ./datasets/
rm $TAR_FILE