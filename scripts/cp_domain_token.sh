SRC_PATH=$1
TGT_PATH=$2
SRC_DOMAIN=$3
TGT_DOMAIN=$4

cp -r $SRC_PATH $TGT_PATH
mv $TGT_PATH/valid_${SRC_DOMAIN}.bin $TGT_PATH/valid_${TGT_DOMAIN}.bin
mv $TGT_PATH/valid_${SRC_DOMAIN}.idx $TGT_PATH/valid_${TGT_DOMAIN}.idx