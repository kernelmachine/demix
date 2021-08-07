data_path=$1
domains=$(find $data_path -maxdepth 1 -type d -exec basename {} \;)

for domain in $domains; do \
    cp ${data_path}/${domain}/valid.bin ${data_path}/${domain}/valid_${domain}.bin
    cp ${data_path}/${domain}/valid.idx ${data_path}/${domain}/valid_${domain}.idx
done
