mkdir ~/wmt_datasets/any_to_en_bin
mkdir ~/wmt_datasets/en_to_any_bin
for lang in cs de es  et  'fi'  fr  gu  hi  kk  lt  lv  ro  ru  tr  zh ; do
    ln -s ~/wmt_datasets/binarized/$lang-en/* ~/wmt_datasets/any_to_en_bin/
done

for lang in cs de es  et  'fi'  fr  gu  hi  kk  lt  lv  ro  ru  tr  zh ; do
    ln -s ~/wmt_datasets/binarized/en-$lang/* ~/wmt_datasets/en_to_any_bin/
done