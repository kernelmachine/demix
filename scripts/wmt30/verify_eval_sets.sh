declare -A test_set
test_set=( ["de-en"]="wmt14" ["ro-en"]="wmt16" ["cs-en"]="wmt18" \
["fr-en"]="wmt14" ["ru-en"]="wmt19" ["zh-en"]="wmt19" ["es-en"]="wmt13" 
["fi-en"]="wmt19" ["et-en"]="wmt18" ["lv-en"]="wmt17" ["lt-en"]="wmt19" 
["hi-en"]="wmt14" ["kk-en"]="wmt19" ["tr-en"]="wmt18" ["gu-en"]="wmt19" )

declare -A valid_set
valid_set=( ["cs-en"]="wmt17" ["fr-en"]="wmt13" ["ru-en"]="wmt18" \
["zh-en"]="wmt18" ["es-en"]="wmt12" ["fi-en"]="wmt18" ["de-en"]="wmt13" \
["et-en"]="wmt18/dev" ["lv-en"]="wmt17/dev" ["lt-en"]="wmt19/dev" ["ro-en"]="wmt16/dev" \
["hi-en"]="wmt14" ["kk-en"]="wmt19/dev" ["tr-en"]="wmt17" ["gu-en"]="wmt19/dev" )

for lang in cs fr ru zh es 'fi' de  et lv lt ro hi kk tr gu ; do
for type in valid test ; do
    key_lang_pair="$lang-en"
    if [ "${type}" == "valid" ] ; then
        testset="${valid_set[${key_lang_pair}]}"
    else
        testset="${test_set[${key_lang_pair}]}"
    fi
    sacrebleu --echo ref -l $lang-en -t $testset > any_to_en
    sacrebleu --echo src -l en-$lang -t $testset > en_to_any
    num_diff=`diff -y --suppress-common-lines any_to_en en_to_any | wc -l`
    echo $lang $type $num_diff
done
done

# cs valid 0
# cs test 0
# fr valid 0
# fr test 3477
# ru valid 0
# ru test 2000
# zh valid 0
# zh test 2000
# es valid 0
# es test 0
# fi valid 0
# fi test 1997
# de valid 0
# de test 4037
# et valid 0
# et test 0
# lv valid 0
# lv test 0
# lt valid 0
# lt test 1000
# ro valid 0
# ro test 0
# hi valid 3637
# hi test 3637
# kk valid 0
# kk test 1000
# tr valid 0
# tr test 0
# gu valid 0
# gu test 1016