files="https://nihcc.box.com/shared/static/sp5y2k799v4x1x77f7w1aqp26uyfq7qz.zip"

files=($files)
num_files=${#files[@]}

for (( i=0; i<${num_files}; i++ ));
do
    idx=$(printf "%02d" "$(expr $i + 1)")
    if [ -f Images_png_$idx.zip ]; then
        echo "Images_png_$idx.zip exist"
    else
        wget ${files[$i]} -O Images_png_$idx.zip
        unzip Images_png_$idx.zip
    fi
done
echo "DeepLesion dataset downloaded!"