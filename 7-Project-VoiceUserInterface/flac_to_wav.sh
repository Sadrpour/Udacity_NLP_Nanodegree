#!/bin/bash


folder=$1


for file in $(find "$folder" -type f -iname "*.flac")
do
    name=$(basename "$file" .flac)
    dir=$(dirname "$file")
    echo ffmpeg -i "$file" "$dir"/"$name".wav
    ffmpeg -i $file $dir/$name.wav
    #rm $file
done
