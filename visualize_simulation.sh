#!/bin/bash

fileroot=test_sim
filetype=hdf5

configuration_file=$fileroot.$filetype
steps=10000
skip=100

# Create tikz files and compile
echo -e "Creating individual images\n"
for ((step=0; $step<$steps; step += $skip))
do
    echo "Processing step $step"
    output_file=$fileroot-$step.tex
    python create_tikz_diagram.py $configuration_file $output_file $step
    pdflatex $output_file > /dev/null
done

# Merge pdfs
echo -e "Merging images\n"
mv $fileroot-0.pdf $fileroot.pdf
for ((step=$skip; $step<$steps; step += $skip))
do
    pdfunite $fileroot.pdf $fileroot-$step.pdf $fileroot-tmp.pdf
    mv $fileroot-tmp.pdf $fileroot.pdf
done
