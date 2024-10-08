#!/bin/bash -

for file in $(<output.txt); do cp ./train_full/"${file%.*}" ./train_text; done
for file in $(<output.txt); do cp ./test_full/"${file%.*}" ./test_text; done
