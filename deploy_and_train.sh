#!/bin/bash

sudo apt get update
git clone https://github.com/josh-freeman/HPshape.git

cd HPshape
mkdir examples

declare -a list=("Book%201%20-%20The%20Philosopher's%20Stone.txt" "Book%202%20-%20The%20Chamber%20of%20Secrets.txt" "Book%203%20-%20The%20Prisoner%20of%20Azkaban.txt" "Book%204%20-%20The%20Goblet%20of%20Fire.txt" "Book%205%20-%20The%20Order%20of%20the%20Phoenix.txt" "Book%206%20-%20The%20Half%20Blood%20Prince.txt" "Book%207%20-%20The%20Deathly%20Hallows.txt")

# About to iterate through list of titles
for title in ${list[@]}; do
   modified="${title//%20/ }"

   # unescaping spaces from url format in the title
   echo Downloading : examples/$modified

   curl --output "examples/$modified" --url https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/$title

done

mkdir ckpt

# at this point, everything is ready to launch training.


# run the model from HPshape/main/__main__.py. From here, each command is vital. Hence we activate -e

set -e
set -Eeuo pipefail

sudo apt install python3-pip
pip3 install -r requirements.txt
python3 -m spacy download en_core_web_sm
python3 -m spacy download en_core_web_trf

python3 -m src.__main__