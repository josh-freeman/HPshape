
@echo off
echo "Deploying HPshape directory next to this script"
git clone https://github.com/josh-freeman/HPshape.git



setlocal EnableDelayedExpansion
(set \n=^
%==%
)

pause

Rem declare an array variable

cd HPshape
mkdir examples


pause
echo About to iterate through list
for /f "delims=" %%i in ("Book%%201%%20-%%20The%%20Philosopher's%%20Stone.txt!\n!Book%%202%%20-%%20The%%20Chamber%%20of%%20Secrets.txt!\n!Book%%203%%20-%%20The%%20Prisoner%%20of%%20Azkaban.txt!\n!Book%%204%%20-%%20The%%20Goblet%%20of%%20Fire.txt!\n!Book%%205%%20-%%20The%%20Order%%20of%%20the%%20Phoenix.txt!\n!Book%%206%%20-%%20The%%20Half%%20Blood%%20Prince.txt!\n!Book%%207%%20-%%20The%%20Deathly%%20Hallows.txt") do (
   pause 
   set "string=%%i"
   set "modified=!string:%%20= !"
   echo Downloading : examples/!modified!
   echo "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/%%i"
   curl --output "examples/!modified!" --url https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/%%i
)
pause

mkdir ckpt

Rem at this point, everything is ready to launch training.

pause

Rem run the model fromHPshape/main/__main__.py


pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf

python -m src.__main__
