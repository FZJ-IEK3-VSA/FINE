@echo off
set /p file=Enter the name of the Excel file name which containts the model input data or hit [Enter] when the data is stored in scenarioInput.xlsx: 

echo[

if not defined file set file=scenarioInput.xlsx 

set /p env=Enter the name of your conda enviroment or hit [Enter] when your regular Python version should be used:

echo[

call activate %env%

echo import FINE as fn > run.py
echo fn.energySystemModelRunFromExcel('%file%') >> run.py

python run.py 

del run.py

PAUSE
