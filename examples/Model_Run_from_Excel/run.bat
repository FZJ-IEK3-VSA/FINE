set /p file=Enter the name of the Excel file name which containts the model input data or hit [Enter] when the data is stored in scenarioInput.xlsx: 

if not defined file set file=scenarioInput.xlsx

echo import FINE as fn > run.py
echo fn.energySystemModelRunFromExcel('%file%') >> run.py
python run.py

del run.py

PAUSE
