from pathlib import Path
from antares.craft import read_study_local

dir_study=Path("C:/Users/brescianomat/Documents/Etudes Antares/BP23_A-Reference_2036")

study=read_study_local(study_path=dir_study)
