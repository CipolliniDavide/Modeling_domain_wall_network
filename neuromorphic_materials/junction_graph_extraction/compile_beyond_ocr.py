#! /usr/bin/env python3
import subprocess
from pathlib import Path

# Start subprocess
junclets_dir = Path("./beyondOCR_junclets/")
run_file_name = junclets_dir / "my_junc"
lib_list = [
    "beyondOCR_junclets.cpp",
    "dflJuncletslib.cpp",
    "dflPenWidth.cpp",
    "dflUtils.cpp",
    "pamImage.cpp",
    "dflBinarylib.cpp",
]
load_lib = [str((junclets_dir / i).resolve()) for i in lib_list]
# Create runnable
create_runnable = ["g++"] + load_lib + ["-pedantic", "-Wall", "-o", run_file_name]
test = subprocess.Popen(create_runnable, stdout=subprocess.PIPE)
output = test.communicate()[0]
