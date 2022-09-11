#!/bin/bash
pip install -r requirements.txt
pip install -e rolf/
pip install -e d4rl/
pip install -e spirl/

cd calvin
sh install.sh
cd ../..
