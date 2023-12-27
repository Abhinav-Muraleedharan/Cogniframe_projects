#!/bin/bash

# Create the project directory
mkdir quantum_unet
cd quantum_unet

# Create data directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/external

# Create notebooks directory
mkdir notebooks
# Create paper directort 
mkdir paper
# Create source code directories and files
mkdir -p src/data
mkdir -p src/models
mkdir -p src/train
mkdir -p src/utils
mkdir -p src/config
touch src/__init__.py  # Create __init__.py files to treat directories as Python packages
touch src/data/__init__.py
touch src/models/__init__.py
touch src/train/__init__.py
touch src/utils/__init__.py
touch src/config/__init__.py

# Create experiments directory
mkdir experiments

# Create tests directory
mkdir tests

# Create documentation directory and files
mkdir docs
touch README.md
touch docs/README.md

# Create requirements.txt and .gitignore
touch requirements.txt
touch .gitignore

# Create LICENSE file (choose appropriate license type and add license text)
touch LICENSE

