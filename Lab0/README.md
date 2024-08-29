<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]




<!-- PROJECT LOGO -->
<br />
<h3 align="center">Dave3625 - Lab 0</h3>
<p align="center">
  <a href="https://github.com/DAVE3625/DAVE3625-24H/tree/main/Lab0">
    <img src="img/logo.png" alt="Environment Setup" width="auto" height="auto">
  </a>

  <p align="center">
    An exercise in setting up your Python environment and performing basic data augmentation tasks. <br /> This lab will walk you through setting up a Python virtual environment, installing necessary packages, and setting up a Jupyter Notebook for further exercises.
    <br />
    ·
    <a href="https://github.com/DAVE3625/DAVE3625-24H/issues">Report Bug</a>
    ·
    <a href="https://github.com/DAVE3625/DAVE3625-24H/issues">Request Feature</a>
  </p>
</p>


<!-- ABOUT THE LAB -->
## About The Lab

This lab focuses on setting up the necessary environment for AI and data science tasks, followed by a simple data augmentation exercise. You'll start by setting up a conda environment, installing essential Python libraries, and setting up Visual Studio Code with Jupyter Notebook for development.

## Part 1: Environment Setup

### 1. Download and Install Conda

Follow the instructions at [miniconda documentation] to download and install Conda.

Test if conda was installed correctly by typing 
```bash
conda
```

### 2. Create a New Conda Environment from `conda-env-yml`

Create a new conda environment named `dave3625`:

```bash
conda create -f conda-env.yml
```

### 3. Activate the Environment
Activate your new environment with the command:

```bash
conda activate dave3625
```

### 4. Install Python Packages

Install the required Python packages (pandas, numpy, matplotlib, scipy, jupyter notebook, ipykernel) using:

```bash 
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install pandas numpy matplotlib scipy jupyter notebook ipykernel
```

### 5. Download Visual Studio Code
Download Visual Studio Code from [here].

### 6. Install the Jupyter Extension
Install the Jupyter extension for Visual Studio Code from the "Extensions" menu on the left.

### 7. Install the Python Extension
Install the official Python extension for Visual Studio Code.

### 8. Create a New Jupyter Notebook
Create a new Jupyter Notebook by creating a file with a .ipynb extension (e.g., new_notebook.ipynb).

### 9. Open the Notebook in VSCode
Open the notebook file in Visual Studio Code and select the dave3625 kernel in the top right.

## Part 2: Data Augmentation
(Details for data augmentation tasks will follow in the upcoming labs, focusing on using pandas and other Python tools for data manipulation.)

## License
Distributed under the MIT License. See `LICENSE` for more information.

<!-- MARKDOWN LINKS & IMAGES --> 
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[here]: https://code.visualstudio.com/
[miniconda documentation]: https://docs.conda.io/en/latest/miniconda.html
