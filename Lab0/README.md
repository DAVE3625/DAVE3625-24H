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

Follow the instructions at [miniconda documentation] or [anaconda documentation] to download and install Conda.

Test if conda was installed correctly by typing 
```bash
conda
```

### 2. Create a New Conda Environment

Creating separate environments for different projects keeps everything organized and prevents dependency problems. It ensures that changes in one project won’t mess up another which will make your work easier to manage and debug.

Create a new conda environment by using the `conda-env.yml`:

```bash
conda create -f conda-env.yml
```

You can also create a new conda environment as follows:

```bash
conda create -n myenvname python=3.8
```


### 3. Activate the Environment

Activate your new environment with the command:

```bash
conda activate dave3625
```

### 4. Install Python Packages

You can install Python packages (pandas, numpy, matplotlib, scipy, jupyter notebook, ipykernel) using:

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

# Tips

#### 1. **Choosing Between Anaconda and Miniconda**
   - **Anaconda**: Includes a large set of pre-installed packages, useful if you want everything set up quickly. However, it takes up more space (~3GB).
   - **Miniconda**: A minimal version of Anaconda that includes only Conda and its dependencies, allowing you to install only the packages you need.

#### 2. **Verifying Conda Installation**
   - After installing, verify that Conda is installed correctly by typing `conda` in your terminal. If the command is not recognized, check your PATH settings.

#### 3. **Managing Environment YAML Files**
   - Use `conda env create -f conda-env.yml` to set up environments from YAML files. This ensures consistent package versions across different systems.

#### 4. **Adding Conda-Forge Channel**
   - Conda-Forge is a community-maintained channel with a vast collection of packages. Adding it with `conda config --add channels conda-forge` is recommended for access to more packages.

#### 5. **Channel Priority**
   - Setting `channel_priority` to `strict` ensures that packages are installed from the highest-priority channel first, which can help avoid conflicts.

#### 6. **Kernel Selection in VS Code**
   - After opening a Jupyter Notebook in VS Code, ensure you select the correct Conda environment (e.g., `dave3625`) from the kernel dropdown in the top right corner to avoid import errors.

# Useful Conda Commands

#### 1. **Basic Conda Commands**
   - **Check Conda Version:**
     ```bash
     conda --version
     ```
   - **Update Conda:**
     ```bash
     conda update conda
     ```

#### 2. **Managing Environments**
   - **Create a New Environment:**
     ```bash
     conda create -n myenv python=3.8
     ```
   - **Activate an Environment:**
     ```bash
     conda activate myenv
     ```
   - **Deactivate the Current Environment:**
     ```bash
     conda deactivate
     ```
   - **List All Environments:**
     ```bash
     conda env list
     ```
   - **Remove an Environment:**
     ```bash
     conda remove -n myenv --all
     ```

#### 3. **Installing and Managing Packages**
   - **Install a Package:**
     ```bash
     conda install package_name
     ```
   - **Install Multiple Packages:**
     ```bash
     conda install package1 package2
     ```
   - **Install a Specific Version of a Package:**
     ```bash
     conda install package_name=2.1
     ```
   - **Update a Package:**
     ```bash
     conda update package_name
     ```
   - **Remove a Package:**
     ```bash
     conda remove package_name
     ```
     
#### 4. **Exporting and Importing Environments**
   - **Export an Environment to a YAML File:**
     ```bash
     conda env export > environment.yml
     ```
   - **Create an Environment from a YAML File:**
     ```bash
     conda env create -f environment.yml
     ```

#### 5. **Miscellaneous Commands**
   - **List All Installed Packages in the Active Environment:**
     ```bash
     conda list
     ```
   - **Search for a Package:**
     ```bash
     conda search package_name
     ```
   - **Clean Up Unused Packages and Tarballs:**
     ```bash
     conda clean --all
     ```

## License
Distributed under the MIT License. See `LICENSE` for more information.

<!-- MARKDOWN LINKS & IMAGES --> 
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[issues-shield]: https://img.shields.io/github/issues/umaimehm/Intro_to_AI_2021.svg?style=for-the-badge
[issues-url]: https://github.com/DAVE3625/DAVE3625-24H/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/DAVE3625/DAVE3625-24H/blob/main/Lab1/LICENSE

[here]: https://code.visualstudio.com/
[miniconda documentation]: https://docs.conda.io/en/latest/miniconda.html
[anaconda documentation]: https://docs.anaconda.com/
