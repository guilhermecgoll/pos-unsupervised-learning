# pos-unsupervised-learning
Contains a eigenfaces sample project in Python 3.

# Running project

First of all it's necessary to create a virtual environment (recommended) and install all dependencies.

## Create virtual environment

- In ``pos-unsupervised-learning`` folder run the following command:

```python -m venv .venv```

## Activate virtual environment

- Still in ``pos-unsupervised-learning`` folder run:

```./venv/Scripts/Activate.ps1```

PS: On Microsoft Windows, it may be required to enable the Activate.ps1 script by setting the execution policy for the user. You can do this by issuing the following PowerShell command:

```PS C:> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser```

See more: https://docs.python.org/3/library/venv.html

## Install dependencies

- Still in ``pos-unsupervised-learning`` folder run:

```pip install --no-cache-dir -r pca-eigenfaces\requirements.txt```

## Run project

- Run:

```python pca-eigenfaces/main.py```

# References

https://docs.opencv.org/4.4.0/d3/df2/tutorial_py_basic_ops.html