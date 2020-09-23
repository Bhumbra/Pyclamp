# Pyclamp

## Installation:

This code is intended for a GNU/Linux environment, with a /tmp/ directory.
It has not been tested outside GNU/Linux. A Python version >= 3.6 is required. 
Since 13th September 2020, Python 3.5 reached its end of life and won't work.
A network connection is required for downloading dependencies and the data.

It is recommended for installers to create a Python virtual environment for 
testing the code, for instance (you may need first to use your Linux package 
manager to  install python3-venv):

```sh
$ python3 -m venv ~/virtual_env
$ source ~/virtual_env/bin/activate
```

If your pip version is out of date, it needs to be upgraded (version>=20.0):
```sh
(virtual_env) $ pip3 --version
(virtual_env) $ pip3 install -U pip
(virtual_env) $ pip3 --version
```

Pyclamp is installed using a pip installer from the working directory path of 
this README.md file:

```sh
(virtual_env) $ file README.md
README.md: ASCII text
(virtual_env) $ pip3 install -e .
```

If this fails, check your Python version is up to date.
 
## Running:

```sh
(virtual_env) $ ./run_pyclamp
```

Help options are available, and can be used to display all plots:

```sh
(virtual_env) $ python3 main/model_data.py -h
(virtual_env) $ python3 main/model_data.py --show-hist=True --show-scat=True
```

## Removing:

After finishing with the package, the data and dependencies can be removed:

```sh
(virtual_env) $ deactivate
$ rm -r ~/virtual_env/
```
