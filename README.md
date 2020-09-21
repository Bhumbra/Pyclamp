#Pyclamp

##INSTALLATION:

Install package python3-PyQt4 for your GNU/Linux distro then run:

```sh
pip3 install --user .
```

It is recommended for installers to create a Python virtual environment for 
trying the code, for instance (you may need first to use your Linux package 
manager to install python3-venv):

```sh
$ python3 -m venv virtual_env
$ source virtual_env/bin/activate
```

If your pip version is out of date, it needs to be upgraded (version>=20.0):
```sh
(virtual_env) $ pip3 --version
(virtual_env) $ pip3 install -U pip
(virtual_env) $ pip3 --version
pip3 install --user .
```

## RUNNING:

```sh
./run_pyclamp
```
