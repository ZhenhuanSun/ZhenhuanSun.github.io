---
author: ["Zhenhuan Sun"]
title: "PyCharm Meets Compute Canada"
summary: "This guide walks through how to use PyCharm IDE with Compute Canada."
date: 2026-08-21
ShowToc: true
---

This guide walks through how to use PyCharm IDE with [Compute Canada](https://docs.alliancecan.ca/wiki/Getting_started) 
for my future reference. I will continue updating it as I explore more features of PyCharm and Compute Canada, or
discover more convenient ways to accomplish any of the following tasks.

## Prerequisites

- Follow [this tutorial](https://docs.alliancecan.ca/wiki/Apply_for_a_CCDB_account) to create a Compute Canada account, 
then follow [this tutorial](https://docs.alliancecan.ca/wiki/Multifactor_authentication) to enable multifactor authentication.

- Review the [course](https://training.sharcnet.ca/courses/course/view.php?id=22) provided by Compute Canada to familiarize 
the basics. A valid Compute Canada account is required to access this course.

- On the [CCDB portal](https://ccdb.alliancecan.ca), navigate to `Resources -> Access Systems` and require access to the 
HPC servers you want to use. A valid Compute Canada account is required to access the portal.

- Navigate to `.ssh/` directory in your `\home` directory by running `cd ~/.ssh/` in terminal.
  - If you see two files with names such as `id_KeyType` and `id_KeyType.pub`, where `KeyType` might be `rsa`, `ed25519`,
  or `dsa`, then you already have a SSH key pair.
  - If not, follow [this tutorial](https://docs.alliancecan.ca/wiki/Using_SSH_keys_in_Linux) to create a SSH key pair.

- View the public key (the file ending in `.pub`) by running `cat id_KeyType.pub`, then copy its contents. On the CCDB
portal, navigate to `My Account -> SSH Keys` and paste the public key there.

After completing these steps, you should be able to access Compute Canada HPC servers from a terminal by running

```bash
ssh username@clustername.alliancecan.ca
```

For example, to connect to the Nibi cluster

```bash
ssh username@nibi.alliancecan.ca
```

With the SSH key configured, a password is not required for authentication; however, Duo two-factor authentication is 
still required to complete the login.

## Virtual Environment

To create a project-specific virtual environment in the remote server, follow these steps:

1.  Check which Python versions are available on the remote server by running one of the following:

    ```bash
    module spider python
    module avail python
    ```
    
    `module avail python` shows Python modules that are currently available to load directly in your present environment,
    whereas, `module spider python` searches more broadly and shows all Python modules known to the module system, including 
    versions that may require you to load another module first.
2.  Decide which Python version your project requires, then load the module corresponding to the Python version required by 
    your project. For example, if your project requires Python 3.10.13, run:

    ```bash
    module load python/3.10.13
    ```
    
    To verify that the correct Python version has been loaded, check the Python version by running `python --version`.

3.  (Optional and Not Recommended for PyCharm) If your project requires commonly used Python packages such as `NumPy`, `SciPy`, 
    and `Matplotlib`, it may be convenient to load the `scipy-stack` module together with the Python module. However, in 
    this case, you need to make sure the `scipy-stack` module you load is compatible with the Python version you intend to 
    use. For example, if you project requires Python 3.10, and by running `module spider scipy-stack`, you know there exists 
    5 `scipy-stack` versions

    ```text
    Versions:
        scipy-stack/2023b
        scipy-stack/2024a
        scipy-stack/2024b
        scipy-stack/2025a
        scipy-stack/2026a
    ```

    In this case, you need to check the compatibility of each `scipy-stack` version with Python 3.10 by running

    ```bash
    module spider scipy-stack/version_number
    ```

    and load the `scipy-stack` module that is compatible with Python 3.10. At the time of writing this post, all five
    `scipy-stack` versions listed above support Python 3.10 except `scipy-stack/2026a`. Thus, for this particular example,
    one compatible choice is to load `python/3.10.13` together with `scipy-stack/2025a`:

    ```bash
    module load python/3.10.13 scipy-stack/2025a
    ```

    Although loading the `scipy-stack` module may seem convenient, it doesn't work well with PyCharm. For reasons I do not 
    fully understand, if you load the `scipy-stack` module and follow the steps in next section to access remote Python interpreter, 
    the packages provided by `scipy-stack` are visible only in the remote shell but not automatically visible to the remote 
    Python interpreter configured in PyCharm.
    
4.  Navigate to your project directory on the remote server, creating one first if it does not already exist, then create 
    a virtual environment named `.venv` there by running

    ```bash
    virtualenv --no-download .venv
    ```
    
5.  Activate the virtual environment by running

    ```bash
    source .venv/bin/activate
    ```
   
    If you run `which python` now, the active Python interpreter should be the one located in the virtual environment. For 
    example, the output should look like

    ```bash
    /your_project_directory/.venv/bin/python
    ```
   
    If the `scipy-stack` module is loaded, then the output of `pip list`, which lists all installed packages in the current
    virtual environment, should include packages such as `NumPy`, `SciPy`, and `Matplotlib`, etc. Otherwise, these packages
    need to be installed separately.

6.  Install packages into the virtual environment by running

    ```bash
    pip install package_name --no-index
    ```
    
    The `--no-index` option tells `pip` to not install from PyPI, but instead to install only from locally available packages.
    I observe, in some cases, running this is noticeably faster than running `pip install package_name`.

After creating the virtual environment, run `deactivate` to exit it. To reuse it later, navigate to the project directory 
on the remote cluster and reactivate the environment. Although the official Compute Canada tutorial recommends loading
the same modules that were used to create the virtual environment before reactivating it, I did not find this step necessary
when using the remote interpreter in PyCharm. For example, if modules `python/3.10.13` and `scipy-stack/2025a` were loaded 
when created the virtual environment, the tutorial recommends running

```bash
module load python/3.10.13 scipy-stack/2025a
```

before reactivating the environment with

```bash
source your_project_directory/.venv/bin/activate
```

However, whether these two modules are loaded or not, the packages provided by `scipy-stack` module are not visible to 
the remote interpreter configured in PyCharm, and the Python interpreter in the virtual environment will have version 3.10.13. 
See [this tutorial](https://docs.alliancecan.ca/wiki/Python) for more details on creating and using a virtual environment on Compute Canada clusters.

<!--
load the same modules used to create it then reactivate it from the project directory. For example, if modules `python/3.10.13` 
and `scipy-stack/2025a` were loaded when created the virtual environment, run

```bash
module load python/3.10.13 scipy-stack/2025a
```

before running

```bash
source your_project_directory/.venv/bin/activate
```
-->

## Access Remote Interpreter

To use a virtual environment located on a remote server from a local PyCharm IDE, you need to configure a remote Python 
interpreter via SSH. To do that in PyCharm

1. Navigate to `Settings → Python → Interpreter`.
2. Select `Add Interpreter → On SSH...`.
3. Select `New SSH connection`, then enter your Compute Canada username and the hostname of the HPC server you want to access. 
For example, if your username is `USERNAME` and you want to access the Nibi cluster, enter `USERNAME` as the username and 
`nibi.alliancecan.ca` as the host. Changing the port number is not necessary.

After step 3, you will be asked to complete Duo two-factor authentication several times and will mostly likely get
stuck at the **Introspecting SSH server** step, where you are asked to complete another Duo two-factor authentication
but are unable to enter the passcode. This seems to be a longstanding and common issue in PyCharm when accessing a remote Python 
interpreter on a cluster that requires Duo two-factor authentication, as discussed in [this Stack Overflow thread](https://stackoverflow.com/questions/75404321/pycharm-stuck-on-introspecting-ssh-server-because-two-factor-login-required-for).

The solution, as suggested in the Stack Overflow thread, is to enable multiplexing and establish a SSH connection to 
the cluster before accessing the remote Python interpreter in PyCharm. To enable multiplexing

1. Create a `config` file in the `.ssh/` directory by running

```bash
touch ~/.ssh/config
chmod 600 ~/.ssh/config
```

See [this tutorial](https://linuxize.com/post/using-the-ssh-config-file/) for more details on how to configure and use an 
SSH `config` file. 

2. Add in `config` file

```bash
# Reuse an existing SSH connection when connecting to Alliance/Compute Canada hosts
Host *.alliancecan.ca
    ControlMaster auto
    ControlPath ~/.ssh/control-%r@%h:%p
```

See [this tutorial](https://www.cyberciti.biz/faq/linux-unix-reuse-openssh-connection/) for more details on what these 
SSH multiplexing options do.

Once multiplexing is enabled, establish an SSH connection to the cluster in terminal, then

1.  Copy the path to the Python interpreter in the virtual environment. The path should look something like

    ```bash
    /your_project_directory/.venv/bin/python
    ```

2.  Repeat the first three steps in this section. This time, you should be able to proceed past the **Introspecting SSH 
    server** step.
 
3.  Check `Select existing` option for the environment, select `Python` as the interpreter type, and paste the copied path 
    into the `Python path` field. Then expand `Target-Specific Properties` and, under `Sync folders`, set the remote path 
    to your project directory on the remote cluster. Also check the `Automatically upload project files to the server` option.

After completing these steps, files in your local project directory will be automatically uploaded to the corresponding 
directory on the remote cluster. To customize which files are uploaded and when uploads occur, go to `Settings -> Build, 
Execution, Deployment -> Deployment -> Options`. To download files from the remote cluster, go to `Tools -> Deployment`.

## Jupyter Notebook

Jupyter Notebook can be run on either a login node or a compute node. Read [this section](https://training.sharcnet.ca/courses/mod/lesson/view.php?id=452&pageid=330) 
of the course provided by Compute Canada to review when each type of node should be used. Make sure `Jupyter` is installed
in your virtual environment.

### Login Node

To access a Jupyter Notebook running on a login node from PyCharm, follow these steps

1.  SSH to the remote cluster from your local machine and activate the virtual environment in your project directory.
2.  Start a Jupyter server on the remote cluster without opening a browser by running

    ```bash
    jupyter notebook --no-browser --port=8888
    ```
    
    Then copy the token that appears after ` http://localhost:8888/tree?token=`. You can use a different port number if 
    you prefer.
3.  In a terminal on your local machine, set up local port forwarding by running

    ```bash
    ssh -L 8888:localhost:8888 USERNAME@nibi.alliancecan.ca
    ```
    
    This will start a remote shell, and take connections to port `8888` on your local machine and forward them 
    through SSH to port `8888` on the remote machine. If you only want the SSH tunnel and do not want a remote shell, use

    ```bash
    ssh -N -L 8888:localhost:8888 USERNAME@nibi.alliancecan.ca
    ```
    
    where `-N` tells SSH not to start a remote shell and to perform only port forwarding.

4.  Navigate to `Settings -> Jupyter -> Jupyter Servers`. In the `Servers` column, select `External Server Notebook/Lab`.
    Set the `Server URL` to `http://localhost:8888`, replacing `8888` with the local port used for SSH port forwarding if
    necessary. Select `Notebook/Lab` as the server type, and paste the token into the `Token/Password` field. These steps 
    basically tell PyCharm to connect to a Jupyter server through port `8888` on the local machine and authenticate with
    the token given. Because local port forwarding has already been configured, connections to this port are forwarded 
    through SSH to port `8888` on the remote cluster, where the Jupyter server is running.

To terminate the running Jupyter Notebook server, press `Ctrl+C` in the remote shell where the server is running, then 
confirm the shutdown when prompted.