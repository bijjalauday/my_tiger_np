===================
Environment Setup
===================

Instructions to setup the development environment for the project:

.. note::
    For Docker deployment guide, please refer :ref:`Deployment Guide`



1. Pre-requisites
====================

1.1 MiniConda
------------------

Ensure you have ``Miniconda`` installed and can be run from your shell.
If not, download the installer for your platform from `here <https://docs.conda.io/en/latest/miniconda.html>`_


.. note::
    - If you already have ``Anaconda`` installed, please still go ahead and install ``Miniconda`` and use it for development.
    - If ``conda`` cmd is not in your path, you can configure your shell by running ``conda init``

1.2 git
----------------

Ensure you have ``git`` installed and can be run from your shell


.. note::
    - If you have installed ``Git Bash`` or ``Git Desktop`` then the ``git`` cli is not accessible by default from cmdline. If so, you can add the path to ``git.exe`` to your system path.
    - Here are the paths on a recent setup::

        %LOCALAPPDATA%\Programs\Git\git-bash.exe
        %LOCALAPPDATA%\GitHubDesktop\app-<ver>\resources\app\git\mingw64\bin\git.exe

1.3 invoke and PyYAML
--------------------------

Ensure `invoke <http://www.pyinvoke.org/index.html>`_ tool and PyYAML are installed in your ``base`` ``conda`` environment.
If not, run::

    (base):~$ pip install invoke
    (base):~$ pip install PyYAML


2. Setup Process
========================

- Switch to the project root folder

- A collection of workflow automation tasks can be seen as follows

.. note::
    - Please make sure there are no spaces in the folder path. Environment setup fails if spaces are present::

        (base):~/<proj-folder>$ inv -l

- To verify pre-requisites, run and check no error messages (Error: ...) are printed::

        (base)~/<proj-folder>$ inv debug.check-reqs

2.1 Environment setup
------------------------

- Environment is divided into two sections

    1. Core - These are must have packages & will be setup by default. These are declared here:

        - ``deploy/conda_envs/<windows/linux>-cpu-64-dev.yml``

    2. Addons - These are for specific purposes you can choose to install. Here are the addon options

        - ``formatting`` - To enforce coding standards in your projects.

        - ``documentation`` - To auto-generate doc from doc strings and/or create rst style documentation to share documentation online

        - ``jupyter`` - To run the notebooks. This includes jupyter extensions for spell check, advances formatting.

    - Edit the addons here ``deploy/conda_envs/<addon-name>-<windows/linux>-cpu-64-dev.yml`` to suit your need.

    - Each of the packages there have line comments with their purpose. From an installation standpoint extras are treated as addons

- The name of the env would be ``npp-pet-id-dev``

2.2 Setup a development environment
---------------------------------------

- Run below to install core libraries::

    (base):~/<proj-folder>$ inv dev.setup-env

- Run below to install addons.::

    (base):~/<proj-folder>$ inv dev.setup-addon --formatting --jupyter --documentation

You now should have a standalone conda python environment and install the code in the current repository along with all required dependencies.

- Get the installation info by running::

    (base):~/<proj-folder>$ inv dev.info

