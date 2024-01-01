# Nestle Purina: Pet Visual Image Recognition

Identify the cat and dog in a household based on the images.


# Pre-requisites

* Ensure you have `Miniconda` installed and can be run from your shell. If not, download the installer for your platform here: https://docs.conda.io/en/latest/miniconda.html

     **NOTE**
     
     * If you already have `Anaconda` installed, pls. still go ahead and install `Miniconda` and use it for development. 
     * If `conda` cmd is not in your path, you can configure your shell by running `conda init`. 


* Ensure you have `git` installed and can be run from your shell

     **NOTE**
     
     * If you have installed `Git Bash` or `Git Desktop` then the `git` cli is not accessible by default from cmdline. 
       If so, you can add the path to `git.exe` to your system path. Here are the paths on a recent setup
      
```
        %LOCALAPPDATA%\Programs\Git\git-bash.exe
        %LOCALAPPDATA%\GitHubDesktop\app-<ver>\resources\app\git\mingw64\bin\git.exe
```

* Ensure [invoke](http://www.pyinvoke.org/index.html) tool and PyYAML are installed in your `base` `conda` environment. If not, run

```
(base):~$ pip install invoke
(base):~$ pip install PyYAML
```

# Getting started

* Switch to the root folder (i.e. folder containing this file)
* A collection of workflow automation tasks can be seen as follows
    **NOTE**
     
     * Please make sure there are no spaces in the folder path. Environment setup fails if spaces are present.

```
(base):~/<proj-folder>$ inv -l
```

* To verify pre-requisites, run

```
(base)~/<proj-folder>$ inv debug.check-reqs
```

and check no error messages (`Error: ...`) are printed.


## Environment setup:

### Introduction
* Environment is divided into two sections

    * Core - These are must have packages & will be setup by default. These are declared here `deploy/conda_envs/<windows/linux>-cpu-64-dev.yml`
    * Addons - These are for specific purposes you can choose to install. Here are the addon options
        * `formatting` - To enforce coding standards in your projects.
        * `documentation` - To auto-generate doc from doc strings and/or create rst style documentation to share documentation online
        * `jupyter` - To run the notebooks. This includes jupyter extensions for spell check, advances formatting.
    * Edit the addons here `deploy/conda_envs/<addon-name>-<windows/linux>-cpu-64-dev.yml` to suit your need.
    * Each of the packages there have line comments with their purpose. From an installation standpoint extras are treated as addons
* You can edit them to your need. All these packages including addons & extras are curated with versions & tested throughly for acceleration.
* While you can choose, please decide upfront for your project and everyone use the same options.
* Name of the env would be `npp-pet-id-dev`.

### Setup a development environment:

* Run below to install core libraries
```
(base):~/<proj-folder>$ inv dev.setup-env
```

* Run below to install addons. Feel free to customize addons as suggested in the introduction.
```
(base):~/<proj-folder>$ inv dev.setup-addon --formatting --jupyter --documentation
```


You now should have a standalone conda python environment and install the code in the current repository along with all required dependencies.

* Get the installation info by running
```
(base):~/<proj-folder>$ inv dev.info
```

* Test your installation by running
```
(base):~/<proj-folder>$ inv test.val-env
```

# Launching Jupyter Notebooks

- In order to launch a jupyter notebook locally in the web server, run

    ```
    (base):~/<proj-folder>$ inv launch.jupyterlab
    ```
     After running the command, type [localhost:8080](localhost:8080) to see the launched JupyterLab.
     
- The `inv` command has a built-in help facility available for each of the invoke builtins. To use it, type `--help` followed by the command:
    ```
    (base):~/<proj-folder>$ inv launch.jupyterlab --help
    ```
- On running the ``help`` command, you get to see the different options supported by it.

    ```
    Usage: inv[oke] [--core-opts] launch.jupyterlab [--options] [other tasks here ...]

    Options:
    -a STRING, --password=STRING
    -e STRING, --env=STRING
    -i STRING, --ip=STRING
    -o INT, --port=INT
    -p STRING, --platform=STRING
    -t STRING, --token=STRING
    ```

## Troubleshooting

### Error: "AttributeError: 'Upsample' object has no attribute 'recompute_scale_factor'"

If you encounter the following error message:

```
AttributeError: 'Upsample' object has no attribute 'recompute_scale_factor'
```

This can occur when using an older version of PyTorch with a model that was trained using a newer version of PyTorch that includes the `recompute_scale_factor` attribute in the `Upsample` module.

To resolve this issue, you can update the `upsampling.py` file in the `<path-to-env>\Lib\site-packages\torch\nn\modules` folder as follows:

```python
def forward(self, input: Tensor) -> Tensor:
    return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners,
                         #recompute_scale_factor=self.recompute_scale_factor
                         )
```

Comment out the `recompute_scale_factor` argument by placing a `#` at the beginning of the line, as shown above. This will disable the `recompute_scale_factor` argument and allow the `Upsample` module to work correctly with the older version of PyTorch.

Note that you may need to restart your Python interpreter or Jupyter notebook for the changes to take effect.