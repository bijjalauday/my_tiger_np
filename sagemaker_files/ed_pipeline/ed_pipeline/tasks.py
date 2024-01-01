import os
import os.path as op
import platform
from contextlib import contextmanager
import shutil
import tempfile
import time
import yaml
from invoke import Collection, UnexpectedExit, task

# Some default values
PACKAGE_NAME = "ta_pet_id"
ENV_PREFIX = "npp-pet-id"
ENV_PREFIX_PYSPARK = "npp-pet-id-pyspark"
NUM_RETRIES = 10
SLEEP_TIME = 1

OS = platform.system().lower()
ARCH = platform.architecture()[0][:2]
PLATFORM = f"{OS}-cpu-{ARCH}"
DEV_ENV = "dev"  # one of ['dev', 'run', 'test']
SLEEP_TIME = 1

HERE = op.dirname(op.abspath(__file__))
SOURCE_FOLDER = op.join(HERE, "src", PACKAGE_NAME)
TESTS_FOLDER = op.join(HERE, 'tests')
CONDA_ENV_FOLDER = op.join(HERE, "deploy", "conda_envs")
PYSPARK_ENV_FOLDER = op.join(HERE, "deploy", "pyspark")
NOTEBOOK_FOLDER = op.join(HERE, "notebooks", "tests")

_TASK_COLLECTIONS = []


# ---------
# Utilities
# ---------
def _get_env_name(platform, env):
    # FIXME: do we need platform ?
    return f"{ENV_PREFIX}-{env}"


def _get_env_name_pyspark(platform, env):
    # FIXME: do we need platform ?
    return f"{ENV_PREFIX_PYSPARK}-{env}"


def _change_permissions_recursive(path, mode):
    for root, dirs, files in os.walk(path, topdown=False):
        for _dir in [os.path.join(root, d) for d in dirs]:
            os.chmod(_dir, mode)
        for _file in [os.path.join(root, f) for f in files]:
            os.chmod(_file, mode)


def _clean_rmtree(path):
    for _try in range(NUM_RETRIES):
        try:
            _change_permissions_recursive(path, 0o777)
            shutil.rmtree(path)
        except Exception as e:
            time.sleep(SLEEP_TIME)
            print(f"{path} Remove failed with error {e}:: Retrying ..")
            continue
        print(f"{path} Remove Success")
        break


@contextmanager
def py_env(c, env_name):
    """Activate a python env while the context is active."""

    # FIXME: This works but takes a few seconds. Perhaps find a few to robustly
    # find the python path and just set the PATH variable ?
    if OS == "windows":
        # we assume conda binary is in path
        cmd = f"conda activate {env_name}"
    else:
        cmd = f'eval "$(conda shell.bash hook)" && conda activate {env_name}'
    with c.prefix(cmd):
        yield


def _create_task_collection(name, *tasks):
    """Construct a Collection object."""
    coll = Collection(name)
    for task_ in tasks:
        coll.add_task(task_)
    _TASK_COLLECTIONS.append(coll)
    return coll


def _create_root_task_collection():
    return Collection(*_TASK_COLLECTIONS)


# ---------
# debug tasks
# ---------


@task(name="check-reqs")
def check_setup_prerequisites(c):
    """Check setup prerequisites required i.e. git and conda availability.

    """
    _failed = []

    # check the folder has no spaces in the path
    if " " in HERE:
        raise RuntimeError("The path to the current folder has whitespaces in it.")

    for binary in ["git", "conda"]:
        try:
            out = c.run(f"{binary} --version", hide="out")
        except UnexpectedExit:
            print(
                f"ERROR: Failed to find `{binary}` in path. "
                "See `pre-requisites` section in `README.md` for some pointers."
            )
            _failed.append(binary)
        else:
            print(f"SUCCESS: Found `{binary}` in path : {out.stdout}")

    # FIXME: Figure out colored output (on windows) and make the output easier
    # to read/interpret.
    # for now, make a splash when failing and make it obvious.
    if _failed:
        raise RuntimeError(f"Failed to find the following binaries in path : {_failed}")


_create_task_collection("debug", check_setup_prerequisites)


# Dev tasks
# ---------
@task(
    help={
        "platform": (
                "Specifies the platform spec. Must be of the form "
                "``{windows|linux}-{cpu|gpu}-{64|32}``"
        ),
        "env": "Specifies the enviroment type. Must be one of ``{dev|test|run}``",
        "force": "If ``True``, any pre-existing environment with the same name will be overwritten",
    }
)
def setup_env(c, platform=PLATFORM, env=DEV_ENV, force=False):
    """Setup a new development environment.

    Creates a new conda environment with the dependencies specified in the file
    ``env/{platform}-{env}.yml``. To overwrite an existing environment with the
    same name, set the flag ``force`` to ``True``.
    """

    # run pre-checks
    check_setup_prerequisites(c)

    force_flag = "" if not force else "--force"
    if platform == 'windows':
        env_file = op.abspath(op.join(CONDA_ENV_FOLDER, f"{platform}-{env}.lock"))
    else:
        env_file = op.abspath(op.join(CONDA_ENV_FOLDER, f"{platform}-{env}.yml"))

    if not op.isfile(env_file):
        raise ValueError(f"The conda env file is not found : {env_file}")

    env_name = _get_env_name(platform, env)

    out = c.run(f"conda env create --name {env_name} --file {env_file}  {force_flag}")

    # check for jupyterlab
    with open(env_file, "r") as fp:
        env_cfg = fp.read()

    # installating jupyter lab extensions
    extensions_file = op.abspath(
        op.join(CONDA_ENV_FOLDER, "jupyterlab_extensions.yml")
    )
    with open(extensions_file) as fp:
        extensions = yaml.safe_load(fp)

    # install the code-template modules
    with py_env(c, env_name):

        # install the current package
        c.run(f"pip install -e {HERE}")

        is_jupyter = False
        if "jupyterlab-" in env_cfg:
            is_jupyter = True

        if is_jupyter:
            # install jupyterlab extensions
            for extension in extensions["extensions"]:
                extn_name = "@{channel}/{name}@{version}".format(**extension)
                c.run(f"jupyter labextension install --no-build {extn_name}", )

            out = c.run("jupyter lab build")


@task(name="setup_addon")
def setup_addon(
        c,
        platform=PLATFORM,
        env=DEV_ENV,
        documentation=False,
        testing=False,
        formatting=False,
        jupyter=False,
        extras=False,
):
    """Installs add on packages related to documentation, jupyter-lab or code-formatting.

    Dependencies related to documentation, testing or code-formatting can be installed on demand.
    By Specifying `--documentation`, documentation related packages get installed. Similarly to install testing
    or formatting related packages, flags `--testing` `formatting` will do the needful installations.
    """
    env_name = _get_env_name(platform, env)

    addons_documentation = op.abspath(
        op.join(CONDA_ENV_FOLDER, f"addon-documentation-{platform}-{env}.yml")
    )
    addons_testing = op.abspath(
        op.join(CONDA_ENV_FOLDER, f"addon-testing-{platform}-{env}.yml")
    )
    addons_formatting = op.abspath(
        op.join(CONDA_ENV_FOLDER, f"addon-code_format-{platform}-{env}.yml")
    )
    addons_jupyter = op.abspath(
        op.join(CONDA_ENV_FOLDER, f"addon-jupyter-{platform}-{env}.yml")
    )
    addons_extras = op.abspath(
        op.join(CONDA_ENV_FOLDER, f"addon-extras-{platform}-{env}.yml")
    )
    with py_env(c, env_name):
        if documentation:
            c.run(
                f"conda env update --name {ENV_PREFIX}-{DEV_ENV} --file {addons_documentation}"
            )
        if testing:
            c.run(
                f"conda env update --name {ENV_PREFIX}-{DEV_ENV} --file {addons_testing}"
            )
        if formatting:
            c.run(
                f"conda env update --name {ENV_PREFIX}-{DEV_ENV} --file {addons_formatting}"
            )
        if jupyter:
            c.run(
                f"conda env update --name {ENV_PREFIX}-{DEV_ENV} --file {addons_jupyter}"
            )

            extensions_file = op.abspath(
                op.join(CONDA_ENV_FOLDER, "jupyterlab_extensions.yml")
            )
            with open(extensions_file) as fp:
                extensions = yaml.safe_load(fp)

            for extension in extensions["extensions"]:
                extn_name = "@{channel}/{name}@{version}".format(**extension)
                c.run(f"jupyter labextension install --no-build {extn_name}", )

            out = c.run("jupyter lab build")
        if extras:
            c.run(
                f"conda env update --name {ENV_PREFIX}-{DEV_ENV} --file {addons_extras}"
            )
    if documentation:
        os.makedirs(op.join(HERE, "docs/build"), exist_ok=True)
        os.makedirs(op.join(HERE, "docs/source"), exist_ok=True)
    if extras:
        os.makedirs(op.join(HERE, "mlruns"), exist_ok=True)


@task(name="format-code")
def format_code(c, platform=PLATFORM, env=DEV_ENV, path="."):
    """Format the code using black and isort tools.

    """
    env_name = _get_env_name(platform, env)
    with py_env(c, env_name):
        c.run(f"black {path}", warn=True)
        c.run(f"isort -rc {path}")


@task(name="refresh-version")
def refresh_version(c, platform=PLATFORM, env=DEV_ENV):
    """Print the version of the core module.

    """
    env_name = _get_env_name(platform, env)
    with py_env(c, env_name):
        res = c.run(f"python {HERE}/setup.py --version")
    return res.stdout


@task(name="info")
def setup_info(c, platform=PLATFORM, env=DEV_ENV):
    """List down all the installed modules in the project env.

    """
    env_name = _get_env_name(platform, env)
    with py_env(c, env_name):
        res = c.run(f"pip list")
    return res.stdout


_create_task_collection(
    "dev",
    setup_env,
    format_code,
    refresh_version,
    setup_addon,
    setup_info
)


# -----------
# Build tasks
# -----------
@task(name="docs")
def build_docs(c, platform=PLATFORM, env=DEV_ENV, regen_api=True, update_credits=False):
    """Build the docs.

    """
    env_name = _get_env_name(platform, env)
    with py_env(c, env_name):
        if regen_api:
            code_path = op.join(HERE, "docs", "source", "_autosummary")
            if os.path.exists(code_path):
                _clean_rmtree(code_path)
            os.makedirs(code_path, exist_ok=True)

        # FIXME: Add others, flake9-black, etc
        if update_credits:
            credits_path = op.join(HERE, "docs", "source", "_credits")
            if os.path.exists(credits_path):
                _clean_rmtree(credits_path)
            os.makedirs(credits_path, exist_ok=True)
            authors_path = op.join(HERE, "docs")
            token = os.environ['GITHUB_OAUTH_TOKEN']
            c.run(
                f"python {authors_path}/generate_authors_table.py {token} {token}"
            )
        c.run("cd docs/source && sphinx-build -T -E -W --keep-going -b html -d ../build/doctrees  . ../build/html")


_create_task_collection("build", build_docs)


# -----------
# Launch stuff
# -----------
@task(name="jupyterlab")
def start_jupyterlab(
        c, platform=PLATFORM, env=DEV_ENV, ip="localhost", port=8080, token="", password=""
):
    """Launch the jupyter lab.

    """
    env_name = _get_env_name(platform, env)
    # FIXME: run as a daemon and support start/stop using pidfile
    with py_env(c, env_name):
        print(f"{'--' * 20} \n Running jupyterlab with {env_name} environment \n {'--' * 20}")
        c.run(
            f"jupyter lab --ip {ip} --port {port} --NotebookApp.token={token} "
            f"--NotebookApp.password={password} --no-browser"
        )


@task(name="docs")
def start_docs_server(c, ip="127.0.0.1", port=8081):
    """Launch the docs.

    """
    # FIXME: run as a daemon and support start/stop using pidfile
    print(f"{'--' * 20} \n Serving docs at: http://{ip}:{port} \n {'--' * 20}")
    c.run(
        f"python -m http.server --bind 127.0.0.1 " f"--directory docs/build/html {port}"
    )


@task(name="ipython")
def start_ipython_shell(c, platform=PLATFORM, env=DEV_ENV):
    """Launch the IPython shell.

    """
    env_name = _get_env_name(platform, env)
    # FIXME: run as a daemon and support start/stop using pidfile
    startup_script = op.join(HERE, "deploy", "ipython", "default_startup.py")
    with py_env(c, env_name):
        c.run(f"ipython -i {startup_script}")


_create_task_collection(
    "launch",
    start_jupyterlab,
    start_docs_server,
    start_ipython_shell,
)


@task(name="enroll")
def run_enroll(c, platform=PLATFORM, env=DEV_ENV, prod_path=None, re_enroll=0):
    """Run enroll process for all households.

    """
    env_name = _get_env_name(platform, env)
    with py_env(c, env_name):
        if prod_path is None:
            prod_path = op.join(HERE, "production")
        c.run(
            f"python {prod_path}/test.py enroll {re_enroll}"
        )


@task(name="inference")
def run_inference(c, platform=PLATFORM, env=DEV_ENV, prod_path=None):
    """Run inference process for all households.

    """
    env_name = _get_env_name(platform, env)
    with py_env(c, env_name):
        if prod_path is None:
            prod_path = op.join(HERE, "production")
        c.run(
            f"python {prod_path}/test.py inference"
        )


@task(name="train_effnet")
def train_effnet(c, platform=PLATFORM, env=DEV_ENV, prod_path=None):
    """Run the training process for EfficientNetB2.

    """
    env_name = _get_env_name(platform, env)
    with py_env(c, env_name):
        if prod_path is None:
            prod_path = op.join(HERE, "production")
        c.run(
            f"python {prod_path}/test.py train-effnet"
        )


@task(name="train_yolo")
def train_yolo(c, platform=PLATFORM, env=DEV_ENV, prod_path=None):
    """Run the training process for YOLOv5.

    """
    env_name = _get_env_name(platform, env)
    with py_env(c, env_name):
        if prod_path is None:
            prod_path = op.join(HERE, "production")
        c.run(
            f"python {prod_path}/test.py train-yolo"
        )


_create_task_collection(
    "run",
    run_enroll,
    run_inference,
    train_effnet,
    train_yolo
)


# --------------
# Root namespace
# --------------
# override any configuration for the tasks here
# FIXME: refactor defaults (constants) and set them as config after
# auto-detecting them
ns = _create_root_task_collection()
config = dict(pty=True, echo=True)

if OS == "windows":
    config["pty"] = False

ns.configure(config)
