# NGC Toolbox

### Table of Contents
1. [New to NGC?](#new-to-ngc)
    - [Prerequisites](#prerequisites)
    - [Setting up](#setting-up)
2. [Using this toolbox](#using-this-toolbox)
    - [Basic workflow](#basic-workflow)
        1. [Create a docker container](#1-create-a-docker-container)
        2. [Create a workspace](#2-create-a-workspace)
        3. [Creating jobs](#3-creating-jobs)
    - [A full working example](#a-full-working-example)
4. [Advanced usage](#advanced-usage)
5. [A small note on Gitlab and NGC](#a-small-note-on-gitlab-and-ngc)
6. [Miscellaneous notes](#miscellaneous-notes)

## New to NGC?

This section is designed to be a minimal guide to getting started with NGC. More comprehensive guides exist, such as the [quick-start guide](https://docs.google.com/document/d/1lWYoqLaqTs8KqP1p1ZfEEYsap9hZFPl8GD6aARTQtG8/) or [user guide](https://docs.google.com/document/d/1kDdYTrEfhmpvTFCAtfw_Ad-KPHSoTv34PaLqL21Tipc/). If any of these instructions are incorrect, [notify James Lucas](mailto:jalucas@nvidia.com) and check one of the previous guides for up-to-date instructions.

### Prerequisites

- [Request access to your NGC team](https://goo.gl/forms/IjKBiZRt4RYZcF3h1) (`ct-toronto-ai` for the Toronto AI team)
    - Once approved, you will receive a sign-in link to set your password
- [Download and install the NGC CLI](https://ngc.nvidia.com/setup/installers/cli)
- [Install docker](https://docs.docker.com/get-docker/)

To use this toolbox, you need python and the `python-fire` and `toml` packages:

`pip install -r requirements.txt`

### Setting up

- [Generate an NGC API key](https://ngc.nvidia.com/configuration/api-key) (and keep it somewhere safe)
- Set key in docker: `docker login -u \$oauthtoken -p <YOUR_API_KEY> nvcr.io `
- Configure NGC CLI: `ngc config set`
    - Input your API key when requested

**You're ready to go!**

## Using this toolbox

This toolbox provides a simple python-based CLI for interacting with NGC. It supports:

- Container management
    - Creating new docker containers
    - Updating docker containers
- Workspace management
    - Creating, mounting, and unmounting workspaces
- Job launching
    - Single jobs
    - Arrays of jobs
    - Interactive jobs

The toolbox is fairly minimal by design and basically provides some lightweight wrappers around the NGC CLI tool.
Documentation for the ngc app interface can be found via `python app.py --config ngc_config/example.toml --help`. You could include this repo as a submodule in an existing repository, or just copy-paste it in directly.

### A full working example

0. Connect to the VPN
1. Change the `ngc.workspace.id` value in the `ngc_config/example.toml` to `<YOUR_NAME>_example_workspace`
2. Change the `docker.name` in the `ngc_config/example.toml` to `<YOUR_NAME>_example_container`
    - Change the team too if necessary
3. Run:

```bash
python app.py --config ngc_config/example.toml create_workspace
python app.py --config ngc_config/example.toml build_docker
python app.py --config ngc_config/example.toml push_docker
python app.py --config ngc_config/example.toml submit_job "python example/example.py" example
```

If successful, you will see the job on the [NGC dashboard](https://ngc.nvidia.com/dashboard).


*Note that the first command only ever needs to be run once. The second and third commands should be run only when you have made a change to your docker container.*

### Configuration

The toolbox is configured via a `.toml` config file. This allows you to set the job parameters, and your workspace/container information. An example config is shown below, but I recommend you skip ahead to the [basic workflow](#basic-workflow) if you're just getting started.

```toml
name = "example_app"
description = "Example NGC Application: An example application for showcasing NGC toolbox."
# This command is run before any other specified commands (see README)
#    Note: workspace_path is pulled from the `[ngc.workspace.path]` value
#          end this command with `;` as additional commands will be appended
base_command = "cd {workspace_path}/experiments/{exp_name}; . ./ngc_prepare.sh;" # <- You shouldn't need to change this

[ngc]
name = "ml-model.notamodel.example_app" # <- You should change this
open_ports = [6006, 8888]
result_path = "/result"

[ngc.ace]
id = 257
name = "nv-us-west-2"
instance = "dgx1v.16g.4.norm" # <- 16G memory, 4 GPUs

# Can mount more by adding more `[ngc.workspaceXYZ]` categories
# but this one will be used by the local workspace commands
[ngc.workspace]
path = "/home/jalucas/workspace" # <- You should change this
id = "example_workspace" # <- You should change this
mode = "RW"

[docker]
team = "nvidian/ct-toronto-ai"
name = "example_container" # <- You should change this
platform = "linux/amd64"
```

### Basic workflow

I'll assume at this point that you can run your code locally. Here I'll walk through the process of getting this same command running on an NGC node. It consists of three parts:

1. Create a docker container (only need to do once)
2. Create a workspace (only need to do once)
3. Create the NGC job(s)

The first two are technically optional for NGC but are (currently) required to use this toolbox. The next few sections walk through how to build the config file and run your job on NGC.

#### **1. Create a docker container**

This step isn't strictly necessary (you can use an existing container), but it is useful to have your own container in place in case you need to add extra dependencies in future.

This folder contains a `Dockerfile.build` that is used to create your container. _You should modify the line `LABEL org.opencontainers.image.authors="jalucas@nvidia.com"` to label yourself as author._

Any other software dependencies, particularly those that are slow to install, should be installed via this file. _For simplicity, we're not going into the details of how to do this here._

You can configure your docker container in the `[docker]` section of the config file. For example:

```toml
[docker]
team = "nvidian/ct-toronto-ai" # <- Your team
name = "example_container" # <- The desired name of your container
platform = "linux/amd64"
```

**Creating/updating the container**

First,

`python app.py --config ngc_config/example.toml build_docker`

Then,

`python app.py --config ngc_config/example.toml push_docker`

#### **2. Create a workspace**

A workspace is just a network drive that can be mounted locally or on NGC. It is a useful place to stick your source code and other persistent files that your job needs access to. _Important: your workspace is not always a good choice for hosting data as it can be slow. Instead use NGC datasets, swiftstack, or move from your workspace into the faster `/raid` storage on NGC nodes._

Workspaces can be configured under the `[ngc.workspace]` header:

```toml
[ngc.workspace]
path = "/home/jalucas/workspace" # <- Path to mount workspace on NGC
id = "example_workspace" # <- The desired name of your workspace
mode = "RW" # <- 'RW' if you need to read and write.
```

**Creating a workspace**

Simply run `python app.py --config ngc_config/example.toml create_workspace`

This will create a new workspace on NGC with the `id` from the config file.

**Mounting the workspace locally**

Mount using,

`python app.py --config ngc_config/example.toml mount_workspace`

This will mount the workspace on your local machine. By default, it will be mounted at the `path` specified in the config file (`/home/jalucas/workspace` above).

And unmount with,

`python app.py --config ngc_config/example.toml unmount_workspace`

**Note that you need to be connected to the corporate VPN to mount workspaces locally.**

#### **3. Creating jobs**

This toolbox assumes a certain workflow (that over time I have found to be the most flexible and extensible). Creating a job with this toolbox consists of two parts. First, synchronize the local directory to your workspace in an experiment subfolder:

`python app.py --config ngc_config/example.toml sync_workspace my_experiment`

This will create a new directory in your workspace at the path `experiments/my_experiment`. And all `.py` source code will be copied into this directory.


The second part dispatches the job to NGC. Suppose you already have a command that executes locally: `python train.py --arg1 value1 --arg2 value2`. If you have followed all steps above, you should be able to simply run:

`python app.py --config ngc_config/example.toml run_job "python train.py --arg1 value1 --arg2 value2"`

The job will spin up on ngc, and should be visible on the [NGC dashboard](https://ngc.nvidia.com/dashboard).

Note that these two steps can be combined using the `submit_job` method:

`python app.py --config ngc_config/example.toml submit_job "python train.py --arg1 value1 --arg2 value2"`

**Interactive jobs**

To run an interactive job: `python app.py --config ngc_config/example.toml run_interactive_job --runtime 4h`

The job will spin up and you'll be assigned a job ID (presented in the command output). You can connect to the job via `ngc batch exec <job_id>`.

**A large batch of jobs**

First, generate a file where each line corresponds to a different command to be run. For example,

> grid_search.txt
```
python train.py --arg1 a1 --arg2 a2
python train.py --arg1 b1 --arg2 a2
python train.py --arg1 a1 --arg2 b2
python train.py --arg1 b1 --arg2 b2
```

Then execute `python app.py --config ngc_config/example.toml generate_job_array grid_search.txt grid_search grid_search_jobs/`.
The final three arguments are the file containing the commands, a name for the experiment, and the output folder to place the job data.
Running this command creates a new directory called `grid_search_jobs` containing the files `[cmd_0.json, cmd_1.json, cmd_2.json, cmd_3.json]`. You can then dispatch all jobs to NGC via,

`python app.py --config ngc_config/example.toml run_job_array grid_search_jobs`

_This could also be achieve in one step, by adding the `--run` flag to the `generate_job_array` command._

## Advanced usage

When I use this toolbox in a larger project I make some minor tweaks.

First, I put the contents of this repository into a subfolder, e.g. `project_root/ngc/`. This means that a couple other things need to be changed:

- The `base_command` in the config needs to be updated to reflect the new path of `ngc_prepare.sh`
- When building docker containers, you need to specify the path to the docker file: `python app.py --config ngc_config/example.toml build_docker --docker_fpath ngc/Dockerfile`

Second, the `sync_workspace`/`submit_job` methods in this toolbox synchronize only `.py` and `.sh` files. You may want to modify this method to include other files, for example YAML config files for specifying hyperparameters.


## A small note on gitlab and NGC

NGC nodes can access gitlab so you can clone repositories within jobs. To access gitlab repos, you should use project access tokens configured to only allow read access. These can be created in `Settings > Access Tokens` via the Gitlab UI. Alternatively, you could manually sync the source code to your workspace and skip the cloning stage, as this toolbox supports by default.

## Miscellaneous notes

- If you get tired of writing out the config path, you can set this as a default option in the `NGCToolbox` `__init__` function within `app.py`.
- This toolbox is designed to be fairly minimal so that it is easy to use. For example, there is no support for NGC datasets and support for multiple workspaces is very limited. However, it should be easy to extend and build upon in your own setting.
- If you have any suggestions for how to improve this toolbox please open issues, MRs, or just [contact me directly](mailto:jalucas@nvidia.com)!

