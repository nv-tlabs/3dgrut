"""A utility script for interfacing with NGC.

## Simple(-ish) explanation of how (single node) NGC jobs work:

Your job consists of a command to run and some metadata. This metadata specifies the kind of machine
that you want the job to run on, the docker image to use, and more.
You submit the job and it is created server-side, enters the queue, and starts once an instance is available.

As the instance spins up, the docker image that you are using is instantiated.
Once it is running, the command is executed within the container environment. When finished, the job terminates.

All storage within the container itself is temporary and will be lost once the job is terminated. But we are able
share access to network drives with the instance. These are provided as "workspaces" or the "results" directory and
can be configured with the job metadata. These persist even after the job terminates.

"""

from fire import Fire
import os
import json
import toml
import logging
from typing import Optional
from copy import deepcopy

def load_config(path):
    config = None
    try:
        config = toml.load(path)
        logging.debug("Configuration: " + str(config))
    except toml.TomlDecodeError as err:
        logging.critical("Configuration error:" + err.msg)
        exit(1)
    return config


def get_cmd_template(config, exp_name):
    template = {
        "name": config["ngc"]["name"],
        "description": config["description"],
        "command": config["base_command"].format(
            workspace_path=config["ngc"]["workspace"]["path"], exp_name=exp_name
        ),
        "aceId": config["ngc"]["ace"]["id"],
        "aceName":config["ngc"]["ace"]["name"],
        "aceInstance":config["ngc"]["ace"]["instance"],
        "dockerImageName": "{0}/{1}:{2}".format(config["docker"]["team"], config["docker"]["name"],config["docker"].get("tag","latest")),
        "publishedContainerPorts": config["ngc"]["open_ports"],
        "datasetMounts": [],
        "resultContainerMountPoint": config["ngc"]["result_path"],
        "runPolicy": {
            "preemptClass": "RESUMABLE"
        },
        "team": "omniverse"
    }

    # Add all workspaces
    workspaces = []
    for key in config["ngc"].keys():
        if key.startswith("workspace"):
            workspaces.append(
                {
                    "containerMountPoint": config["ngc"][key]["path"],
                    "id": config["ngc"][key]["id"],
                    "mountMode": config["ngc"][key]["mode"]
                }
            )
    template["workspaceMounts"] = workspaces
    return template


def dump_and_run(fpath, cmd_json, dry_run=False, cleanup=False):
    with open(fpath, "w") as f:
        json.dump(cmd_json, f)
    cmd = "ngc batch run -f " + fpath
    print(cmd)
    if not dry_run:
        os.system(cmd)
        if cleanup:
            os.remove(fpath)


class NGCToolbox:
    """NGC Toolbox

    A set of useful utility functions to make interfacing with NGC simpler.

    Args:

        config: Path to config file.
    """

    def __init__(self, config = "./utils/ngc/ngc_config/3dgrt.toml"):
        self._cfg = load_config(config)


    def build_docker(self, docker_fpath: Optional[os.PathLike] = None):
        """Build the docker container.
        """
        cmd = 'docker build -f ./utils/ngc/Dockerfile.build . -t {0}'.format(self._cfg["docker"]["name"])
        if docker_fpath is not None:
            cmd += f" -f {docker_fpath}"
        print(cmd)
        os.system(cmd)


    def push_docker(self):
        """Tag and push the docker container to nvcr.io
        """
        cmd = 'docker tag {0} nvcr.io/{1}/{0}'.format(self._cfg["docker"]["name"], self._cfg["docker"]["team"])
        print(cmd)
        os.system(cmd)

        cmd = 'docker push nvcr.io/{1}/{0}'.format(self._cfg["docker"]["name"], self._cfg["docker"]["team"])
        print(cmd)
        os.system(cmd)

    def create_workspace(self):
        """Create a new workspace using the given config file.
        """
        cmd = "ngc workspace create --name {0}".format(self._cfg["ngc"]["workspace"]["id"])
        print(cmd)
        os.system(cmd)

    def mount_workspace(self, local_dir=None, mode="RW"):
        """Mount workspace to local storage

        Args:
            local_dir:
                Path to directory in which to mount the workspace. By default
                uses the path set in the config.
            mode:
                Mount mode (default: RW).
        """
        ws_path = self._cfg["ngc"]["workspace"]["path"]
        ws_id = self._cfg["ngc"]["workspace"]["id"]
        cmd = "ngc workspace mount {0} {1} --mode {2}".format(ws_id, local_dir if local_dir is not None else ws_path, mode)
        print(cmd)
        os.system(cmd)

    def unmount_workspace(self, local_dir=None):
        """Unmount workspace from local storage

        Args:
            local_dir:
                Path to directory in which to mount the workspace. By default
                uses the path set in the config.
        """
        ws_path = self._cfg["ngc"]["workspace"]["path"]
        cmd = "ngc workspace unmount {0}".format(local_dir if local_dir is not None else ws_path)
        print(cmd)
        os.system(cmd)


    def sync_workspace(self, exp_name, dry_run=False):
        """Sync the local directory to the workspace under a
        new experiments/{exp_name} directory.

        This method copies all python files in the current local directory and places
        them in the NGC workspace under the directory path `experiments/{exp_name}.

        Note: You will likely need to modify this method to include other file paths.
        For example, you may also want to sync YAML config files.

        Args:
            exp_name:
                Identifier for the experiment. The workspace will be synced
                in the directory `experiments/<exp_name>`.
            dry_run:
                Create a temporary sync folder at /tmp/{exp_name} but do
                not upload to the NGC workspace.
        """
        # use NGC workspace upload
        ws_id = self._cfg["ngc"]["workspace"]["id"]
        tmpdir = os.path.join('/tmp', exp_name)
        os.makedirs(tmpdir, exist_ok=True)

        rsync_base_cmd = "rsync -arzh --prune-empty-dirs --no-links -R --include=*/ --exclude=runs/** --exclude=outputs/** --exclude=wandb/** "
        rsync_cmds = [f"{rsync_base_cmd}  --include=*.py --exclude=* . {tmpdir}/ --delete"]
        rsync_cmds.append(f"{rsync_base_cmd} --include=*.cpp --exclude=* . {tmpdir}/")
        rsync_cmds.append(f"{rsync_base_cmd} --include=*.h --exclude=* . {tmpdir}/")
        rsync_cmds.append(f"{rsync_base_cmd} --include=*.cu --exclude=* . {tmpdir}/")
        rsync_cmds.append(f"{rsync_base_cmd} --include=*.yaml --exclude=* . {tmpdir}/")        
        rsync_cmds.append(f"{rsync_base_cmd} --include=*.sh --exclude=* . {tmpdir}/")
        for rs_cmd in rsync_cmds:
            print(rs_cmd)
            os.system(rs_cmd)

        # Store current commit hash and time in the workspace
        os.system(f"date +\"%A, %m %d %Y %H:%M\" >> {tmpdir}/commit.txt")
        os.system(f"git rev-parse HEAD >> {tmpdir}/commit.txt")

        cmd = f"ngc workspace upload --destination experiments/{exp_name} --source {tmpdir} {ws_id}"
        print(cmd)
        if not dry_run:
            os.system(cmd)
        else:
            print(f"DRY RUN    Workspace snapshot saved to {tmpdir}")

    def submit_job(self, command, exp_name, dry_run=False, cleanup=True):
        """Combines sync_workspace and run_job together.
        """
        self.sync_workspace(exp_name, dry_run=dry_run)
        self.run_job(command, exp_name, dry_run=dry_run, cleanup=cleanup)


    def run_interactive_job(self, exp_name="", runtime='4h', dry_run=False, cleanup=True):
        """Run NGC interactive job.

        Runs a job with a final sleep command. Can be later connected to via `ngc batch exec <job_id>`.

        Args:
            runtime:
                Length of time to sleep for. (Default: `4h`)
            exp_name:
                Experiment name. The interactive job will attempt to change directory to
                experiments/{exp_name} on the mounted workspace. This is set to an empty
                string by default.
            dry_run:
                Set up and print the command without sending to NGC. Will always keep
                JSON job configuration. (Default: `False`)
            cleanup:
                If `True`, remove the JSON job configuration file unless doing a dry-run. (Default: `True`)

        """
        cmd_json = get_cmd_template(self._cfg, exp_name)
        cmd_json["name"] += "_interactive"
        cmd_json["command"] += "sleep {}".format(runtime)
        dump_and_run("interactive.json", cmd_json, dry_run, cleanup)

    def run_job(self, command, exp_name, dry_run=False, cleanup=True):
        """Run a single job from a given command.

        Args:
            command:
                Command to run on NGC (executed after the base_command in the config).
            dry_run:
                Set up and print the command without sending to NGC. Will always keep
                JSON job configuration. (Default: `False`)
            cleanup:
                If `True`, remove the JSON job configuration file unless doing a dry-run. (Default: `True`)
        """
        cmd_json = get_cmd_template(self._cfg, exp_name)
        cmd_json["command"] += command
        cmd_json["command"] += " ; . ./utils/ngc/ngc_post_job.sh"
        dump_and_run("command.json", cmd_json, dry_run, cleanup)


    def generate_job_array(self, cmd_file, exp_name, jobdir, keep_dir=False, run=False):
        """Generate a job array to submit in batch to NGC.

        Args:
            cmd_file:
                Path to file containing commands to run
            jobdir:
                Directory to store the output command files in.
                Note: This directory should not yet exist. Set the `keep_dir` flag
                to allow jobs to be placed in an existing directory.
            keep_dir:
                Allow an existing job directory to be used and files overwritten.
                IMPORTANT: If `True`, existing files in the directory are not deleted.
                (Default: `False`)
            run:
                Run the job after creating the NGC commands. (Default: `False`)
        """
        # Create the job directory to store the commands
        try:
            jobdir = os.path.join('./utils/ngc/grid_search_configs/',jobdir)
            os.makedirs(jobdir, exist_ok=keep_dir)
        except FileExistsError:
            print("Job directory already exists. Please delete it (or provide --keep_dir flag)")
            print("Terminating...")
            return

        with open(cmd_file) as f:
            cmds = f.readlines()
        # Iterate through the file directory
        job_template = get_cmd_template(self._cfg, exp_name)
        for i, cmd in enumerate(cmds):
            job_id = str(i)
            cmd_json = deepcopy(job_template)
            cmd_json["name"] += "." + job_id
            cmd_json["command"] += cmd.format(exp_name=exp_name)
            cmd_json["command"] += " ; . ./utils/ngc/ngc_post_job.sh"
            job_fpath = os.path.join(jobdir, "cmd_{}.json".format(job_id))

            with open(job_fpath , "w") as f:
                json.dump(cmd_json, f)

        if run:
            self.run_job_array(jobdir)
        
    def run_job_array(self, jobdir):
        """Run an array of jobs on NGC

        jobdir:
            Directory containing the json command files.
        """
        for entry in os.scandir(jobdir):
            if entry.is_file() and entry.name.lower().endswith(".json"):
                cmd = "ngc batch run -f " + entry.path
                os.system(cmd)



if __name__ == '__main__':
    Fire(NGCToolbox)
