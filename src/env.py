import subprocess as subp
import logging
import sys
import os

log = logging.getLogger("base")

# TODO add docs about parameters
path = os.path.dirname(os.path.abspath(__file__))


def setup_ca(dir_path):
    # FIXME: setup_ca.sh is updated
    """
    Call bash sript for settingup the local CA
    :param dir_path: working directory
    """
    # assert os.path.exists(dir_path), "Path is not exist"
    # assert os.path.isdir(dir_path), "Not a directory"
    log.debug("Start setup of local CA")
    out = subp.run(
        ["ansible-playbook", f"{path}/env/main.yml"])

    assert out.returncode == 0, "Something break in setup playbook :("
    log.debug("Setup of local CA is completed")


def cleanup_ca(dir_path):
    log.debug("Start cleanup of local CA")
    out = subp.run(
        ["bash", f"{dir_path}/SCAutolib/src/env/cleanup_ca.sh"])

    assert out.returncode == 0, "Something break in setup script :("
    log.debug("Cleanup of local CA is completed")


if __name__ == "__main__":
    assert len(sys.argv) < 4, "Too many input arguments"
    fnc = {"setup_ca": setup_ca, "cleanup_ca": cleanup_ca}
    try:
        name = sys.argv[1]
        dir_path = sys.argv[2]
        fnc[name](dir_path)
    except KeyError:
        log.error(f"Wrong function name: {name}")
        exit(1)
