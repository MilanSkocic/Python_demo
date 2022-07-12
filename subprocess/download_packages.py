r""""
Download packages for  local repositoryy
"""

import sys
import pathlib
import subprocess
import configparser
import argparse


DEST = pathlib.Path.home() / "wheels"
LOG = pathlib.Path.home() / "packages.log"

def download_packages(packages_to_download, 
                      platforms_to_download, 
                      python_versions_to_download,
                      dest=DEST, 
                      verbose=False,
                      timeout=60,
                      pre=False):

    r"""Download packages."""
    if verbose:
        print(f"Saving packages in {str(DEST):s}")
    cfg = configparser.ConfigParser()

    cfg.read("./packages.conf")

    if packages_to_download[0]=="all":
        sections = cfg.sections()
    else:
        sections = packages_to_download

    for i in packages_to_download:
        if i != "all":
            if i not in cfg.sections():
                raise ValueError(f"Package {i:s} was not found in the configuration file.")

    no_deps = False
    logfile = open(LOG, "w")
    n = len(sections)
    for ix, package in enumerate(sections):
        print(f"########## Downloading {package} {ix:02d}/{n:02d} ##########")
        _platforms = [i.replace(" ", "") for i in cfg[package]["platforms"].split(',')]
        if platforms_to_download[0] == "all":
            platforms = _platforms
        else:
            platforms = platforms_to_download

        for i in platforms_to_download:
            if i != "all":
                if i not in _platforms:
                    raise ValueError(f"Platform {i:s} was not found for package {package:s}")

        try:
            py_no_deps = [i.replace(" ","") for i in cfg[package]["no_deps"].split(',')]
        except KeyError:
            py_no_deps = []
        for platform in platforms:
            for key in cfg[package]:
                if key.startswith("py"):
                    python_version = key.split("_")[1]
                    if python_versions_to_download[0] == "all":
                        python_versions_to_download.append(python_version)
                    if python_version in python_versions_to_download:
                        versions=[i.replace(" ","") for i in cfg[package][key].split(',')]
                        for i in py_no_deps:
                            if i == python_version:
                                no_deps=True
                            else:
                                if i == "*":
                                    no_deps = True
                                else:
                                    no_deps = False
                        for version in versions:
                            if version != "None":
                                if version == "*":
                                    package_to_download = f"{package:s}"
                                else:
                                    package_to_download = f"{package}=={version}"

                                pip_command =  [sys.executable, "-m", "pip", "download", 
                                                f"--dest={str(dest):s}", 
                                                f"--only-binary=:all:",
                                                f"--platform={platform:s}",
                                                f"--python-version={python_version:s}"]
                                if no_deps:
                                    pip_command += ["--no-deps"]

                                if pre:
                                    pip_command += ["--pre"]
                                    
                                pip_command += [package_to_download]

                                try: 
                                    res = subprocess.run(pip_command, check=True, capture_output=True, timeout=timeout)
                                    logfile.write(res.stdout.decode("utf-8"))
                                    logfile.write(res.stderr.decode("utf-8"))
                                    if verbose:
                                        print("stdout = ", res.stdout.decode("utf-8")[:-1])
                                        print("stderr = ", res.stderr.decode("utf-8")[:-1])
                                    print(f"Success: {package} {platform} {python_version}")
                                except subprocess.CalledProcessError as e:
                                    logfile.write(e.stdout.decode("utf-8"))
                                    logfile.write(e.stderr.decode("utf-8"))
                                    if verbose:
                                        print("stdout = ", e.stdout.decode("utf-8")[:-1])
                                        print("stderr = ", e.stderr.decode("utf-8")[:-1])
                                    print(f"Failed: {package} {platform} {python_version} (see packages.log)")
                                except subprocess.TimeoutExpired as e:
                                    logfile.write(f"Timeout: {package} {platform} {python_version}")
                                    if verbose:
                                        print(f"Timeout: {package} {platform} {python_version}\n")
                                    print(f"Failed: {package} {platform} {python_version} (see packages.log)")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download packages for local repository.")
    parser.add_argument("packages", type=str, nargs="*", default=["all"], 
                        help="Packages to download. Default all. The packages must be defined in a config file.")
    parser.add_argument("--platforms", "-p", type=str, nargs="*", default=["all"],
                        help="Platforms to download. Default all. The platforms must be defined in the config file.")
    parser.add_argument("--python_versions", "-pyv", type=str, nargs="*", default=["all"],
                        help="Python versions to download. Default all. The python versions must be defined in the config file.")
    parser.add_argument("--dest", "-d", type=str, default=DEST, help="Path to the download folder.")
    parser.add_argument("--verbose", "-v", 
                        action="store_true", 
                        default=False, help="Display stdout and stderr.")
    parser.add_argument("--timeout", "-t", type=int, default=60,
                        help="Timeout in seconds.")
    parser.add_argument("--pre", "-a", action="store_true", default=False, help="Allow pre-releases.")
    args = parser.parse_args()
    download_packages(packages_to_download=args.packages, 
                      platforms_to_download=args.platforms,
                      python_versions_to_download=args.python_versions,
                      dest=args.dest, 
                      verbose=args.verbose,
                      timeout=args.timeout,
                      pre=args.pre)  
