import sys
import math


def decompose_filepath(filepath):
    """
    decompose filepath into three components:
    directory path, file name and extension
    """
    parent_directories = filepath.split("/")[:-1]
    try:
        if parent_directories[-1] == "":
            del parent_directories[-1]
        dir_path = "/".join(parent_directories)
    except IndexError:
        dir_path = None

    File = filepath.split("/")[-1]
    try:
        [filename, extension] = File.split(".")
    except ValueError:
        filename = File
        extension = None

    return (dir_path, filename, extension)
