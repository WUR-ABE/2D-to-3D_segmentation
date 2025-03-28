"""Utils """
import logging
from logging.handlers import RotatingFileHandler
from json import load
from sys import stdout
from os import makedirs, path, remove
from multiprocessing import Manager


def remove_files(data_folder, keystring: str):
    """Remove existing files that match the keystring"""
    for filename in list(data_folder.glob(keystring)):
        remove(filename)


def load_json(filepath):
    """Helper function to load a json file """
    try:
        with open(filepath, 'r') as f:
            data_json = load(f)
            return data_json
    except Exception as e:
        logging.error(f'Error loading file: {filepath}, {str(e)}')
    # Empty dictionary
    return {}


def get_logfolder():
    """Helper to create logfolder """
    cfd = path.dirname(path.realpath(__file__))
    log_folder = path.join(cfd, "logs")
    if not path.exists(log_folder) & path.isdir(log_folder):
        makedirs(log_folder)
    return log_folder


def setup_logger(name="marvinlib", mark_start=True):
    """Setup logger """
    log_file = path.join(get_logfolder(), name + '.log')

    log_format = "[%(asctime)s]{%(filename)s:%(lineno)d} %(levelname)s - %(message)s"
    # Add a file handler to the logger to store log messages long term, 20x 100mb = 2 GB of log files max
    handlers = [RotatingFileHandler(log_file, 'a', 1000000, 20), logging.StreamHandler(stdout)]#, logging.StreamHandler()
    logging.basicConfig(format=log_format, level=logging.INFO, handlers=handlers)

    logger = logging.getLogger(name)
    # Mark a new session in the logfile
    # Only when true, to avoid confusion when different processes initiate a logger
    if mark_start is True:
        with open(log_file, 'a') as file:
            file.write("\n\n")
            file.write("New session")
            file.write("\n\n")

    return logger
