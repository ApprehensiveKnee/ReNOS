import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SRC_DIR)
CONFIG_FILES_DIR = os.path.join(MAIN_DIR, "config_files")
RESTART_DIR = os.path.join(SRC_DIR, "restart")
ARCH_FILE = os.path.join(CONFIG_FILES_DIR, "arch.json")
RUN_FILES_DIR = os.path.join(CONFIG_FILES_DIR, "runs")
CONFIG_DUMP_DIR = os.path.join(CONFIG_FILES_DIR, "dumps")
DATA_DIR = os.path.join(MAIN_DIR, "data")
PYTHON_MODULE_DIR = os.path.join(MAIN_DIR, "build/lib")