import os, logging
from Record.file_management import create_directory

class Logger():
    def __init__(self, log_file):
        self.filename = log_file
        if len(log_file) > 0:
            log_path, filename = os.path.split(log_file)
            logging.basicConfig(filename=os.path.join(create_directory(log_path), filename), filemode='w', level=logging.DEBUG) # all filenames must match otherwise it's going to send to the last one
    
    def logout(self, log_string):
        logging.info(log_string)
        print(log_string)

    def logval(self, val):
        logging.info(str(val))
        print(val)
