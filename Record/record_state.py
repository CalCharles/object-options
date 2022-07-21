import os, time
import imageio as imio
from Record.file_management import append_string, create_directory, action_chain_string

class FullRecord():
    def __init__(self, itr, save_dir, recycle, save_raw, all_dir=""):
        '''
        starting iteration number
        '''
        self.itr, self.save_path, self.recycle, self.save_raw = itr, save_dir, recycle, save_raw
        
        # create directories and initialize files to be appended to
        create_directory(os.path.join(save_dir, "logs"))
        object_dumps = open(os.path.join(self.save_path, "object_dumps.txt"), 'w') # create file if it does not exist, clears out existing files
        action_dumps = open(os.path.join(self.save_path, "action_dumps.txt"), 'w')
        option_dumps = open(os.path.join(self.save_path, "option_dumps.txt"), 'w')
        param_dumps = open(os.path.join(self.save_path, "param_dumps.txt"), 'w')
        object_dumps.close(), action_dumps.close(), option_dumps.close(), param_dumps.close()

    def save(self, entity_state, frame, toString): # TODO: put into parent class
        '''
        entity state is the factored state
        frame is the raw image
        toString is a function from the Environment
        '''
        if self.recycle > 0:
            state_path = os.path.join(self.save_path, str((self.itr % self.recycle)//2000))
            count = self.itr % self.recycle
        else:
            state_path = os.path.join(self.save_path, str(self.itr//2000))
            count = self.itr
        create_directory(state_path)
        
        append_string(os.path.join(self.save_path, "action_dumps.txt"), action_chain_string([entity_state["Action"]]) + "\t")
        append_string(os.path.join(self.save_path, "object_dumps.txt"), toString(entity_state) + "\n")
        if self.save_raw: imio.imsave(os.path.join(state_path, "state" + str(count % 2000) + ".png"), frame)
