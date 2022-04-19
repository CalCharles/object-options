
    def set_save(self, itr, save_dir, recycle, save_raw, all_dir=""):
        self.save_path=save_dir
        print(save_dir)
        self.itr = itr
        self.recycle = recycle
        self.all_dir = all_dir
        self.save_raw = save_raw
        try:
            os.makedirs(save_dir)
            os.makedirs(os.path.join(save_dir, "logs"))
        except OSError as e:
            print(e)
            pass
        object_dumps = open(os.path.join(self.save_path, "object_dumps.txt"), 'w') # create file if it does not exist
        action_dumps = open(os.path.join(self.save_path, "action_dumps.txt"), 'w')
        object_dumps.close()
        action_dumps.close()

    def write_objects(self, entity_state, frame): # TODO: put into parent class
        if self.recycle > 0:
            state_path = os.path.join(self.save_path, str((self.itr % self.recycle)//2000))
            count = self.itr % self.recycle
        else:
            state_path = os.path.join(self.save_path, str(self.itr//2000))
            count = self.itr
        try:
            os.makedirs(state_path)
        except OSError:
            pass

        if entity_state is not None:
            action_dumps = open(os.path.join(self.save_path, "action_dumps.txt"), 'a')
            action_dumps.write(action_toString(entity_state["Action"]) + "\t")
            action_dumps.close()
            object_dumps = open(os.path.join(self.save_path, "object_dumps.txt"), 'a')
            object_dumps.write(self.toString(entity_state) + "\n") # TODO: recycling does not stop object dumping
            object_dumps.close()
        if self.save_raw:
            imio.imsave(os.path.join(state_path, "state" + str(count % 2000) + ".png"), frame)
