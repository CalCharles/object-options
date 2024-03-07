import os

breakout_configs = [os.path.join("/hdd", "datasets", "object_data", "breakout", "random"), os.path.join("configs", "Breakout", "action_paddle_interaction.yaml"), os.path.join("configs", "Breakout", "action_paddle_mask.yaml"), os.path.join("configs", "Breakout", "action_paddle_option.yaml"), os.path.join("configs", "Breakout", "paddle_ball_interaction.yaml"), os.path.join("configs", "Breakout", "paddle_ball_mask.yaml"), os.path.join("configs", "Breakout", "paddle_ball_option.yaml")]
breakout_variant_configs = [os.path.join("configs", "Breakout", "center.yaml"), os.path.join("configs", "Breakout", "single.yaml"), os.path.join("configs", "Breakout", "neg.yaml"), os.path.join("configs", "Breakout", "prox.yaml"), os.path.join("configs", "Breakout", "big.yaml"), os.path.join("configs", "Breakout", "hard.yaml")]
robopushing_configs = [os.path.join("/nfs", "data", "object_data", "robopush", "random"), os.path.join("configs", "RoboPushing", "action_gripper_interaction.yaml"), os.path.join("configs", "RoboPushing", "action_gripper_mask.yaml"), os.path.join("configs", "RoboPushing", "action_gripper_option.yaml"), os.path.join("configs", "RoboPushing", "gripper_block_interaction.yaml"), os.path.join("configs", "RoboPushing", "gripper_block_mask.yaml"), os.path.join("configs", "RoboPushing", "gripper_block_option.yaml")]
obstacle_config = os.path.join("configs", "RoboPushing", "obstacles.yaml")
printer_configs = [os.path.join("/hdd", "datasets", "object_data", "minibehavior", "installing_printer", "random"), 
                   os.path.join("configs", "MiniBehavior", "installing_printer", "action_agent_interaction.yaml"), 
                   os.path.join("configs", "MiniBehavior", "installing_printer", "action_agent_mask.yaml"), 
                   os.path.join("configs", "MiniBehavior", "installing_printer", "action_agent_option.yaml"), 
                   os.path.join("configs", "MiniBehavior", "installing_printer", "agent_printer_interaction.yaml"), 
                   os.path.join("configs", "MiniBehavior", "installing_printer", "agent_printer_mask.yaml"), 
                   os.path.join("configs", "MiniBehavior", "installing_printer", "agent_printer_option.yaml"), ]
                #    os.path.join("configs", "MiniBehavior", "installing_printer", "printer_table_interaction.yaml"), 
                #    os.path.join("configs", "MiniBehavior", "installing_printer", "printer_table_mask.yaml"), 
                #    os.path.join("configs", "MiniBehavior", "installing_printer", "printer_table_option.yaml")]
install_printer_config = os.path.join("configs", "MiniBehavior", "installing_printer", "printer_install.yaml")

thaw_configs = [os.path.join("/hdd", "datasets", "object_data", "minibehavior", "thawing", "random"), 
                   os.path.join("configs", "MiniBehavior", "thawing", "action_agent_interaction.yaml"), 
                   os.path.join("configs", "MiniBehavior", "thawing", "action_agent_mask.yaml"), 
                   os.path.join("configs", "MiniBehavior", "thawing", "action_agent_option.yaml"), 
                   os.path.join("configs", "MiniBehavior", "thawing", "agent_fish_interaction.yaml"), 
                   os.path.join("configs", "MiniBehavior", "thawing", "agent_fish_mask.yaml"), 
                   os.path.join("configs", "MiniBehavior", "thawing", "agent_fish_option.yaml"), ]
thaw_fish_config = os.path.join("configs", "MiniBehavior", "thawing", "thaw_fish.yaml")
thaw_date_config = os.path.join("configs", "MiniBehavior", "thawing", "thaw_date.yaml")
thaw_olive_config = os.path.join("configs", "MiniBehavior", "thawing", "thaw_olive.yaml")
thaw_any_two_config = os.path.join("configs", "MiniBehavior", "thawing", "thaw_any_two.yaml")
thaw_all_config = os.path.join("configs", "MiniBehavior", "thawing", "thaw_all.yaml")

clean_configs = [os.path.join("/hdd", "datasets", "object_data", "minibehavior", "cleaning_car", "random"), 
                   os.path.join("configs", "MiniBehavior", "cleaning_car", "action_agent_interaction.yaml"), 
                   os.path.join("configs", "MiniBehavior", "cleaning_car", "action_agent_mask.yaml"), 
                   os.path.join("configs", "MiniBehavior", "cleaning_car", "action_agent_option.yaml"), 
                   os.path.join("configs", "MiniBehavior", "cleaning_car", "agent_rag_interaction.yaml"), 
                   os.path.join("configs", "MiniBehavior", "cleaning_car", "agent_rag_mask.yaml"), 
                   os.path.join("configs", "MiniBehavior", "cleaning_car", "agent_rag_option.yaml"), ]
clean_soak_rag_config = os.path.join("configs", "MiniBehavior", "cleaning_car", "soak_rag.yaml")
clean_car_config = os.path.join("configs", "MiniBehavior", "cleaning_car", "clean_car.yaml")
clean_rag_config = os.path.join("configs", "MiniBehavior", "cleaning_car", "clean_rag.yaml")

igibson_configs = [os.path.join("/hdd", "datasets", "object_data", "igibson", "random"), 
                   os.path.join("configs", "iGibson", "action_agent_interaction.yaml"), 
                   os.path.join("configs", "iGibson", "action_agent_mask.yaml"), 
                   os.path.join("configs", "iGibson", "action_agent_option.yaml"), 
                   os.path.join("configs", "iGibson", "agent_rag_interaction.yaml"), 
                   os.path.join("configs", "iGibson", "agent_rag_mask.yaml"), 
                   os.path.join("configs", "iGibson", "agent_rag_option.yaml"), ]
# clean_soak_rag_config = os.path.join("configs", "iGibson", "soak_rag.yaml")
# clean_car_config = os.path.join("configs", "iGibson", "clean_car.yaml")
# clean_rag_config = os.path.join("configs", "iGibson", "clean_rag.yaml")

