mini_variant_specs ={
    "discrete_obs": False,
    "installing_printer": {
        "room_size": 8,
        "max_steps": 150,
        "use_stage_reward": False,
        "random_obj_pose": False,
        "reward_variant": "install_printer",
    },
    "thawing":{
        "room_size": 10,
        "max_steps": 300,
        "use_stage_reward": False,
        "random_obj_pose": False,
        "reward_variant": "thaw_all",
        },
    "cleaning_car": {
        "room_size": 10,
        "max_steps": 300,
        "use_stage_reward": False,
        "random_obj_pose": False,
        "reward_variant": "clean_rag",
    }
}