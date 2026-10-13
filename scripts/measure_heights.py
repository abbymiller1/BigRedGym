"""Print the Go2 body height for several leg poses (geometry only, no physics)."""
import mujoco
import gym.envs  # noqa: F401  (registers the tasks)
from gym.utils.task_registry import task_registry

# 1. Build the go2trot environment just to get its MuJoCo robot model.
env_cfg, train_cfg = task_registry.get_cfgs("go2trot")
env_cfg.env.num_envs = 1
env_cfg.seed = 0
train_cfg.seed = 0
task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
env = task_registry.make_env("go2trot", env_cfg, device="cpu",
                             headless=True, backend="mujoco")
model = env._backend._mjm          # the MuJoCo model the env loaded
data = mujoco.MjData(model)        # a fresh "pose" to play with

joint_names = [model.joint(i).name for i in range(model.njnt)]
# The feet are spheres; we use them to find the lowest point of the robot.
feet = [g for g in range(model.ngeom)
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
        and "foot" in model.body(model.geom_bodyid[g]).name]

poses = [(0.96, -1.36),  # original home pose
         (0.30, -0.84), (0.50, -1.00), (0.80, -1.60),
         (1.10, -2.00), (1.30, -2.40), (1.50, -2.70)]

for thigh, calf in poses:
    data.qpos[:] = 0.0
    data.qpos[3] = 1.0                 # body quaternion w=1 -> upright, not rotated
    for j, name in enumerate(joint_names):
        if "thigh" in name:
            data.qpos[model.jnt_qposadr[j]] = thigh
        if "calf" in name:
            data.qpos[model.jnt_qposadr[j]] = calf
    mujoco.mj_kinematics(model, data)  # compute where every body part is
    # lowest foot point = sphere center z minus sphere radius
    lowest = min(data.geom_xpos[g][2] - model.geom_size[g][0] for g in feet)
    height = data.qpos[2] - lowest     # body center z minus lowest foot point
    print(f"thigh={thigh:5.2f}  calf={calf:5.2f}  ->  height={height:.3f} m")