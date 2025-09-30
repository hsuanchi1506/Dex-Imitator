export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
mkdir data/smplx_extra
curl -L https://raw.githubusercontent.com/oakink/OakInk2/dev/asset/smplx_extra/body_upper_idx.pt   -o data/smplx_extra/body_upper_idx.pt
tar -xjf smplx_locked_head.tar.bz2 --no-same-owner --no-same-permissions

mkdir -p object_preview && tar -xf object_preview.tar -C object_preview --strip-components=1 --no-same-owner --no-same-permissions || tar -xf object_preview.tar -C object_preview --no-same-owner --no-same-permissions
mkdir -p program && (tar -xf program.tar -C program --strip-components=1 --no-same-owner --no-same-permissions || tar -xf program.tar -C program --no-same-owner --no-same-permissions) && (tar -xf program_extension.tar -C program --strip-components=1 --no-same-owner --no-same-permissions || tar -xf program_extension.tar -C program --no-same-owner --no-same-permissions)




python - <<'PY'
import os, glob, pathlib

ROOT = "data/OakInk-v2"
COACD = f"{ROOT}/coacd_object_preview/align_ds"

TEMPLATE = """<?xml version="1.0"?>
<robot name="{name}">
  <material name="obj_color">
    <color rgba="1.0 0.423529411765 0.0392156862745 1.0"/>
  </material>
  <link name="base">
    <visual>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="{mesh_basename}" scale="1 1 1"/>
      </geometry>
      <material name="obj_color"/>
    </visual>
    <collision>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="{mesh_basename}" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
</robot>
"""

for src in glob.glob(os.path.join(COACD, "**", "*.*"), recursive=True):
    if not src.lower().endswith((".obj", ".ply")):
        continue
    p = pathlib.Path(src)
    name = p.stem
    urdf_path = p.with_suffix(".urdf")
    xml = TEMPLATE.format(name=name, mesh_basename=p.name)
    urdf_path.write_text(xml)
    print("written:", urdf_path)
PY


# Training
# for Inspire Hand
python main/rl/train.py task=ResDexHand dexhand=inspire side=RH headless=true num_envs=4096 learning_rate=2e-4 test=false randomStateInit=true rh_base_model_checkpoint=assets/imitator_rh_inspire.pth lh_base_model_checkpoint=assets/imitator_lh_inspire.pth dataIndices=[g0] early_stop_epochs=100 actionsMovingAverage=0.4 experiment=cross_g0_inspire

# Evaluation
python main/rl/train.py task=ResDexHand dexhand=inspire side=RH headless=false num_envs=4 learning_rate=2e-4 test=true randomStateInit=true rh_base_model_checkpoint=assets/imitator_rh_inspire.pth lh_base_model_checkpoint=assets/imitator_lh_inspire.pth dataIndices=[g0] actionsMovingAverage=0.4 checkpoint=runs/cross_g0_inspire__08-30-01-30-59/nn/last_cross_g0_inspire_ep_198_rew__229.11__sr_0.9937888383865356_fr_0.0062111797742545605.pth


python main/rl/train.py task=ResDexHand dexhand=inspire side=BiH headless=false num_envs=4096 learning_rate=2e-4 test=false randomStateInit=true dataIndices=[20aed@0] rh_base_model_checkpoint=assets/imitator_rh_inspire.pth lh_base_model_checkpoint=assets/imitator_lh_inspire.pth early_stop_epochs=1000 actionsMovingAverage=0.4 experiment=cross_20aed@0_inspire

python main/rl/train.py task=ResDexHand dexhand=inspire side=BiH headless=false num_envs=4 learning_rate=2e-4 test=true randomStateInit=true dataIndices=[20aed@0] rh_base_model_checkpoint=assets/imitator_rh_inspire.pth lh_base_model_checkpoint=assets/imitator_lh_inspire.pth actionsMovingAverage=0.4 checkpoint=runs/cross_20aed@0_inspire__08-31-08-25-38/nn/cross_20aed@0_inspire.pth


data/OakInk-v2/data/scene_03__A004++seq__751fb66f8f9b436a99b4__2023-04-26-19-49-07.tar

# for Inspire Hand
python main/dataset/mano2dexhand.py --data_idx 20aed@0 --side right --dexhand inspire --headless --iter 7000
python main/dataset/mano2dexhand.py --data_idx 20aed@0 --side left --dexhand inspire --headless --iter 7000
# for other hands, just replace `inspire` with the corresponding hand name

python main/rl/train.py task=ResDexHand dexhand=inspire side=BiH headless=false num_envs=16 learning_rate=2e-4 test=false randomStateInit=true dataIndices=[20aed@0] rh_base_model_checkpoint=assets/imitator_rh_inspire.pth lh_base_model_checkpoint=assets/imitator_lh_inspire.pth early_stop_epochs=1000 actionsMovingAverage=0.4 experiment=cross_20aed@0_inspire


# train with render
python main/rl/train.py task=ResDexHand dexhand=inspire side=RH headless=false num_envs=4096 learning_rate=2e-4 test=false randomStateInit=true rh_base_model_checkpoint=assets/imitator_rh_inspire.pth lh_base_model_checkpoint=assets/imitator_lh_inspire.pth dataIndices=[g0] early_stop_epochs=100 actionsMovingAverage=0.4 experiment=cross_g0_inspire

