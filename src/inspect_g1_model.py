import argparse
from pathlib import Path

import mujoco


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    print("\nG1 model loaded successfully.")
    print(f"Bodies: {model.nbody}")
    print(f"Joints: {model.njnt}")
    print(f"Degrees of freedom: {model.nv}")
    print(f"Positions: {model.nq}")
    print(f"Actuators: {model.nu}")
    print(f"Sensors: {model.nsensor}")
    print(f"Geometries: {model.ngeom}")

    mujoco.mj_forward(model, data)

    if args.no_viewer:
        print("\nViewer disabled.")
        return

    print("\nOpening MuJoCo viewer.")

    from mujoco import viewer

    viewer.launch(model, data)


if __name__ == "__main__":
    main()
