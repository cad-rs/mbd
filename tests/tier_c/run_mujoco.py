"""Tier C reference runner: simulate every exported MJCF case with MuJoCo
and write tests/tier_c/<case>.json (body-frame poses and energy per output
time). Usage: run_mujoco.py <export dir> <output dir>."""

import json
import pathlib
import sys

import mujoco


def run(xml: pathlib.Path, end: float, hout: float) -> dict:
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    steps = round(hout / model.opt.timestep)
    count = round(end / hout)
    names = [model.body(i).name for i in range(1, model.nbody)]
    bodies = {n: [] for n in names}
    times, energy = [], []

    def record(t: float) -> None:
        mujoco.mj_forward(model, data)
        times.append(t)
        for n in names:
            b = data.body(n)
            bodies[n].append([float(x) for x in list(b.xpos) + list(b.xquat)])
        energy.append([float(data.energy[0]), float(data.energy[1])])

    record(0.0)
    for k in range(1, count + 1):
        for _ in range(steps):
            mujoco.mj_step(model, data)
        record(k * hout)
    integrator = {0: "Euler", 1: "RK4", 2: "implicit", 3: "implicitfast"}.get(int(model.opt.integrator), "?")
    return {
        "source": f"MuJoCo {mujoco.__version__}, {integrator}, dt = {model.opt.timestep}",
        "times": times,
        "bodies": bodies,
        "energy": energy,
    }


def main() -> None:
    src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    for line in (src / "cases.txt").read_text().splitlines():
        name, end, hout = line.split()
        ref = run(src / f"{name}.xml", float(end), float(hout))
        (dst / f"{name}.json").write_text(json.dumps(ref) + "\n")
        print(f"wrote {dst / (name + '.json')} ({len(ref['times'])} frames)")


if __name__ == "__main__":
    main()
