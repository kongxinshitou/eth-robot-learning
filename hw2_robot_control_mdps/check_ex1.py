"""Self-check for HW2 Exercise 1 (Lemniscate keypoints + Damped Least Squares IK).

Run from hw2_robot_control_mdps with the hw2 venv active:

    python check_ex1.py

It only *tests* your implementation in exercises/ex1.py - it does not solve it.
"""
import numpy as np
import mujoco

from scripts import TORQUE_CTRL_XML_PATH
from exercises.ex1 import get_lemniscate_keypoint, build_keypoints, ik_track

A = 0.2
WIDTH = 0.25
X_OFF = 0.3
Z_OFF = 0.25


def implicit_residual(y, z, a):
    """Lemniscate of Bernoulli for the parametrisation used here:
    (y^2 + z^2)^2 == a^2 * (y^2 - z^2).
    """
    return (y ** 2 + z ** 2) ** 2 - a ** 2 * (y ** 2 - z ** 2)


def section(title):
    print("\n" + "-" * 62)
    print(title)
    print("-" * 62)


def main():
    ok = True

    # ------------------------------------------------------------------ part 1
    section("1) get_lemniscate_keypoint(t, a)")
    try:
        y0, z0 = get_lemniscate_keypoint(0.0, a=A)
        y1, z1 = get_lemniscate_keypoint(np.pi / 2, a=A)
        t = np.linspace(0.0, 2 * np.pi, 64, endpoint=False)
        y, z = get_lemniscate_keypoint(t, a=A)
        y, z = np.asarray(y, dtype=float), np.asarray(z, dtype=float)

        print(f"  scalar input -> y={float(y0):+.4f} z={float(z0):+.4f} (expect +a, 0)")
        print(f"  t=pi/2       -> y={float(y1):+.4f} z={float(z1):+.4f} (expect 0, 0)")
        print(f"  vector input -> shapes y={y.shape} z={z.shape} (expect (64,))")

        res = np.abs(implicit_residual(y, z, A)).max()
        print(f"  max |implicit equation residual| = {res:.2e}  (expect < 1e-9)")

        checks = [
            ("t=0 gives (a, 0)", abs(float(y0) - A) < 1e-9 and abs(float(z0)) < 1e-9),
            ("t=pi/2 gives (0, 0)", abs(float(y1)) < 1e-9 and abs(float(z1)) < 1e-9),
            ("vectorised", y.shape == (64,) and z.shape == (64,)),
            ("stays on the curve", res < 1e-9),
        ]
    except NotImplementedError:
        print("  !! get_lemniscate_keypoint still raises NotImplementedError")
        checks, ok = [], False
    for name, good in checks:
        print(f"  [{'PASS' if good else 'FAIL'}] {name}")
        ok &= good

    # ------------------------------------------------------------------ part 2
    section("2) build_keypoints(count=16, width=0.25, x_offset=0.3, z_offset=0.25)")
    kp = None
    try:
        kp = np.asarray(build_keypoints())
        print(f"  shape = {kp.shape}  (expect (16, 3))")
        print(f"  x column: unique values = {np.unique(np.round(kp[:, 0], 9))}  (expect [0.3])")
        print(f"  y range = [{kp[:, 1].min():+.3f}, {kp[:, 1].max():+.3f}]"
              f"   z range = [{kp[:, 2].min():+.3f}, {kp[:, 2].max():+.3f}]")
        res = np.abs(implicit_residual(kp[:, 1], kp[:, 2] - Z_OFF, WIDTH)).max()
        print(f"  max |curve residual| = {res:.2e}  (expect < 1e-9)")
        checks = [
            ("shape is (16, 3)", kp.shape == (16, 3)),
            ("x is fixed at x_offset", np.allclose(kp[:, 0], X_OFF)),
            ("z shifted by z_offset", res < 1e-9),
            ("no duplicated first/last point",
             not np.allclose(kp[0], kp[-1])),
        ]
    except NotImplementedError:
        print("  !! build_keypoints still raises NotImplementedError")
        checks = []
    for name, good in checks:
        print(f"  [{'PASS' if good else 'FAIL'}] {name}")
        ok &= good

    # ------------------------------------------------------------------ part 3
    section("3) ik_track(...) on every keypoint")
    if kp is None:
        print("  skipped - build_keypoints did not return keypoints")
        checks = []
        for name, good in checks:
            print(f"  [{'PASS' if good else 'FAIL'}] {name}")
        section("RESULT")
        print("  Some checks failed (or are not implemented yet) - see above.")
        return 1
    try:
        model = mujoco.MjModel.from_xml_path(str(TORQUE_CTRL_XML_PATH))
        data = mujoco.MjData(model)
        site = "ee_site"

        lo, hi = model.jnt_range[:, 0], model.jnt_range[:, 1]
        errs, qs = [], []
        for target in kp:
            q = np.asarray(ik_track(model, data, site, target), dtype=float)
            qs.append(q)
            data.qpos[:] = q
            mujoco.mj_kinematics(model, data)
            errs.append(float(np.linalg.norm(data.site(site).xpos - target)))
        errs = np.array(errs)
        print("  per-keypoint final error (m):")
        print("   ", " ".join(f"{e:.4f}" for e in errs))
        print(f"  mean = {errs.mean():.4f}   max = {errs.max():.4f}   (target: < 0.01)")

        q_bad = sum(int(((q < lo - 1e-6) | (q > hi + 1e-6)).any()) for q in qs)
        print(f"  keypoints whose IK solution exceeds joint limits: {q_bad} / {len(kp)}")
        if q_bad:
            print("  NOTE: plain DLS-IK ignores joint limits. Clipping qpos to")
            print("        model.jnt_range each iteration removes the violations but")
            print("        costs accuracy - a good point for theoretical question 4.")

        checks = [
            ("all keypoints reached within 1 cm", errs.max() < 0.01),
        ]
    except NotImplementedError:
        print("  !! ik_track still raises NotImplementedError")
        checks = []
    for name, good in checks:
        print(f"  [{'PASS' if good else 'FAIL'}] {name}")
        ok &= good

    section("RESULT")
    print("  ALL CHECKS PASSED - you can move on to exercise 2." if ok
          else "  Some checks failed (or are not implemented yet) - see above.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
