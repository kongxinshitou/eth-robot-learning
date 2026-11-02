# ETH Zürich · Robot Learning — 课程作业

课程：[**Robot Learning: From Fundamentals to Foundation Models**](https://cvg.ethz.ch/lectures/Robot-Learning/)（263-5911-00L，Spring 2026，讲师 Oier Mees）
官方代码仓库：[mees-robot-learning-course/ethz-course-2026](https://github.com/mees-robot-learning-course/ethz-course-2026)

本仓库只放**我自己写的作业代码**，不含课程组提供的脚手架、讲义与数据集（版权归课程组所有）。

## 内容

| 路径 | 说明 |
|---|---|
| `hw2_robot_control_mdps/exercises/ex1.py` | HW2 Exercise 1 实现：Bernoulli 双纽线关键点 + 阻尼最小二乘（DLS）逆运动学追踪 |
| `hw2_robot_control_mdps/check_ex1.py` | 我自己写的自检脚本，验证 ex1.py 三部分实现是否正确 |

## 用法

把文件放回课程仓库对应位置即可：

```text
ethz-course-2026/
└── hw2_robot_control_mdps/
    ├── exercises/ex1.py      ← hw2_robot_control_mdps/exercises/ex1.py
    └── check_ex1.py          ← hw2_robot_control_mdps/check_ex1.py
```

然后在 hw2 的虚拟环境中运行（实测 CPU 即可）：

```bat
cd hw2_robot_control_mdps
python check_ex1.py
```

实测输出：16 个关键点全部收敛，平均末端位置误差 1.1 mm，最大 2.8 mm。

```text
  per-keypoint final error (m):
    0.0010 0.0028 0.0010 ... 0.0010
  mean = 0.0011   max = 0.0028   (target: < 0.01)
  ALL CHECKS PASSED
```

## 实现要点

- `get_lemniscate_keypoint`：`y = a·cos t / (1 + sin²t)`，`z = a·cos t·sin t / (1 + sin²t)`，支持标量与向量输入（`np.asarray`）。
- `build_keypoints`：`t = linspace(0, 2π, count, endpoint=False)` 避免首尾重复点，拼成 `(count, 3)` 的 `[x_offset, y, z + z_offset]`。
- `ik_track`：每次迭代 `mj_kinematics` → 位置误差 → `mj_jacSite` 取雅可比 → 解 `(J Jᵀ + λI) x = [pos_gain·e, 0]`，再 `q̇ = Jᵀx`，用 `np.linalg.solve` 而不是显式求逆；`q̇` 夹到 ±2 防止过冲，误差 < 1e-3 提前收敛，结束时恢复原始 `qpos`。

## 进度

- [x] HW1 PyTorch tutorial
- [x] HW2 Exercise 1
- [ ] HW2 Exercise 2+
- [ ] HW3 Imitation Learning
- [ ] HW4 Reinforcement Learning
