# 强化学习 & GAN 学习笔记

本仓库存放基于李宏毅 RL 课程整理的学习笔记（Markdown 版），由 Obsidian 导出并移除了 Obsidian 专属语法，可在任何标准 Markdown 阅读器中阅读。

## 目录结构

```
reinforcement-learning-notes/
├── GAN/
│   ├── GAN.md
│   └── 附件/            # GAN 笔记图片
└── 强化学习/
    ├── RL Lecture.md            # 强化学习基础（Policy Gradient / Value Function）
    ├── DQN.md                   # DQN：状态-动作价值函数
    ├── PPO.md                   # PPO：近端策略优化
    ├── GAE.md                   # 广义优势估计
    ├── Q函数(From GPT).md       # Q 函数与 Q-Learning
    ├── Double DQN(From GPT).md  # Double DQN
    ├── RL ON LLMs(From GPT)/    # LLM 后训练中的强化学习
    │   ├── RL ON LLMs.md
    │   ├── GRPO.md
    │   ├── Reward Model.md
    │   └── Value Model 与 Reward Model 区别.md
    └── 附件/            # 强化学习笔记图片
```

## 阅读建议

建议按以下顺序阅读：

1. [RL Lecture](强化学习/RL%20Lecture.md) — 强化学习基础
2. [DQN](强化学习/DQN.md) — 状态-动作价值函数
3. [PPO](强化学习/PPO.md) — 近端策略优化
4. [GAE](强化学习/GAE.md) — 广义优势估计
5. [RL ON LLMs](强化学习/RL%20ON%20LLMs(From%20GPT)/RL%20ON%20LLMs.md) — LLM 后训练
6. [GRPO](强化学习/RL%20ON%20LLMs(From%20GPT)/GRPO.md) — 组相对策略优化

数学公式为标准 LaTeX（`$...$` / `$$...$$`），GitHub 原生支持渲染。