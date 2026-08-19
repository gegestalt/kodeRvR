# Reinforcement Learning for an Adaptive IPS

The intrusion detector estimates what traffic represents. The IPS policy solves
a different problem: which defensive action minimizes attack damage without
causing unacceptable service disruption. That second problem is sequential
because a weak response can allow an attack to progress, while an aggressive
response changes later network availability.

## Environment contract

- **State:** detector probability, anomaly score, normalized attack stage,
  compromise estimate, service criticality, recent attack rate, response budget.
- **Actions:** allow, monitor, rate-limit, drop flow, temporarily block source,
  block destination port, isolate host.
- **Transition:** a defensive action changes the probability that an attack is
  contained or progresses toward compromise.
- **Reward:** containment is positive; compromise, false prevention, critical
  service disruption, and time under attack are negative.
- **Safety:** an action mask sits outside the learned policy. The agent cannot
  isolate or port-block a critical service without strong evidence.

## Why Double DQN first

DQN is an interpretable baseline for this small discrete action space. Double
DQN reduces the optimistic value bias caused by using one network both to choose
and evaluate the next action. PPO is the later comparison, after the environment
and reward sensitivity tests establish that the simulator is meaningful.

## Bellman target

For transition `(s, a, r, s', done)`, select the best valid next action with the
online network and evaluate it with the target network:

```text
a* = argmax_valid Q_online(s', a)
target = r                              if done
target = r + gamma * Q_target(s', a*)  otherwise
```

The action mask must be applied before `argmax`. The loss should be Huber
(`SmoothL1Loss`) because it is less sensitive than squared error to occasional
large security penalties.

## Evaluation discipline

Training and evaluation scenario seeds must remain disjoint. Compare against
allow-only, aggressive, and rule-based policies using containment rate,
compromise rate, false prevention, disruptive actions, episode return, and
variance across seeds. A learned policy is not better merely because its reward
is higher under one arbitrary reward configuration; repeat a reward-sensitivity
ablation before making a strong claim.
