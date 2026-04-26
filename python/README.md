# 🎛️ PID Simulator

Simple PID controller simulator with a graphical interface in Python.

---

## Features

- Closed-loop PID simulation
- Second-order plant
- 3 real-time plots:
  - Step response
  - Control signal
  - P, I, D terms
- Basic metrics:
  - Overshoot
  - Rise time
  - Settling time
  - Steady-state error

---

## Run

```bash
pip install matplotlib numpy
python pid_simulator.py

| Name      | Meaning           |
| --------- | ----------------- |
| Kp        | Proportional gain |
| Ki        | Integral gain     |
| Kd        | Derivative gain   |
| Time      | Simulation time   |
| Reference | Setpoint          |
