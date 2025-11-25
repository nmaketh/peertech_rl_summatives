project Overview

PeerTech is an advanced Reinforcement Learning Tutor System designed for real-time adaptive teaching. It models a “student” whose internal states evolve based on the difficulty of challenges, subject choice, and peer assistance.

 Mission

Maximise mastery while preventing disengagement and fatigue.
The agent must act intelligently across multiple dimensions of learning to support the virtual student.

 Key Abilities of the Tutor

Adjust challenge difficulty (Easy/Medium/Hard)

Switch between subjects (Math/Physics/ICT)

Select peer match quality (High/Low)

React to fatigue, frustration, and low engagement

Maintain long-term mastery growth

High-Level Architecture
[Agent] → chooses action
   ↓
[Environment] → updates mastery, engagement, fatigue
   ↓
[Reward Function] → teaches agent optimal behaviour
   ↓
[Pygame Renderer] → visualises inner state

3. Environment Design

Environment Design

The environment is implemented in environment/custom_env.py and follows Gymnasium API standards.

 Observation Space (9-dimensional)
Index	Feature	Range	Description
0	mastery_level	[0,1]	Student mastery
1	engagement	[0,1]	Attention & motivation
2	fatigue	[0,1]	Mental tiredness
3	peer_match_quality	[0,1]	Peer compatibility
4	past_success_rate	[0,1]	EMA of success
5	difficulty_norm	[0,1]	Difficulty level normalized
6–8	subject_one_hot	{0,1}	Math, Physics, ICT
Action Space (6 discrete actions)
ID	Action	Description
0	Easy challenge	Good for fatigue recovery
1	Medium challenge	Balanced learning
2	Hard challenge	High reward, high risk
3	Switch subject	Prevents boredom
4	High-compat peer	Boosts engagement
5	Low-compat peer	Risky exploration
 Reward Function (shaped)

+8 × mastery gain

+2 success

−1.5 failure

±0.8 engagement change

penalties for fatigue & low engagement

bonus for phase progression

Termination Conditions

fatigue ≥ 0.97

engagement ≤ 0.08

max steps reached (lesson end)

 
 4. RL Algorithms

Copy-paste into:
wiki › Reinforcement-Learning-Algorithms.md

Reinforcement Learning Algorithms

Four algorithms were implemented:

1️⃣ Deep Q-Networks (DQN) — Best Performer

Off-policy

Highly stable

Best at long-term planning

Highest evaluation reward

2️⃣ A2C (Advantage Actor-Critic)

On-policy

Learns fast

Medium stability

3️⃣ PPO (Proximal Policy Optimization)

Very stable

Smooth learning curves

Second-best performer after DQN

4️⃣ REINFORCE (Policy Gradient)

High variance

Slow learning

Weakest performance

Included for comparison

 Hyperparameter Sweeps

40 total experiments:

10 × DQN

10 × PPO

10 × A2C

10 × REINFORCE

Stored under:

evaluation/sweeps/

📄 5. Visualization & OmniGlass UI


<img width="1239" height="699" alt="Skermskoot 2025-11-25 030734" src="https://github.com/user-attachments/assets/2cdcdc30-fb08-4e2a-8c07-c348ebb91e36" />

Visualization & OmniGlass UI

The simulator uses a professionally designed AAA-style futuristic interface.

 Major Visual Components
✔ Mastery Progress Ring
✔ Reward Collection Outer Ring
✔ Difficulty Nodes Orbit
✔ Subject Strip (Math / Physics / ICT)
✔ Psychology Panel

frustration

confidence

motivation

student profile type

✔ Robot Tutor Avatar

eye glow

confidence brain ring

coloured aura

floating arms

✔ Fatigue Vignette
✔ Glass UI Panels


📄 6. Training & Evaluation



Training & Evaluation

Training scripts located in /training/

Run:

python training/train_dqn.py
python training/train_ppo.py
python training/train_a2c.py
python training/train_reinforce.py

📊 Evaluation Metrics

episode reward

success rate

episode length

reward variance

mastery progression

🏆 Final Results Summary
Model	Avg Reward	Stability	Notes
DQN	10.99	★★★★★	Best
PPO	9.46	★★★★☆	Stable
A2C	9.11	★★★☆☆	Fast learner
REINFORCE	2–5	★☆☆☆☆	Very unstable
📄 7. Demo Script (3 minutes)

video DEMO: https://drive.google.com/file/d/1cOPfG1HzfoXGofp8bihVUQ-TWKeK80Kn/view?usp=sharing



8. Repository Structure

Repository Structure
peertech_rl_summatives/
│
├── environment/
│   ├── custom_env.py
│   ├── rendering.py
│
├── training/
│
├── evaluation/
│   ├── sweeps/
│   ├── plots/
│
├── models/      (optional in .gitignore)
├── main.py
├── README.md
└── requirements.txt

9. Future Work

Copy-paste into:
wiki › Future-Work.md

Future Work

Multi-agent peer collaboration

Curriculum-learning stages

Emotional modelling (stress, curiosity)

Real student data integration

Adaptive text generation feedback

Multi-modal robot communication
