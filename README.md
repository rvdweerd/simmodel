# simmodel
Simulation environment for Reinforcement Learning experiments in Search & Pursuit on Graphs.
Thesis work-in-progress for UvA/AI MSc program (![Thesis Proposal](modules/sim/Thesis_proposal.pdf)

* Goal: predicting escape routes in a passive search scenario with partial observability</br></br>
![escape_demo](modules/sim/escape_route.gif)

* Demo: a Graph Neural Net based policy model, trained using PPO with invalid action masking, can generalize and be applied to unseen graphs</br></br>
![ppo_demo](modules/sim/PPO_best_metro-evade-demo_right-1.png)

* Demo: escape agent traverses from Dam Square (Amsterdam) to a target node, while avoiding pursuers that move to observation positions. Escape behavior is based on graph representation learning on smaller graphs, combined with reinforcement learning using PPO</br></br>
![escape_demo](modules/sim/final3.png)



