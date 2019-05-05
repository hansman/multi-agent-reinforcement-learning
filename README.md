Multi-Agent Deep Reinforcement Learning
===

An implementation of Multi-Agent Reinforcement Learning with Deep Sarsa Agents using [tensorflow-js](https://github.com/tensorflow/tfjs). *n* agents train each a sarsa policy network on gridworld, a game similar to the gym environment in [gym-gridworld](https://github.com/maximecb/gym-minigrid#included-environments).

Intuition: Distributing exploratory moves over n agents speeds up game exploration and overall convergence.

The algorithm presented in this codebase obtains an action by predictions from *n* simultaneously trained sarsa policy networks. On startup a master process spawns *n* worker processes, for each agent one. Workers communicate with each other via IPC messages using [node cluster](https://nodejs.org/api/cluster.html). To obtain an action, agent *A* broadcasts a prediction request to *n-1* agents and generates one prediction from its own sarsa network model. Each worker is running the same agent, trains on the same game and sends/responds to prediction requests. When all agents responded with their prediction, the action with the largest value (Q_value) from all *n* predictions is obtained. During network training a factor *epsilon* between *[epsilon_min,1]* determines the fraction of times the policy action is picked versus a random action is obtained to transfer to the next state. *Epsilon* decays with increasing learning episodes. Each policy network has *4* hidden layers activated with *ReLU*.

Multiple collaborating agents require less episodes to solve gridworld. I think of it like a team of people that initially explore a maze individually by picking random routes. As they build up knowledge they take turns on intersections based on their collectively learned experience and get to the end quicker than they would all by themselves.

![Alt text](results/score-n-1.png?raw=true "Score over episodes n-1")
Score over episodes - each agent shares experience with 2 adjacent neighbors

![Alt text](results/score-n-n.png?raw=true "Score over episodes n-n")
Score over episodes - each agent shares experience with all other agents

## How to run
    $ yarn install
    $ yarn start

    # optional, to generate a new game
    $ yarn generate-game

## Configure the game, cluster and agents
./config.json
```json
{
  "workers": 8,
  "neighboringGradient": true,
  "agent": {
    "discountFactor": 0.9,
    "episodes": 2000,
    "learningRate": 0.001,
    "epsilonDecay": 0.9998,
    "epsilonMin": 0.01
  },
  "game": {
    "width": 13,
    "height": 13,
    "enemies": 3
  }
}
```

## LICENSE

MIT License. Please see the LICENSE file for details.
