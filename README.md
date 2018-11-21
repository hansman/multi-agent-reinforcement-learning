Collaborative Deep Reinforcement Learning
===

An implementation of Collaborative Reinforcement Learning with Deep Sarsa Agents using ([tensorflow-js](https://github.com/tensorflow/tfjs)). *n* agents train each a Sarsa Policy Network on gridworld, a game similar to [gym-gridworld](https://github.com/maximecb/gym-minigrid).

The algorithm presented in this codebase obtains an action by predictions from *n* simultaneously trained sarsa policy networks. On startup a master process spawns *n* worker processes, for each agent one. Workers communicate with each other via Inter Process Communication (IPC) using [node cluster](https://nodejs.org/api/cluster.html). When a new action is to be predicted, agent *A* broadcasts a prediction request to *n-1* agents and generates one prediction from its own sarsa network model. Each worker is running the same agent, trains on the same game and sends/responds to prediction requests. When all agents responded with their prediction, the action with the largest value (Q_value) from all *n* predictions is obtained. During network training a factor *epsilon* between *[epsilon_min,1]* determines the fraction of times the policy action is picked versus a random action is obtained to transfer to the next state. Each policy network has *4* hidden layers activated with ReLU.

## How to run

    $ yarn install
    $ yarn start

## Configure the game, cluster and agents

```
{
  "workers": 4,
  "game": {
    "width": 20,
    "height": 20,
    "enemies": 10
  },
  "agent": {
    "discountFactor": 0.9,
    "episodes": 5000,
    "learningRate": 0.001,
    "epsilonDecay": 0.9998,
    "epsilonMin": 0.01
  }
}
```

## LICENSE

MIT License. Please see the LICENSE file for details.
