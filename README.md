Collaborative Deep Sarsa Learning
===

A collaborative deep sarsa learning implementation. `n` agents train their sarsa network on the gridworld game. A new action is determined by querying the best reward from `n` agents. Every agent runs in an independent process.

## How to run

    $ yarn install
    $ yarn start

## Configure the algorithm for gridworld

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
    "learningRate": 0.001,
    "epsilonDecay": 0.9998,
    "epsilonMin": 0.01
  }
}
```

## LICENSE

MIT License. Please see the LICENSE file for details.
