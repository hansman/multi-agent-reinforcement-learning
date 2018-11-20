const config = require('../config')
const tf = require('@tensorflow/tfjs')
const Environment = require('./environment')
const _ = require('lodash')

const actionLookup = {
  0: 'left',
  1: 'down',
  2: 'up',
  3: 'right'
}

async function train(agent, width, height, enemies, locations) {
  const env = new Environment(width, height, enemies, locations)
  let data = env.data
  const episodes = config.agent.episodes

  for (let episode = 0; episode < episodes; episode++) {
    env.initializeGame()
    data = env.data
    let state = [ _.cloneDeep(data) ]
    let result, action, actionResult, nextState
    let done = false,
      reward = 0,
      score = 0

    while (!done) {
      // take action
      action = await agent.get_action(state)
      actionResult = env.step(actionLookup[action])
      reward = actionResult.reward
      done = actionResult.done

      // take next action
      nextState = [ _.cloneDeep(actionResult.state) ]
      nextAction = await agent.get_action(nextState)

      await agent.train_model(state, action, reward, nextState, nextAction, done)

      // set new state and score
      state = [ _.cloneDeep(actionResult.state) ]
      score += reward

      await tf.nextFrame()
    }

    console.info(`${episode}th episode scored: ${score.toFixed(3)}`)
  }

}

module.exports = train
