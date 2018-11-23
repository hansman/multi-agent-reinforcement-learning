const config = require('../config')
const Environment = require('./environment')
const fs = require('fs')
const tf = require('@tensorflow/tfjs')
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

  let scores = []

  fs.appendFileSync('./results', `workers: ${config.workers}\n`)

  fs.appendFileSync('./results', `game: ${JSON.stringify(config.game)}\n`)

  for (let episode = 1; episode <= episodes; episode++) {
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
    scores.push(score)

    console.info(`${episode}th episode scored: ${score.toFixed(3)}`)
    if (!(episode % 10)) {
      fs.appendFileSync('./results', `episode,${episode},score,${_.mean(scores.slice(-10))}\n`)
    }
  }


  console.info(`training mean score: ${_.mean(scores)}`)

}

module.exports = train
