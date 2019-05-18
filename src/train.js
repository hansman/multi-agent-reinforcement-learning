const config = require('../config')
const Environment = require('./environment')
const fs = require('fs')
const tf = require('@tensorflow/tfjs')
const _ = require('lodash')

const setupResultsFile = (agent) => {
  const { environmentId, workers } = config
  const resultsFolder = `./results/${environmentId}`
  if (!fs.existsSync(resultsFolder)) {
    fs.mkdirSync(resultsFolder)
  }
  if (!fs.existsSync(`${resultsFolder}/${workers}`)) {
    fs.mkdirSync(`${resultsFolder}/${workers}`)
  }
  return `${resultsFolder}/${workers}/${agent.id}.csv`
}

async function train(agent) {
  const env = await new Environment()

  const resultsFilename = setupResultsFile(agent)
  fs.appendFileSync(resultsFilename, `actions, states,
    ${env.actionSpace}, ${env.stateSpace}\n`)

  agent.makeModel(env.actionSpace, env.stateSpace)
  const { episodes } = config.agent

  let scores = []

  for (let episode = 1; episode <= episodes; episode++) {
    let observation = await env.initializeSpace()
    let done = false
    let score = 0

    while (!done) {
      // take action
      const action = await agent.getAction(observation)
      const actionResult = await env.step(action)
      const { reward } = actionResult
      const nextObservation = actionResult.observation
      done = actionResult.done

      // take next action
      nextAction = await agent.getAction(nextObservation)
      await agent.trainModel(observation, action, reward, nextObservation, nextAction, done)

      // set new state and score
      observation = nextObservation
      score += reward

      await tf.nextFrame()
    }
    scores.push(score)

    if (!(episode % 10)) {
      fs.appendFileSync(resultsFilename, `${episode},score,${_.mean(scores.slice(-10))}\n`)
    }
  }
}

module.exports = train
