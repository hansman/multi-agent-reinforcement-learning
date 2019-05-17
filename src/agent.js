const config = require('../config.json')
const tf = require('@tensorflow/tfjs')
const uuid = require('uuid')
const _ = require('lodash')

/**
 * Collaborative Deep Sarsa Agent.
 * Derived from https://github.com/Hulk89/gridworld_tfjs and
 * inspired by https://cs.stanford.edu/people/karpathy/reinforcejs/
 */
class DeepSarsaAgent {

  constructor(sendPredictionRequest) {
    this.id = uuid.v4()
    this.discountFactor = config.agent.discountFactor
    this.learningRate = config.agent.learningRate

    this.epsilonDecay = config.agent.epsilonDecay
    this.epsilonMin = config.agent.epsilonMin
    this.epsilon = 1.0
    this.numFrame = 0
    this.sendPredictionRequest = sendPredictionRequest
  }

  makeModel(actionSpace = 4, stateSpace) {

    console.log('actionSpace', actionSpace)

    this.actionSpace = actionSpace
    this.stateSpace = stateSpace
    const model = tf.sequential()

    model.add(tf.layers.dense({
      inputShape: [ stateSpace ],
      units: 128,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }))
    model.add(tf.layers.dense({
      units: 128,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }))
    model.add(tf.layers.dense({
      units: 128,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }))
    model.add(tf.layers.dense({
      units: actionSpace,
      kernelInitializer: 'VarianceScaling',
      activation: 'linear'
    }))
    model.compile({
      loss: 'meanSquaredError',
      optimizer: tf.train.adam(this.learningRate)
    })

    this.model = model
  }

  makeInput(data) {
    // is scalar?
    if (typeof data == 'number') {
      data = [ data ]
    }
    // is 2-d?
    if (data[0].length) {
      const input = tf.tensor3d(data, [1, this.height, this.width])
      const flatten = input.reshape([1, this.width * this.height])
      input.dispose()
      return flatten
    } else {
      return tf.tensor2d(data, [1, data.length])
    }
  }

  predictAction(state) {
    const inputData = this.makeInput(state)
    const prediction = this.model.predict(inputData)
    inputData.dispose()
    return prediction
  }

  async getAction(state) {
    if (Math.random() <= this.epsilon) {
      // explorative step
      return _.random(0, this.actionSpace - 1)
    } else {
      // exploitive step
      // get predictions from cluster workers
      let predictions = (config.workers > 1) ? await this.sendPredictionRequest(state) : []

      // convert array to tensor
      predictions = predictions.map(prediction => {
        return tf.tensor2d([_.values(prediction)])
      })

      // add local prediction
      predictions.push(this.predictAction(state))

      // parse best action from predictions
      const results = await Promise.all(predictions.map((prediction) => {return prediction.max(1).data()}))
      let max = results[0][0]
      let index = 0

      for (let i = 0; i < results.length; i++) {
        if (results[i][0] > max) {
          max = results[i][0]
          index = i
        }
      }

      let action = (await predictions[index].argMax(1).data())[0]
      predictions.forEach(p => p.dispose())
      return action
    }
  }

  async trainModel(state, action, reward, nextState, nextAction, done) {
    if (this.epsilon > this.epsilonMin) {
      this.epsilon *= this.epsilonDecay
    }

    state = this.makeInput(state)
    nextState = this.makeInput(nextState)

    var target = this.model.predict(state)
    var targetVal = this.model.predict(nextState)

    var qRes = await target.data()
    var targetReward = (await targetVal.data())
    targetReward = targetReward[nextAction]
    if (done) {
      qRes[action] = reward
    } else {
      qRes[action] = reward + this.discountFactor * targetReward
    }

    const res = Array.from(qRes)
    // tensor of rewards for four actions: up, down, left, right
    const q = tf.tensor2d(res, [1, this.actionSpace])

    // train model
    const h = await this.model.fit(state, q, {epoch: 1})

    this.numFrame += 1
    if (this.numFrame % 100 == 0) {
      // console.info(`frame ${this.num_frame}. loss ${h.history.loss[0]}.`)
      console.log('epsilon', this.epsilon)
    }
    state.dispose()
    nextState.dispose()
    target.dispose()
    targetVal.dispose()
    q.dispose()
  }
}

module.exports = DeepSarsaAgent
