const config = require('../config.json')
const tf = require('@tensorflow/tfjs')
const _ = require('lodash')

/**
 * Collaborative Deep Sarsa Agent.
 * Derived from https://github.com/Hulk89/gridworld_tfjs and
 * inspired by https://cs.stanford.edu/people/karpathy/reinforcejs/
 */
class DeepSarsaAgent {

  constructor(width, height, sendPredictionRequest) {
    this.discount_factor = config.agent.discountFactor
    this.learning_rate = config.agent.learningRate

    this.epsilon_decay = config.agent.epsilonDecay
    this.epsilon_min = config.agent.epsilonMin
    this.epsilon = 1.0
    this.width = width
    this.height = height
    this.model = this.make_model(this.learning_rate)
    this.num_frame = 0
    this.sendPredictionRequest = sendPredictionRequest
  }

  make_model(lr) {
    const model = tf.sequential()

    model.add(tf.layers.dense({
      inputShape: [ this.width * this.height ],
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
      units:4,
      kernelInitializer: 'VarianceScaling',
      activation: 'linear'
    }))
    model.compile({
      loss: 'meanSquaredError',
      optimizer: tf.train.adam(lr)
    })

    return model
  }

  makeInput(data) {
    const input = tf.tensor3d(data, [1, this.height, this.width])
    const flatten = input.reshape([1, this.width * this.height])
    input.dispose()
    return flatten
  }

  predict_action(state) {
    const input_data = this.makeInput(state)
    const prediction = this.model.predict(input_data)
    input_data.dispose()
    return prediction
  }

  async get_action(state) {
    if (Math.random() <= this.epsilon) {
      return _.random(0, 3)
    } else {

      // get predictions from cluster workers
      let predictions = (config.workers > 1) ? await this.sendPredictionRequest(state) : []

      // convert array to tensor
      predictions = predictions.map((prediction) => {
        return tf.tensor2d([_.values(prediction)])
      })

      // add local prediction
      predictions.push(this.predict_action(state))

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

      return (await predictions[index].argMax(1).data())[0]
    }
  }

  async train_model(state, action, reward, next_state, next_action, done) {

    if (this.epsilon > this.epsilon_min) {
      this.epsilon *= this.epsilon_decay
    }

    state = this.makeInput(state)
    next_state = this.makeInput(next_state)

    var target = this.model.predict(state)
    var target_val = this.model.predict(next_state)

    var q_res = await target.data()
    var target_reward = (await target_val.data())
    target_reward = target_reward[next_action]
    if (done) {
      q_res[action] = reward
    } else {
      q_res[action] = reward + this.discount_factor * target_reward
    }

    const res = Array.from(q_res)
    // tensor of rewards for four actions: up, down, left, right
    const q = tf.tensor2d(res, [1, 4])

    // train model
    const h = await this.model.fit(state, q, {epoch: 1})

    this.num_frame += 1
    if (this.num_frame % 100 == 0) {
      // console.info(`frame ${this.num_frame}. loss ${h.history.loss[0]}.`)
      console.log('epsilon', this.epsilon)
    }
    state.dispose()
    next_state.dispose()
    target.dispose()
    target_val.dispose()
    q.dispose()

  }

}

module.exports = DeepSarsaAgent
