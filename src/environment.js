const config = require('../config.json')
const Promise = require('bluebird')
const requestPromise = require('request-promise')
const url = require('url')

class Environment {
  constructor() {
    const options = {
      method: 'POST',
      uri: url.format(config.gym.api),
      body: {
        'env_id': config.environmentId
      },
      json: true
    }
    return (async () => {
      await requestPromise(options).then(body => {
        this.instanceId = body.instance_id
        console.info('this.instanceId', this.instanceId)
      }).catch(console.warn)
      await this.initializeSpace()
      return this
    })()
  }

  /*
   * returns initial observation, Array of Numbers
   * Example:
   * [
   *   0.021447051116053537,
   *   0.024554621758206052,
   *   -0.008282778032650082,
   *   0.0011348577826760242
   * ]
   */
  async initializeSpace() {
    const options = {
      method: 'POST',
      uri: `${url.format(config.gym.api)}${this.instanceId}/reset/`,
      data: {},
      headers: {
        'Content-Type': 'application/json'
      },
      json: true
    }
    const actionSpace = await this.getActionSpace()
    this.actionSpace = actionSpace.n
    return requestPromise(options).then(body => {
      console.log('body', body)
      if ((typeof body.observation) == 'number') {
        this.stateSpace = 1
      } else if (Array.isArray(body.observation)) {
        this.stateSpace = body.observation.length
      } else {
        return Promise.reject({msg: 'unknown state space', observation: body.observation})
      }
      return body.observation
    }).catch(err => {
      console.warn('Environment#initializeSpace failed', err)
    })
  }

  async getActionSpace() {
    const options = {
      method: 'GET',
      uri: `${url.format(config.gym.api)}${this.instanceId}/action_space/`,
      json: true
    }
    return requestPromise(options).then(body => {
      return body.info
    }).catch(err => {
      console.warn('Environment#getActionSpace failed', err)
    })
  }

  async step(action) {
    const options = {
      method: 'POST',
      uri: `${url.format(config.gym.api)}${this.instanceId}/step/`,
      headers: {
        'Content-Type': 'application/json'
      },
      body: {
        action,
        render: config.gym.render
      },
      json: true
    }
    return requestPromise(options).catch(err => {
      console.warn('Environment#step failed', err)
    })
  }
}

module.exports = Environment
