#!/usr/bin/env node

const Agent           = require('../src/agent')
const cluster         = require('cluster')
const config          = require('../config.json')
const fs              = require('fs')
const Promise         = require('bluebird');
const train           = require('../src/train')
const uuid            = require('uuid')
const _               = require('lodash')


// use tensorflow c++ bindings
// require('@tensorflow/tfjs-node')

const listeners = {} // broadcast response listeners
const callbacks = {} // final request callbacks


const getRandomSubarray = (arr, size) => {
  var shuffled = arr.slice(0), i = arr.length, temp, index
  while (i--) {
    index = Math.floor((i + 1) * Math.random())
    temp = shuffled[index]
    shuffled[index] = shuffled[i]
    shuffled[i] = temp
  }
  return shuffled.slice(0, size)
}

const getLocations = (num, width, height) => {
  var spaces = []
  for (let i = 0; i < width; i++) {
    for (let j = 0; j < height; j++) {
      spaces.push([i, j])
    }
  }
  return getRandomSubarray(spaces, num)
}


if (cluster.isMaster) {
  /**
   * Master process. Setup message handler and fork workers.
   */
  console.info(`Master ${process.pid} is running`)

  const { enemies, width, height } = config.game

  // Generate new game (optional)
  config.game.locs = getLocations(2 + enemies, width, height)
  fs.writeFileSync(`${__dirname}/../config.json`, JSON.stringify(config, undefined, 2))

  // Setup message handler
  const messageHandler = (senderId, msg) => {
    const { type, id, re, data } = msg

    if (re && listeners[re]) {
      /*
       * This is a response to a message. Call the response handler.
       */
      listeners[re].handler(data)
      return
    }

    if (type == 'predictionRequest') {
      /*
       * A worker is requesting a prediction from peers.
       * Setup final callback, response listeners and broadcast
       * request to all other workers.
       */

      // setup callback
      callbacks[id] = (data) => {
        cluster.workers[senderId].send({id: uuid.v4(), re: id, type: 'predictionResponse', data})
      }

      // setup listener
      if (!listeners[id]) {

        const handler = (data) => {
          if (!listeners[id]) {
            console.warn('response on outdated message', id)
            return
          }
          let { dataObjects } = listeners[id]
          dataObjects.push(data)
          if (dataObjects.length == (Object.keys(cluster.workers).length - 1)) {
            callbacks[id](dataObjects)
          }
        }

        let dataObjects = []
        listeners[msg.id] = { handler, dataObjects }

        // remove listener and callback after a timeout
        setTimeout(() => {
          delete listeners[msg.id]
          delete callbacks[msg.id]
        }, 20e3)

      }

      // broadcast request to workers
      _.forEach(cluster.workers, worker => {
        if (worker.id != senderId) {
          worker.send({type, data, id})
        }
      })

    } else {
      console.warn('unrecognized message type', type)
    }

  }

  // Start workers and listen for messages.
  for (let i = 0; i < config.workers; i++) {
    let worker = cluster.fork()
    worker.on('message', messageHandler.bind(this, worker.id))
  }

  cluster.on('death', (worker) => {
    console.warn('Worker ' + worker.pid + ' died.')
  })

} else {
  console.info(`Worker ${process.pid} is running`)
  /**
   * Worker process. Setup message handlers,
   * create rl agent and train the model.
   */
  const { width, height, enemies } = config.game

  // create rl agent

  /*
   * sendPredictionRequest
   * Sends a prediction request from worker to master via ipc channel
   */
  const sendPredictionRequest = Promise.promisify((state, cb) => {
    const id = uuid.v4()
    callbacks[id] = cb
    process.send({type: 'predictionRequest', id, data: state})
    setTimeout(() => {
      delete callbacks[id]
    }, 20e3)
  })
  const agent = new Agent(width, height, sendPredictionRequest)


  // handle messages from master
  process.on('message', (msg) => {
    const { type, id, re, data } = msg

    if (type == 'predictionRequest') {
      /**
       * A peer agent is requesting a prediction for a given state.
       * Get next action prediction from current model and respond.
       */
      agent.predict_action(data).data().then((actions) => {
        process.send({type: type + '-re', id: uuid.v4(), re: id, data: actions})
      });
    } else if (type == 'predictionResponse') {
      if (callbacks[re]) {
        callbacks[re](null, data)
      }
    } else {
      console.warn('unrecognized message type', type)
    }

  })

  const locs = config.game.locs
  train(agent, width, height, enemies, locs)

  //viewModel(agent, locs)

}
