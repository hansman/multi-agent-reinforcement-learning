#!/usr/bin/env node

const Agent = require('../src/agent')
const cluster = require('cluster')
const config = require('../config.json')
const game = require('../game.json')
const Promise = require('bluebird');
const train = require('../src/train')
const uuid = require('uuid')
const _ = require('lodash')

// use tensorflow c++ bindings
require('@tensorflow/tfjs-node')

const listeners = {} // broadcast response listeners
const callbacks = {} // final request callbacks

if (cluster.isMaster) {
  /**
   * Master process. Setup message handler and fork workers.
   */
  console.info(`Master ${process.pid} is running`)

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

      // setup callback for accumulated prediction response
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
          if (config.neighboringGradient && (dataObjects.length == 2)) {
            callbacks[id](dataObjects)
          } else if (dataObjects.length == (Object.keys(cluster.workers).length - 1)) {
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

      if (!config.neighboringGradient) {
        // broadcast request to workers
        _.forEach(cluster.workers, worker => {
          if (worker.id != senderId) {
            worker.send({type, data, id})
          }
        })
      } else {
        // send request to neighboring workers only
        const workers = _.values(cluster.workers)
        for (let i = 0; i < workers.length; i++) {
          const worker = workers[i]
          if (worker.id == senderId) {
            const leftNeighbor = workers[i ? (i - 1) : (workers.length - 1)]
            leftNeighbor.send({ type, data, id })
            const rightNeighbor = workers[(i + 1) == workers.length ? 0 : (i + 1)]
            rightNeighbor.send({ type, data, id })
            break;
          }
        }
      }
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

  // create rl agent
  const agent = new Agent(width, height, sendPredictionRequest)

  // handle messages from master
  process.on('message', (msg) => {
    const { type, id, re, data } = msg

    if (type == 'predictionRequest') {
      /**
       * A peer agent is requesting a prediction for a given state.
       * Get next action prediction from current model and respond.
       */
      const prediction = agent.predict_action(data)
      prediction.data().then((actions) => {
        process.send({type: type + '-re', id: uuid.v4(), re: id, data: actions})
        prediction.dispose()
      });
    } else if (type == 'predictionResponse') {
      if (callbacks[re]) {
        callbacks[re](null, data)
      }
    } else {
      console.warn('unrecognized message type', type)
    }
  })

  train(agent, width, height, enemies, game)
}
