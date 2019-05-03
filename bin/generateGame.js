#!/usr/bin/env node

const config = require('../config')
const fs = require('fs')

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

// Generate new game (optional)
const { enemies, width, height } = config.game
game = getLocations(2 + enemies, width, height)
fs.writeFileSync(`${__dirname}/../game.json`, JSON.stringify(game, undefined, 2))
