#!/usr/bin/env bash

ps ux | awk '/derp/ { print $2 }' | xargs kill
