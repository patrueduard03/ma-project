#!/bin/bash
echo "Please select an option:"
echo "1. Quick test"
echo "2. Full experiment"
read -p "Choose [1/2]: " choice

if [ "$choice" = "2" ]; then
    docker-compose build experiment
    docker-compose run --rm experiment
else
    docker-compose build quick-test
    docker-compose run --rm quick-test
fi
