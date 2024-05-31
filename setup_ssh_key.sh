#!/usr/bin/env bash

eval "$(ssh-agent -s)"

# Generate a new SSH key
ssh-keygen -t ed25519 -C "lukasbraach@gmail.com" -N "" -f /root/.ssh/id_ed25519

# Add your SSH private key to the ssh-agent
ssh-add /root/.ssh/id_ed25519