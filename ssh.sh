#!/bin/bash

# SSH Setup Script - Creates SSH config and keys
# This script helps set up your SSH environment on Linux/Unix systems

echo "SSH Setup Script"
echo "================"
echo

# Define function at the beginning
add_host_to_config() {
    echo
    echo "Enter hostname or alias for this connection (e.g., myserver):"
    read HOST_ALIAS
    
    echo "Enter the actual hostname or IP address:"
    read HOST_ADDRESS
    
    echo "Enter the username for this connection:"
    read HOST_USER
    
    echo "Enter the SSH port (default: 22):"
    read HOST_PORT
    if [ -z "$HOST_PORT" ]; then 
        HOST_PORT=22
    fi
    
    echo "Enter the identity file path (or leave blank for default):"
    read IDENTITY_FILE
    
    echo
    echo "# Host entry added on $(date)" >> "$CONFIG_FILE"
    echo "Host $HOST_ALIAS" >> "$CONFIG_FILE"
    echo "    HostName $HOST_ADDRESS" >> "$CONFIG_FILE"
    echo "    User $HOST_USER" >> "$CONFIG_FILE"
    echo "    Port $HOST_PORT" >> "$CONFIG_FILE"
    
    if [ ! -z "$IDENTITY_FILE" ]; then
        echo "    IdentityFile $IDENTITY_FILE" >> "$CONFIG_FILE"
    fi
    
    echo "    ServerAliveInterval 60" >> "$CONFIG_FILE"
    echo >> "$CONFIG_FILE"
    
    echo "Host entry added to config file."
}

# Check if SSH is installed
if ! command -v ssh &> /dev/null; then
    echo "ERROR: SSH is not installed on your system."
    echo "Please install OpenSSH using your package manager."
    echo "For example: sudo apt-get install openssh-client"
    exit 1
fi

# Set SSH directory
SSH_DIR="$HOME/.ssh"

# Create SSH directory if it doesn't exist
if [ ! -d "$SSH_DIR" ]; then
    echo "Creating SSH directory at $SSH_DIR..."
    mkdir -p "$SSH_DIR"
    chmod 700 "$SSH_DIR"
    echo "SSH directory created."
else
    echo "SSH directory already exists at $SSH_DIR."
fi

# Check if config file exists
CONFIG_FILE="$SSH_DIR/config"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Creating new SSH config file..."
    
    echo "# SSH Configuration File" > "$CONFIG_FILE"
    echo "# Created on $(date)" >> "$CONFIG_FILE"
    echo >> "$CONFIG_FILE"
    
    # Get user input for a host entry
    echo "Do you want to add a host entry to the config file? (y/n)"
    read ADD_HOST
    
    if [[ "$ADD_HOST" =~ ^[Yy]$ ]]; then
        add_host_to_config
    fi
    
    chmod 600 "$CONFIG_FILE"
    echo "SSH config file created at $CONFIG_FILE."
else
    echo "SSH config file already exists at $CONFIG_FILE."
    echo "Do you want to add a new host entry? (y/n)"
    read ADD_HOST
    
    if [[ "$ADD_HOST" =~ ^[Yy]$ ]]; then
        add_host_to_config
    fi
fi

# Check for SSH key generation
echo
echo "Do you want to generate a new SSH key pair? (y/n)"
read GEN_KEY

if [[ "$GEN_KEY" =~ ^[Yy]$ ]]; then
    # Ask for key type
    echo
    echo "Select SSH key type:"
    echo "1. RSA (2048 bits)"
    echo "2. RSA (4096 bits)"
    echo "3. Ed25519 (more secure, recommended)"
    read KEY_TYPE
    
    # Ask for key name
    echo
    echo "Enter a name for your key (default: id_rsa or id_ed25519):"
    read KEY_NAME
    
    if [ -z "$KEY_NAME" ]; then
        if [ "$KEY_TYPE" == "3" ]; then
            KEY_NAME="id_ed25519"
        else
            KEY_NAME="id_rsa"
        fi
    fi
    
    # Generate key based on type
    echo
    echo "Generating SSH key pair..."
    
    if [ "$KEY_TYPE" == "1" ]; then
        ssh-keygen -t rsa -b 2048 -f "$SSH_DIR/$KEY_NAME" -C "$KEY_NAME created on $(date)"
    elif [ "$KEY_TYPE" == "2" ]; then
        ssh-keygen -t rsa -b 4096 -f "$SSH_DIR/$KEY_NAME" -C "$KEY_NAME created on $(date)"
    elif [ "$KEY_TYPE" == "3" ]; then
        ssh-keygen -t ed25519 -f "$SSH_DIR/$KEY_NAME" -C "$KEY_NAME created on $(date)"
    else
        echo "Invalid key type selected. Using Ed25519 by default."
        ssh-keygen -t ed25519 -f "$SSH_DIR/$KEY_NAME" -C "$KEY_NAME created on $(date)"
    fi
    
    echo "SSH key pair generated:"
    echo "- Private key: $SSH_DIR/$KEY_NAME"
    echo "- Public key: $SSH_DIR/$KEY_NAME.pub"
    
    # Display the public key
    echo
    echo "Your public key is:"
    echo "--------------------------------------------------"
    cat "$SSH_DIR/$KEY_NAME.pub"
    echo "--------------------------------------------------"
    echo
    echo "Copy this public key to any servers you want to connect to."
    echo "You can use ssh-copy-id to copy it to a remote server:"
    echo "ssh-copy-id -i $SSH_DIR/$KEY_NAME.pub username@remoteserver"
fi

# Create known_hosts file if it doesn't exist
KNOWN_HOSTS="$SSH_DIR/known_hosts"
if [ ! -f "$KNOWN_HOSTS" ]; then
    echo "Creating empty known_hosts file..."
    echo "# Known hosts file" > "$KNOWN_HOSTS"
    echo "# Created on $(date)" >> "$KNOWN_HOSTS"
    chmod 600 "$KNOWN_HOSTS"
    echo "known_hosts file created."
fi

echo
echo "SSH setup complete!"
echo "Your SSH directory is located at: $SSH_DIR"
echo
echo "You can now use SSH to connect to remote servers."
echo "To connect, use: ssh username@hostname"
echo