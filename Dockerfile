FROM rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.7.1

# Install system dependencies
RUN apt-get update && apt-get install -y git curl sudo

# Create a non-root user with the same UID and GID as the host user
ARG USER_ID
ARG GROUP_ID
RUN groupadd -g $GROUP_ID loki && \
    useradd -u $USER_ID -g $GROUP_ID -s /bin/bash -m loki && \
    usermod -aG sudo loki && \
    echo "loki ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to the non-root user
USER loki
WORKDIR /home/loki

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/loki/.local/bin:${PATH}"

# Create a working directory
WORKDIR /home/loki/app

# Copy the project files
COPY --chown=loki:loki . .

# Install Python dependencies
RUN uv pip install -e .[dev]

# Set the entrypoint to the speedrun script
ENTRYPOINT ["bash", "speedrun.sh"]