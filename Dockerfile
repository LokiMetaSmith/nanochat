# Use the official ROCm PyTorch image as the base
FROM rocm/pytorch:rocm6.0_ubuntu22.04_py3.10_pytorch_2.1.0

# Create a non-root user to handle permissions
RUN useradd -ms /bin/bash -u 1001 user
USER user
WORKDIR /home/user/app

# Copy the project files into the container
COPY --chown=user:user . .

# Set the entrypoint to the speedrun script
ENTRYPOINT ["./speedrun.sh"]