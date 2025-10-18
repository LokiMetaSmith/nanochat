FROM rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.7.1

# Install system dependencies
RUN apt-get update && apt-get install -y git curl

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Create a working directory
WORKDIR /app

# Copy the project files
COPY . .

# Install Python dependencies
RUN uv pip install -e .[dev]

# Set the entrypoint to the speedrun script
ENTRYPOINT ["bash", "speedrun.sh"]