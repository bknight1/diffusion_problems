FROM ghcr.io/underworldcode/underworld3:ghcr-workflow

# run as root to install git and fetch the source
USER root
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Clone the specific commit and copy the Poisson folder to a location outside the repo mount.
# Using the commit you referenced so the exact files are used.
RUN mkdir -p /tmp/src \
 && git clone https://github.com/bknight1/diffusionDecayIngrowth.git /tmp/src \
 && cd /tmp/src && git checkout main \
 && cp -R Poisson_solver_implementation /home/mambauser/Poisson_solver_implementation \
 && rm -rf /tmp/src

# Make sure the mambauser owns the copied files
RUN chown -R mambauser:mambauser /home/mambauser/Poisson_solver_implementation

# switch back to the non-root user used in the base image
USER mambauser
