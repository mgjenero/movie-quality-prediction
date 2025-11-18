FROM python:3.12.1-slim-bookworm

# Install `uv` package manager
RUN pip install uv uvicorn pandas fastapi scikit-learn

# Set the working directory to /app
WORKDIR /app

# Copy the necessary configuration files (pyproject.toml, uv.lock, .python-version)
COPY pyproject.toml uv.lock .python-version ./  

# Install the dependencies using `uv sync --locked` (installs dependencies listed in pyproject.toml and uv.lock)
RUN uv sync --locked

# Install `uvicorn` explicitly (just to be sure it's in the PATH)
RUN pip install uvicorn

# Copy the entire project into the container
COPY . /app

# Set the PYTHONPATH environment variable so Python knows where to look for your modules
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Expose the port FastAPI will use
EXPOSE 9696

# Set the entrypoint to run the FastAPI app using `uvicorn`
ENTRYPOINT ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "9696"]
