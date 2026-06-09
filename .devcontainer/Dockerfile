FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

WORKDIR /workspaces/open-spx

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt