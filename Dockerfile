#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

FROM python:3.12-slim

WORKDIR /app

# Install git (required for local mode to generate diffs). Not pinned so image builds on both Bookworm and Trixie base images.
# hadolint ignore=DL3008
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source
COPY src/ ./src/

# Ensure Python finds the action's src package when container runs with -w GITHUB_WORKSPACE
ENV PYTHONPATH=/app

# Support both CI and local modes by allowing arguments to be passed
ENTRYPOINT ["python", "-m", "src.main"]
