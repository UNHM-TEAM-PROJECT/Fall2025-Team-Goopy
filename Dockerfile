# build frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

# copy package files and install dependencies
COPY frontend/package.json frontend/package-lock.json ./
RUN npm install

# copy frontend source and build static export
COPY frontend/ ./
RUN npm run build

# setup backend + serve frontend
FROM python:3.10-slim
WORKDIR /app

# set public URL environment variable
ENV PUBLIC_URL=https://whitemount.sr.unh.edu/t3/

# install backend dependencies
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy backend code
COPY backend/ ./backend
COPY scrape/ ./scrape

# copy frontend build from the builder stage
COPY --from=frontend-builder /app/frontend ./frontend

# expose backend port only
EXPOSE 8003
ENV PYTHONUNBUFFERED=1

# start backend with uvicorn; frontend static files are served from FastAPI
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8003"]
