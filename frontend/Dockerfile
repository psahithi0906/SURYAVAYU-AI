FROM node:18

WORKDIR /frontend

RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-pip

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 3000

CMD ["npm", "start"]