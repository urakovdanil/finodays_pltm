FROM python:3.7 AS builder

COPY requirements.txt .

RUN pip install --user -r requirements.txt

FROM python:3.7-slim
WORKDIR /usr/src/app

COPY --from=builder /root/.local /root/.local
COPY . .

CMD ["python", "-u", "./main.py"]
