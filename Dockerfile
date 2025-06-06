FROM ubuntu:22.04

RUN apt-get update -y

RUN apt-get install -y python3-pip
RUN apt-get install -y \
    build-essential \
    curl \
    software-properties-common
#git

RUN  rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR /app

#RUN git clone https://github.com/jrmd24/DIT_DEVOPS.git .

COPY . .

RUN pip install -r requirements.txt

#EXPOSE 8501
EXPOSE 5000

HEALTHCHECK CMD curl --fail http://localhost:5000/_stcore/health

ENTRYPOINT ["flask", "--app", "ml_project_front", "run", "--host=0.0.0.0", "--port=5000"]
