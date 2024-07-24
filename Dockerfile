FROM continuumio/miniconda3

WORKDIR /app
COPY environment.yml /app/environment.yml
RUN conda update -n base -c defaults conda
RUN conda env create --file environment.yml
RUN echo "source activate torch" > ~/.bashrc
# Ensure conda environment is activated:
SHELL ["conda", "run", "-n", "torch", "/bin/bash", "-c"]

RUN conda install -c conda-forge nodejs

# Copy the rest of the application code
COPY . /app

WORKDIR /app/ui
RUN npm install
RUN npm run build
EXPOSE 5000
WORKDIR /app
CMD ["conda", "run", "--no-capture-output", "-n", "torch", "python", "main.py"]