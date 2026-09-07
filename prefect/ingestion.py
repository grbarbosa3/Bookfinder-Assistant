import os
from datetime import timedelta
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from prefect import flow, task

load_dotenv()

@task(retries=2, retry_delay_seconds=60)
def download_kaggle_dataset():

    os.environ["KAGGLE_USERNAME"] = os.getenv("ENV_KAGGLE_USERNAME", "")
    os.environ["KAGGLE_API_TOKEN"] = os.getenv("ENV_KAGGLE_API_TOKEN", "")
    
    api = KaggleApi()
    api.authenticate()

    dataset = 'bhavikm07/google-books-dataset'
    

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'data_kaggle')
    

    os.makedirs(output_dir, exist_ok=True)

    print(f"Download start...: {output_dir}")

    api.dataset_download_files(dataset, path=output_dir, unzip=True)
    
    arquivos_baixados = os.listdir(output_dir)
    print(f"Arquivos na pasta após download: {arquivos_baixados}")
    print('--> Finished <--')

@flow(name="Kaggle Retrieve Flow")
def kaggle_pipeline():
    download_kaggle_dataset()

if __name__ == "__main__":
    kaggle_pipeline()
