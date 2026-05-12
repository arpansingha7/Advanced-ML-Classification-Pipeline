import logging
from src.pipeline_runner import PipelineRunner

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Welcome to the Advanced ML Pipeline!")
    
    runner = PipelineRunner()
    runner.run_all()

if __name__ == "__main__":
    main()
