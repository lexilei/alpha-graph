"""Run all data pipelines: python -m alpha_graph.data"""

from loguru import logger


def main():
    logger.info("=== Downloading SEC filings ===")
    from alpha_graph.data.filings import download_all as download_filings

    download_filings()

    logger.info("=== All data pipelines complete ===")


if __name__ == "__main__":
    main()
