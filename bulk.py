import os
import arxiv
import urllib.request


def download_arxiv_papers(dest_folder, max_results=100):
    # Search arxiv and get a list of papers in the cs.LG category
    papers = arxiv.Search(
        query="cat:cs.AI",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    ).results()

    for paper in papers:
        arxiv_id = paper.entry_id.split("/")[-1]
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        filename = f"{arxiv_id}.pdf"
        dest_path = os.path.join(dest_folder, filename)

        print(f"Downloading {arxiv_id}...")
        urllib.request.urlretrieve(pdf_url, dest_path)


if __name__ == "__main__":
    dest_folder = "inputs"

    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)

    download_arxiv_papers(dest_folder)
