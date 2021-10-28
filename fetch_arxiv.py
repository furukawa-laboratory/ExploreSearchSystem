import arxiv
import pandas as pd

def fetch_search_result(search_query):
    search = arxiv.Search(
        query=search_query,
        max_results=50,
    )
    titles = []
    absts = []
    urls = []
    for result in search.results():
        titles.append(result.title)
        absts.append(result.summary.replace('\n', ' '))
        urls.append(result.entry_id)
    num_results = len(titles)
    keywords = [search_query] * num_results
    rankings = list(range(1, num_results + 1))
    df = pd.DataFrame(data=dict(
        keyword=keywords,
        site_name=titles,
        URL=urls,
        snippet=absts,
        ranking=rankings,
    ))
    return df


if __name__ == '__main__':
    import time

    search_str = input("> ")

    start = time.time()
    df = fetch_search_result(search_str)
    duration = time.time() - start
    print(f"duration: {duration}s")
    df.to_csv(search_str + ".csv")
