import arxiv
import pandas as pd
import datetime


def fetch_search_result(search_query, within_5years=False):
    five_years_ago = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=5*365)
    max_results = 50

    search = arxiv.Search(
        query=search_query,
        max_results=max_results * (3 if within_5years else 1),
    )
    titles = []
    absts = []
    urls = []
    years = []
    for result in search.results():
        if within_5years and result.published < five_years_ago:
            continue
        titles.append(result.title)
        absts.append(result.summary.replace('\n', ' '))
        urls.append(result.entry_id)
        years.append(result.published.year)
    num_results = len(titles)
    keywords = [search_query] * num_results
    rankings = list(range(1, num_results + 1))
    df = pd.DataFrame(data=dict(
        keyword=keywords[:max_results],
        site_name=titles[:max_results],
        URL=urls[:max_results],
        snippet=absts[:max_results],
        ranking=rankings[:max_results],
        year=years[:max_results],
    ))
    return df


if __name__ == '__main__':
    import time

    search_str = input("> ")

    start = time.time()
    df = fetch_search_result(search_str, True)
    duration = time.time() - start
    print(f"duration: {duration}s")
    df.to_csv(search_str + ".csv")
