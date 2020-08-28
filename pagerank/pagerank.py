import os
import random
import re
import sys
import copy

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    numOfLinks = len(corpus[page])
    numOfPages = len(corpus)
    distribution = dict()
    if numOfLinks != 0:
        pageProbability = (1 - damping_factor) / numOfPages
        linksProbabilities = damping_factor / numOfLinks + pageProbability
        for pagina in corpus:
            if pagina not in corpus[page]:
                distribution[pagina] = pageProbability
            else:
                distribution[pagina] = linksProbabilities
        return distribution
    else:
        pageProbability = (1 - damping_factor) / numOfPages
        for pagina in corpus:
            distribution[pagina] = pageProbability
        return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # KEEPING TRACK OF HOW MANY TIMES VISIT THE PAGES
    corpusPages = []
    trackOfVisits = dict()

    # EXTRACT A RANDOM PAGE FROM CORPUS FOR START AND SETS TRACK OF VISITS OF EACH PAGE = 0
    for page, links in corpus.items():
        corpusPages.append(page)
        trackOfVisits[page] = 0
    if corpusPages:
        randomPage = random.choice(corpusPages)
    else:
        print("corpusPages list is empty")
        raise Exception

    # FIRST SAMPLE
    distribution = transition_model(corpus, randomPage, damping_factor)
    keys = []
    values = []
    for page, value in distribution.items():
        keys.append(page)
        values.append(value)
    start = random.choices(keys, weights=values)
    mypage = start[0]
    trackOfVisits[mypage] = trackOfVisits[mypage] + 1

    # START OF N SAMPLES
    for k in range(n):
        keys = []
        values = []
        newDistribution = transition_model(corpus, start[0], damping_factor)
        for page, value in newDistribution.items():
            keys.append(page)
            values.append(value)
        start = random.choices(keys, weights=values)
        mypage = start[0]
        trackOfVisits[mypage] = trackOfVisits[mypage] + 1

    finalProb = dict()

    for page in corpusPages:
        finalProb[page] = trackOfVisits[page] / n

    print("Sum: ", sum(finalProb.values()))
    return finalProb


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # INITILIZE RANKS WITH 1 / N
    ranks = dict()
    N = len(corpus)
    for page, links in corpus.items():
        ranks[page] = 1 / N
    first = (1 - damping_factor) / N

    # ITERATES THROUGH PAGES UNTIL REACHES CONVERGENCE OF 0.001
    while True:
        limitsOfPages = copy.deepcopy(ranks)
        for page, links in corpus.items():
            # SETS SECOND = 0 WHEN CHANGING TO OTHER PAGE
            second = 0
            for supraPage, supraLinks in corpus.items():
                if supraLinks:
                    if page in supraLinks:
                        numOfLinks = len(corpus[supraPage])
                        second = second + ranks[supraPage] / numOfLinks
            probability = first + (damping_factor * second)
            ranks[page] = probability
            if abs(ranks[page] - limitsOfPages[page]) < 0.001:
                total = sum(ranks.values())
                # CALCULATES THE PROBABILITY OF PAGES BASED ON THE TOTAL SUMATORY OF PROBABILITIES
                for page, value in ranks.items():
                    ranks[page] = value / total
                print("Sum: ", sum(ranks.values()))
                return ranks


if __name__ == "__main__":
    main()
