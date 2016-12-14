
# coding: utf-8

# In[ ]:

# first let's do all of the import statements
import requests
import urllib.request
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from string import punctuation
from heapq import nlargest
from math import log
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# In[ ]:

def getWashPostText(url,token):
    # THis function takes the URL of an article in the 
    # Washington Post, and then returns the article minus all 
    # of the crud - HTML, javascript etc. How? By searching for
    # everything that lies between the tags titled 'token'
    # Like most web-scraping, this will only work for urls where
    # we know the structure (eg say all articles in the WashPo are
    # enclosed in <article></article> tags). This will also change from
    # time to time as different HTML formats are employed in the website
    try:
        page = urllib.request.urlopen(url).read().decode('utf8')
    except:
        # if unable to download the URL, return title = None, article = None
        return (None,None)
    soup = BeautifulSoup(page)
    if soup is None:
        return (None,None)
    # If we are here, it means the error checks were successful, we were
    # able to parse the page
    text = ""
    if soup.find_all(token) is not None:
        # Search the page for whatever token demarcates the article
        # usually '<article></article>'
        text = ''.join(map(lambda p: p.text, soup.find_all(token)))
        # mush together all the text in the <article></article> tags
        soup2 = BeautifulSoup(text)
        # create a soup of the text within the <article> tags
        if soup2.find_all('p')!=[]:
            # now mush together the contents of what is in <p> </p> tags
            # within the <article>
            text = ''.join(map(lambda p: p.text, soup2.find_all('p')))
    return text, soup.title.text
    # what did we just do? Let's go through and understand
    # finally return the result tuple with the title and the body of the article
    


# In[ ]:

# Now we will do something very very similar, but this time for the New York Times
def getNYTText(url,token):
    response = requests.get(url)
    # THis is an alternative way to get the contents of a URL
    soup = BeautifulSoup(response.content)
    page = str(soup)
    title = soup.find('title').text
    mydivs = soup.findAll("p", {"class":"story-body-text story-content"})
    text = ''.join(map(lambda p:p.text, mydivs))
    return text, title
    # Notice again how important it is to know the structure of the page
    # we are seeking to scrape. If we did not know that articles in the NYT
    # come contained in these tags - an outer tag <p> and an inner tag
    # of class = story-body-text story-content, we would be unable to parse


# In[ ]:

# Ok! Now we have a way to extract the contents and title of an individual
# URL. Let's hook this up inside another function that will take the URL
# of an entire section of a newspaper - say the Technology or Sports section
# of a newspaper - and parse all of the URLs for articles linked off that
# section. 
# Btw, these sections also come with plenty of non-news links - 'about',
# how to syndicate etc, so we will employ a little hack - we will consider
# something to be a news article only if the url has a dateline. THis is 
# actually very safe - its pretty much the rule for articles to have a 
# date, and virtually all important newspapers mush this date into the URL.
def scrapeSource(url, magicFrag='2015',scraperFunction=getNYTText,token='None'):
    urlBodies = {}
    request = urllib.request.Request(url)
    response = urllib.request.urlopen(request)
    soup = BeautifulSoup(response)
    # we are set up with a Soup of the page - now find the links
    # Remember that links are always of the form 
    # <a href='link-url'> link-text </a>
    numErrors = 0
    for a in soup.findAll('a'):
        try:
            url = a['href']
            if( (url not in urlBodies) and 
               ((magicFrag is not None and magicFrag in url) 
               or magicFrag is None)):
                body = scraperFunction(url,token)
                # this line above is important - scraperFunction 
                # refers to the individual scraper function for the 
                # new york times or the washington post etc.
                if body and len(body) > 0:
                    urlBodies[url] = body
                print(url)
        except:
            numErrors += 1
            # plenty of parse errors happen - links might not
            # be external links, might be malformed and so on -
            # so don't mind if there are exceptions.
    return urlBodies


# In[ ]:

# Now for the frequency summarizer class - which we have encountered
# before. To quickly jog our memories - given an (title,article-body) tuple
# the frequency summarizer has easy ways to find the most 'important'
# sentences, and the most important words. How is 'important' defined?
# Important = most frequent, excluding 'stopwords' which are generic
# words like 'the' etc which can be ignored
class FrequencySummarizer:
    def __init__(self,min_cut=0.1,max_cut=0.9):
        # class constructor - takes in min and max cutoffs for 
        # frequency
        self._min_cut = min_cut
        self._max_cut = max_cut
        self._stopwords = set(stopwords.words('english') +
                              list(punctuation) +
                              [u"'s",'"'])
        # notice how the stopwords are a set, not a list. 
        # its easy to go from set to list and vice-versa
        # (simply use the set() and list() functions) - 
        # but conceptually sets are different from lists
        # because sets don't have an order to their elements
        # while lists do
    
    def _compute_frequencies(self,word_sent,customStopWords=None):
        freq = defaultdict(int)
        # we have encountered defaultdict objects before
        if customStopWords is None:
            stopwords = set(self._stopwords)
        else:
            stopwords = set(customStopWords).union(self._stopwords)
        for sentence in word_sent:
            for word in sentence:
                if word not in stopwords:
                    freq[word] += 1
        m = float(max(freq.values()))
        for word in list(freq.keys()):
            freq[word] = freq[word]/m
            if freq[word] >= self._max_cut or freq[word] <= self._min_cut:
                del freq[word]
        return freq
    
    def extractFeatures(self,article,n,customStopWords=None):
        # The article is passed in as a tuple (text, title)
        text = article[0]
        # extract the text
        title = article[1]
        # extract the title
        sentences = sent_tokenize(text)
        # split the text into sentences
        word_sent = [word_tokenize(s.lower()) for s in sentences]
        # split the sentences into words 
        self._freq = self._compute_frequencies(word_sent,customStopWords)
        # calculate the word frequencies using the member function above
        if n < 0:
            # how many features (words) to return? IF the user has
            # asked for a negative number, this is a sign that we don't
            # do any feature selection - we return ALL features
            # THis is feature extraction without any pruning, ie no
            # feature selection (beyond simply picking words as the features)
            return nlargest(len(self._freq_keys()),self._freq,key=self._freq.get)
        else:
            # if the calling function has asked for a subset then
            # return only the 'n' largest features - ie here the most
            # important words (important == frequent, barring stopwords)
            return nlargest(n,self._freq,key=self._freq.get)
        # let's summarize what we did here. 
    
    def extractRawFrequencies(self, article):
        # very similar, except that this method will return the 'raw'
        # frequencies - literally just the word counts
        text = article[0]
        title = article[1]
        sentences = sent_tokenize(text)
        word_sent = [word_tokenize(s.lower()) for s in sentences]
        freq = defaultdict(int)
        for s in word_sent:
            for word in s:
                if word not in self._stopwords:
                    freq[word] += 1
        return freq
    
    def summarize(self, article,n):
        text = article[0]
        title = article[1]
        sentences = sent_tokenize(text)
        word_sent = [word_tokenize(s.lower()) for s in sentences]
        self._freq = self._compute_frequencies(word_sent)
        ranking = defaultdict(int)
        for i,sentence in enumerate(word_sent):
            for word in sentence:
                if word in self._freq:
                    ranking[i] += self._freq[word]
        sentences_index = nlargest(n,ranking,key=ranking.get)

        return [sentences[j] for j in sentences_index]


# In[ ]:


urlWashingtonPostNonTech = "https://www.washingtonpost.com/sports"
urlNewYorkTimesNonTech = "https://www.nytimes.com/pages/sports/index.html"
urlWashingtonPostTech = "https://www.washingtonpost.com/business/technology"
urlNewYorkTimesTech = "http://www.nytimes.com/pages/technology/index.html"

washingtonPostTechArticles = scrapeSource(urlWashingtonPostTech,
                                          '2016',
                                         getWashPostText,
                                         'article') 
washingtonPostNonTechArticles = scrapeSource(urlWashingtonPostNonTech,
                                          '2016',
                                         getWashPostText,
                                         'article')
                
                
newYorkTimesTechArticles = scrapeSource(urlNewYorkTimesTech,
                                       '2016',
                                       getNYTText,
                                       None)
newYorkTimesNonTechArticles = scrapeSource(urlNewYorkTimesNonTech,
                                       '2016',
                                       getNYTText,
                                       None)


# In[ ]:

# Now let's collect these article summaries in an easy to classify form
articleSummaries = {}
for techUrlDictionary in [newYorkTimesTechArticles, washingtonPostTechArticles]:
    for articleUrl in techUrlDictionary:
        if techUrlDictionary[articleUrl][0] is not None:
            if len(techUrlDictionary[articleUrl][0]) > 0:
                fs = FrequencySummarizer()
                summary = fs.extractFeatures(techUrlDictionary[articleUrl],25)
                articleSummaries[articleUrl] = {'feature-vector': summary,
                                               'label': 'Tech'}
for nontechUrlDictionary in [newYorkTimesNonTechArticles, washingtonPostNonTechArticles]:
    for articleUrl in nontechUrlDictionary:
        if nontechUrlDictionary[articleUrl][0] is not None:
            if len(nontechUrlDictionary[articleUrl][0]) > 0:
                fs = FrequencySummarizer()
                summary = fs.extractFeatures(nontechUrlDictionary[articleUrl],25)
                articleSummaries[articleUrl] = {'feature-vector': summary,
                                               'label': 'Non-Tech'}


# In[ ]:

def getDoxyDonkeyText(testUrl,token):
    response = requests.get(testUrl)
    soup = BeautifulSoup(response.content)
    page = str(soup)
    title = soup.find("title").text
    mydivs = soup.findAll("div", {"class":token})
    text = ''.join(map(lambda p:p.text,mydivs))
    return text,title
    # our test instance, just like our training data, is nicely
    # setup as a (title,text) tuple

testUrl = "http://doxydonkey.blogspot.in"
testArticle = getDoxyDonkeyText(testUrl,"post-body")

fs = FrequencySummarizer()
testArticleSummary = fs.extractFeatures(testArticle, 25)


# In[ ]:

similarities = {}
for articleUrl in articleSummaries:
    oneArticleSummary = articleSummaries[articleUrl]['feature-vector']
    similarities[articleUrl] = len(set(testArticleSummary).intersection(set(oneArticleSummary)))

labels = defaultdict(int)    
knn = nlargest(5, similarities, key=similarities.get)
for oneNeighbor in knn:
    labels[articleSummaries[oneNeighbor]['label']] += 1

nlargest(1,labels,key=labels.get)


# In[ ]:

cumulativeRawFrequencies = {'Tech':defaultdict(int),'Non-Tech':defaultdict(int)}
trainingData = {'Tech':newYorkTimesTechArticles,'Non-Tech':newYorkTimesNonTechArticles}
for label in trainingData:
    for articleUrl in trainingData[label]:
        if len(trainingData[label][articleUrl][0]) > 0:
            fs = FrequencySummarizer()
            rawFrequencies = fs.extractRawFrequencies(trainingData[label][articleUrl])
            for word in rawFrequencies:
                cumulativeRawFrequencies[label][word] += rawFrequencies[word]


# In[ ]:

techiness = 1.0
nontechiness = 1.0
for word in testArticleSummary:
    # for each 'feature' of the test instance - 
    if word in cumulativeRawFrequencies['Tech']:
        techiness *= 1e3*cumulativeRawFrequencies['Tech'][word] / float(sum(cumulativeRawFrequencies['Tech'].values()))
        # we multiply the techiness by the probability of this word
        # appearing in a tech article (based on the training data)
    else:
        techiness /= 1e3
        # THis is worth paying attention to. If the word does not appear
        # in the tech articles of the training data at all,we could simply
        # set that probability to zero - in fact doing so is the 'correct'
        # way mathematically, because that way all of the probabilities would
        # sum to 1. But that would lead to 'snap' decisions since the techiness
        # would instantaneously become 0. To prevent this, we decide to take
        # the probability as some very small number (here 1 in 1000, which is 
        # actually not all that low)
    # Now the exact same deal- but for the nontechiness. We are intentionally
    # copy-pasting code (not a great software development practice) in order
    # to make the logic very clear. Ideally, we would have created a function
    # and called it twice rather than copy-pasting this code. In any event..
    if word in cumulativeRawFrequencies['Non-Tech']:
        nontechiness *= 1e3*cumulativeRawFrequencies['Non-Tech'][word] / float(sum(cumulativeRawFrequencies['Non-Tech'].values()))
        # we multiply the techiness by the probability of this word
        # appearing in a tech article (based on the training data)
    else:
        nontechiness /= 1e3

# we are almost done! Now we simply need to scale the techiness 
# and non-techiness by the probabilities of overall techiness and
# non-techiness. THis is simply the number of words in the tech and 
# non-tech articles respectively, as a proportion of the total number
# of words
techiness *= float(sum(cumulativeRawFrequencies['Tech'].values())) / (float(sum(cumulativeRawFrequencies['Tech'].values())) + float(sum(cumulativeRawFrequencies['Non-Tech'].values())))
nontechiness *= float(sum(cumulativeRawFrequencies['Non-Tech'].values())) / (float(sum(cumulativeRawFrequencies['Tech'].values())) + float(sum(cumulativeRawFrequencies['Non-Tech'].values())))
if techiness > nontechiness:
    label = 'Tech'
else:
    label = 'Non-Tech'
print(label, techiness, nontechiness)


# In[ ]:

def getAllDoxyDonkeyPosts(url,links):
    request = urllib.request.Request(url)
    response = urllib.request.urlopen(request)
    soup = BeautifulSoup(response)
    for a in soup.findAll('a'):
        try:
            url = a['href']
            title = a['title']
            if title == "Older Posts":
                print(title, url)
                links.append(url)
                getAllDoxyDonkeyPosts(url,links)
        except:
            title = ""
    return


# In[ ]:


blogUrl = "http://doxydonkey.blogspot.in"
links = []
getAllDoxyDonkeyPosts(blogUrl,links)
doxyDonkeyPosts = {}
for link in links:
    doxyDonkeyPosts[link] = getDoxyDonkeyText(link,'post-body')


documentCorpus = []
for onePost in doxyDonkeyPosts.values():
    documentCorpus.append(onePost[0])


# In[ ]:

vectorizer = TfidfVectorizer(max_df=0.5,min_df=2,stop_words='english')
X = vectorizer.fit_transform(documentCorpus)
km = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 100, n_init = 1, verbose = True)
km.fit(X)

keywords = {}
for i,cluster in enumerate(km.labels_):
    oneDocument = documentCorpus[i]
    fs = FrequencySummarizer()
    summary = fs.extractFeatures((oneDocument,""),
                                100,
                                [u"according",u"also",u"billion",u"like",u"new", u"one",u"year",u"first",u"last"])
    if cluster not in keywords:
        keywords[cluster] = set(summary)
    else:
        keywords[cluster] = keywords[cluster].intersection(set(summary))


# In[ ]:



