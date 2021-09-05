from urllib.parse import urlparse
import pickle
import logging

import requests


_LOGGER = logging.getLogger("web_helpers")

ACTUAL_SITES = {
    '2wsb.tv': 'wsbtv.com',
    'abcn.ws': 'abcnews.go.com',
    'abc7.la': 'abc7.com',
    'aol.it': 'aol.com',
    'a.msn.com': 'msn.com',
    'aje.io': 'aljazeera.com',
    'apne.ws': 'apnews.com',
    'amp.cnn.com': 'cnn.com',
    'amp.theguardian.com': 'theguardian.com',
    'bbc.in': 'bbc.com',
    'bbc.co.uk': 'bbc.com',
    'bloom.bg': 'bloomberg.com',
    'bos.gl': 'bostonglobe.com',
    'bzfd.it': 'buzzfeed.com',
    'cbsn.ws': 'cbsnews.com',
    'chn.ge': 'change.org',
    'cnb.cx': 'cnbc.com',
    'cnet.co': 'cnet.com',
    'cbsloc.al': 'cbslocal.com',
    'chng.it': 'change.org',
    'choice.npr.org': 'npr.org',
    'cnn.it': 'cnn.com',
    'dailym.ai': 'dailymail.co.uk',
    'dpo.st': 'denverpost.com',
    'econ.st': 'economist.com',
    'edition.cnn.com': 'cnn.com',
    'f-st.co': 'fastcompany.com',
    'f24.my': 'france24.com',
    'fxn.ws': 'foxnews.com',
    'jd.fo': 'forward.com',
    'gju.st': 'globaljustice.org.uk',
    'huffp.st': 'huffingtonpost.com',
    'huffpost.com': 'huffingtonpost.com',
    'i24ne.ws': 'i24news.tv',
    'ihe.art': 'iheart.com',
    'interc.pt': 'theintercept.com',
    'insider.foxnews.com': 'foxnews.com',
    'hill.cm': 'thehill.com',
    'hrld.us': 'miamiherald.com',
    'kptv.tv': 'kptv.com',
    'lat.ms': 'latimes.com',
    'lemde.fr': 'lemonde.fr',
    'lzne.ws': 'lifezette.com',
    'lnkd.in': 'linkedin.com',
    'm.jpost.com': 'jpost.com',
    'mag.time.com': 'time.com',
    'mol.im': 'dailymail.co.uk',
    'natl.io': 'nationalreview.com',
    'nbcnews.to': 'nbcnews.com',
    'newscienti.st': 'newscientist.com',
    'nytv.to': 'naytev.com',
    'on.rt.com': 'rt.com',
    'on.msnbc.com': 'msnbc.com',
    'nyp.st': 'nypost.com',
    'nyti.ms': 'nytimes.com',
    'nym.ag': 'nymag.com',
    'n.pr': 'npr.org',
    'politi.co': 'politico.com',
    'pops.ci': 'popsci.com',
    'read.bi': 'businessinsider.com',
    'reut.rs': 'reuters.com',
    'reut.tv': 'reuters.com',
    'reuters.tv': 'reuters.com',
    'rol.st': 'rollingstone.com',
    'sco.lt': 'scoop.it',
    'strib.mn': 'startribune.com',
    'tcrn.ch': 'techcrunch.com',
    'ti.me': 'time.com',
    'tnw.me': 'thenextweb.com',
    'tdig.it': 'truthdig.com',
    'usat.ly': 'usatoday.com',
    'wapo.st': 'washingtonpost.com',
    'washex.am': 'washingtonexaminer.com',
    'wb.md': 'webmd.com',
}


def get_site_from_url(url):
    if not url:
        return url
    netloc = urlparse(url).netloc.lower()
    netloc = netloc[netloc.startswith('www.') and 4:]
    if netloc in ACTUAL_SITES:
        return ACTUAL_SITES[netloc]
    return netloc


class UrlExpander:
    def __init__(self, url_map_pickle="url_map.pickle"):
        self.successful_requests_count = 0
        self.requests_count = 0
        self.duplicates = 0
        self.session = requests.Session()
        try:
            with open(url_map_pickle, 'rb') as f:
                self.url_map = pickle.load(f)
            _LOGGER.info('Loaded url map from [{}]'.format(url_map_pickle))
        except:
            self.url_map = dict()

    def pickle_url_map(self, url_map_pickle="url_map.pickle"):
        with open(url_map_pickle, 'wb') as f:
            pickle.dump(self.url_map, f, protocol=pickle.HIGHEST_PROTOCOL)

    def unshorten_url(self, url, ignore_sites=None, use_cache=False, timeout=5):
        if ignore_sites is None:
            ignore_sites = set()
    
        url_netloc = urlparse(url).netloc.lower()
        url_netloc = url_netloc[url_netloc.startswith('www.') and 4:]
        if url_netloc in ignore_sites:
            return url
    
        if use_cache and url in self.url_map:
            self.duplicates += 1
            return self.url_map[url]
    
        try:
            self.successful_requests_count += 1
            self.requests_count += 1
            r = self.session.head(url, timeout=timeout)
            r.raise_for_status()
            if 'Location' in r.headers:
                next_ = r.next.url
                self.url_map[url] = next_
                return next_
            else:
                self.url_map[url] = url
                return url
        except Exception as e:
            _LOGGER.debug(e)
            self.successful_requests_count -= 1
            return url
